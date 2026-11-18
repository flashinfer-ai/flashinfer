#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <utility>
#include <vector>

namespace ffi = tvm::ffi;
namespace wmma = nvcuda::wmma;
namespace cg = cooperative_groups;

namespace {

constexpr int E = 128;
constexpr int TOPK = 6;
constexpr int H = 2688;
constexpr int I = 1856;
constexpr int THREADS = 256;
constexpr int NT = 32;
constexpr int DECODE_NT = 32;
constexpr int DECODE_NT_SMALL = 16;
constexpr int DECODE_NT_LARGE = 64;
constexpr int DECODE_THREADS_SMALL = 128;
constexpr int KT = 256;
constexpr int MT = 8;
constexpr int W1_NTILES = I / NT;
constexpr int W2_NTILES = H / NT;
constexpr int MMA_M = 16;
constexpr int PF_M = 64;
constexpr int MMA_N = 64;
constexpr int MMA_K = 64;
constexpr int GMMA_K = 128;
constexpr int GMMA_N1 = 64;
constexpr int GMMA_N2 = 128;
constexpr int MMA_WARPS = MMA_N / 16;

struct Workspace {
  void* ptr = nullptr;
  size_t bytes = 0;
  int device = -1;
  std::mutex mutex;
};

Workspace g_workspace;

inline size_t align_up(size_t x, size_t a = 256) { return (x + a - 1) & ~(a - 1); }

// Persistent-grid sizes depend on the launch device's SM count and occupancy;
// cache them per device so a value computed on one GPU is never reused on a
// smaller one (where it could exceed the resident-block capacity and leave
// blocks waiting at the grid barrier).
constexpr int kMaxCachedDevices = 64;

// Buffers retired by workspace growth or a device change stay alive for the
// process lifetime: a CUDA graph captured earlier may still replay kernels
// holding the old address. Growth is monotone along the warmup ladder, so
// the retained total stays bounded (under 2x the final capacity per device).
std::vector<void*> g_retired_workspaces;

void* ensure_workspace(size_t bytes, int device) {
  if (g_workspace.ptr != nullptr && (g_workspace.bytes < bytes || g_workspace.device != device)) {
    g_retired_workspaces.push_back(g_workspace.ptr);
    g_workspace.ptr = nullptr;
    g_workspace.bytes = 0;
  }
  if (g_workspace.ptr == nullptr) {
    cudaSetDevice(device);
    cudaError_t err = cudaMalloc(&g_workspace.ptr, bytes);
    TVM_FFI_ICHECK(err == cudaSuccess && g_workspace.ptr != nullptr)
        << "specialized MoE workspace cudaMalloc(" << bytes
        << ") failed: " << cudaGetErrorString(err);
    cudaError_t zerr = cudaMemset(g_workspace.ptr, 0, bytes);
    TVM_FFI_ICHECK(zerr == cudaSuccess)
        << "specialized MoE workspace cudaMemset(" << bytes
        << ") failed: " << cudaGetErrorString(zerr);
    g_workspace.bytes = bytes;
    g_workspace.device = device;
  }
  return g_workspace.ptr;
}

__device__ __forceinline__ float fp8_e4m3fn_to_float(uint8_t x) {
  const uint32_t sign = uint32_t(x & 0x80) << 24;
  const uint32_t mag = x & 0x7f;
  const uint32_t exp = mag >> 3;
  const uint32_t mant = mag & 7;
  if (mag == 0x7f) return __int_as_float(int(sign | 0x7fffffff));
  if (exp == 0) {
    float v = float(mant) * 0.001953125f;
    return sign ? -v : v;
  }
  return __uint_as_float(sign | ((exp + 120) << 23) | (mant << 20));
}

__device__ __forceinline__ float2 fp8_e4m3fnx2_to_float2(uint16_t x) {
  uint32_t bits;
  asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(bits) : "h"(x));
  union {
    uint32_t u;
    __half2 h;
  } cvt;
  cvt.u = bits;
  return __half22float2(cvt.h);
}

__device__ __forceinline__ float e2m1_to_float(uint32_t q) {
  const uint32_t a = q & 7;
  float v;
  if (a < 4) {
    v = 0.5f * float(a);
  } else {
    v = float((2 + (a & 1)) << ((a - 4) >> 1));
  }
  return (q & 8) ? -v : v;
}

__device__ __forceinline__ float e2m1_bits_to_float(uint32_t q) {
  const uint32_t sign = (q & 8) << 28;
  const uint32_t exp = (q >> 1) & 3;
  const uint32_t mant = q & 1;
  const uint32_t normal = sign | ((exp + 126) << 23) | (mant << 22);
  const uint32_t subnormal = sign | (mant ? 0x3f000000u : 0u);
  return __uint_as_float(exp ? normal : subnormal);
}

__device__ __forceinline__ __half2 e2m1x2_to_half2(uint8_t q) {
  uint32_t bits;
  const uint32_t src = q;
  asm volatile(
      "{\n"
      ".reg .b8 b0, b1, b2, b3;\n"
      "mov.b32 {b0, b1, b2, b3}, %1;\n"
      "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
      "}\n"
      : "=r"(bits)
      : "r"(src));
  union {
    uint32_t u;
    __half2 h;
  } cvt;
  cvt.u = bits;
  return cvt.h;
}

__device__ __forceinline__ uint32_t e2m1x2_to_bf16x2(uint8_t q) {
  uint32_t bits;
  const uint32_t src = q;
  asm volatile(
      "{\n"
      ".reg .b8 b0, b1, b2, b3;\n"
      "mov.b32 {b0, b1, b2, b3}, %1;\n"
      "cvt.rn.bf16x2.e2m1x2 %0, b0;\n"
      "}\n"
      : "=r"(bits)
      : "r"(src));
  return bits;
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int d = 16; d != 0; d >>= 1) x += __shfl_down_sync(0xffffffffu, x, d);
  return x;
}

__device__ __forceinline__ uint32_t fp4_byte_to_bf16x2(uint8_t q) {
  float2 f = make_float2(e2m1_to_float(q & 15), e2m1_to_float(q >> 4));
  union {
    __nv_bfloat162 b;
    uint32_t u;
  } cvt;
  cvt.b = __float22bfloat162_rn(f);
  return cvt.u;
}

__device__ __forceinline__ void mma_m16n8k16_bf16(float& d0, float& d1, float& d2, float& d3,
                                                  uint32_t a0, uint32_t a1, uint32_t a2,
                                                  uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void cp_async_16(void* dst, const void* src, int src_bytes) {
  const uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :
               : "r"(smem), "l"(src), "r"(src_bytes));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }

__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group 0;\n"); }

__global__ void count_routes_kernel(const int32_t* __restrict__ ids, int routes,
                                    int32_t* __restrict__ counts) {
  for (int r = int(blockIdx.x) * blockDim.x + threadIdx.x; r < routes;
       r += int(gridDim.x) * blockDim.x) {
    const int e = ids[r];
    if (unsigned(e) < E) atomicAdd(counts + e, 1);
  }
}

__global__ void prefix_routes_kernel(const int32_t* __restrict__ counts,
                                     int32_t* __restrict__ offsets, int32_t* __restrict__ cursor,
                                     int32_t* __restrict__ tiles1, int32_t* __restrict__ tiles2) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  int route_sum = 0;
  int tile1_sum = 0;
  int tile2_sum = 0;
#pragma unroll
  for (int e = 0; e < E; ++e) {
    offsets[e] = route_sum;
    cursor[e] = 0;
    tiles1[e] = tile1_sum;
    tiles2[e] = tile2_sum;
    const int mtiles = (counts[e] + MT - 1) / MT;
    route_sum += counts[e];
    tile1_sum += mtiles * W1_NTILES;
    tile2_sum += mtiles * W2_NTILES;
  }
  offsets[E] = route_sum;
  tiles1[E] = tile1_sum;
  tiles2[E] = tile2_sum;
}

__global__ void prefix_mma_routes_kernel(const int32_t* __restrict__ counts,
                                         int32_t* __restrict__ offsets,
                                         int32_t* __restrict__ cursor, int32_t* __restrict__ tiles1,
                                         int32_t* __restrict__ tiles2) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  int route_sum = 0;
  int tile1_sum = 0;
  int tile2_sum = 0;
#pragma unroll
  for (int e = 0; e < E; ++e) {
    offsets[e] = route_sum;
    cursor[e] = 0;
    tiles1[e] = tile1_sum;
    tiles2[e] = tile2_sum;
    const int mtiles = (counts[e] + PF_M - 1) / PF_M;
    route_sum += counts[e];
    tile1_sum += mtiles * (I / GMMA_N1);
    tile2_sum += mtiles * (H / GMMA_N2);
  }
  offsets[E] = route_sum;
  tiles1[E] = tile1_sum;
  tiles2[E] = tile2_sum;
}

__global__ void scatter_routes_kernel(const int32_t* __restrict__ ids, int routes,
                                      const int32_t* __restrict__ offsets,
                                      int32_t* __restrict__ cursor,
                                      int32_t* __restrict__ route_list) {
  for (int r = int(blockIdx.x) * blockDim.x + threadIdx.x; r < routes;
       r += int(gridDim.x) * blockDim.x) {
    const int e = ids[r];
    if (unsigned(e) < E) {
      const int p = atomicAdd(cursor + e, 1);
      route_list[offsets[e] + p] = r;
    }
  }
}

template <int K, int N, int KG>
__global__ __launch_bounds__(32, 16) void decode_fc1_mma_kernel(
    const __nv_bfloat16* __restrict__ input, const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales, const float* __restrict__ alpha,
    const int32_t* __restrict__ ids, int routes, __nv_bfloat16* __restrict__ intermediate) {
  const int lane = threadIdx.x;
  const int g = lane >> 2;
  const int q = lane & 3;
  const int mtile = int(blockIdx.x) / routes;
  const int r = int(blockIdx.x) - mtile * routes;
  const int e = ids[r];
  const int mbase = mtile * 16;
  const int m0 = mbase + g;
  const int m1 = m0 + 8;
  float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

  if (unsigned(e) < E) {
    const __nv_bfloat16* x = input + size_t(r / TOPK) * K;
    const size_t row0 = size_t(e) * N + m0;
    const size_t row1 = size_t(e) * N + m1;
    const float al = alpha[e];
    for (int k0 = 0; k0 < K; k0 += 16) {
      const size_t p0 = row0 * (K / 2) + (k0 >> 1) + q;
      const size_t p1 = row1 * (K / 2) + (k0 >> 1) + q;
      const uint32_t a0 = fp4_byte_to_bf16x2(packed[p0]);
      const uint32_t a1 = fp4_byte_to_bf16x2(packed[p1]);
      const uint32_t a2 = fp4_byte_to_bf16x2(packed[p0 + 4]);
      const uint32_t a3 = fp4_byte_to_bf16x2(packed[p1 + 4]);
      uint32_t b0 = 0, b1 = 0;
      if (g == 0) {
        b0 = *reinterpret_cast<const uint32_t*>(x + k0 + q * 2);
        b1 = *reinterpret_cast<const uint32_t*>(x + k0 + 8 + q * 2);
      }
      float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
      mma_m16n8k16_bf16(t0, t1, t2, t3, a0, a1, a2, a3, b0, b1);
      float s0 = 0.0f, s1 = 0.0f;
      if (q == 0) {
        const int kg = k0 >> 4;
        s0 = fp8_e4m3fn_to_float(scales[row0 * KG + kg]) * al;
        s1 = fp8_e4m3fn_to_float(scales[row1 * KG + kg]) * al;
      }
      s0 = __shfl_sync(0xffffffffu, s0, lane & ~3);
      s1 = __shfl_sync(0xffffffffu, s1, lane & ~3);
      c0 = fmaf(t0, s0, c0);
      c1 = fmaf(t1, s0, c1);
      c2 = fmaf(t2, s1, c2);
      c3 = fmaf(t3, s1, c3);
    }
  }

  if (q == 0) {
    float v0 = c0 < 0.0f ? 0.0f : c0;
    float v1 = c2 < 0.0f ? 0.0f : c2;
    intermediate[size_t(r) * N + m0] = __float2bfloat16_rn(v0 * v0);
    intermediate[size_t(r) * N + m1] = __float2bfloat16_rn(v1 * v1);
  }
}

template <int K, int N, int KG>
__global__ __launch_bounds__(32, 16) void decode_fc2_combine_mma_kernel(
    const __nv_bfloat16* __restrict__ intermediate, const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales, const float* __restrict__ alpha,
    const int32_t* __restrict__ ids, const float* __restrict__ topk_weights, int tokens,
    __nv_bfloat16* __restrict__ output) {
  const int lane = threadIdx.x;
  const int g = lane >> 2;
  const int q = lane & 3;
  const int mtile = int(blockIdx.x) / tokens;
  const int tok = int(blockIdx.x) - mtile * tokens;
  const int mbase = mtile * 16;
  const int m0 = mbase + g;
  const int m1 = m0 + 8;
  float total0 = 0.0f, total1 = 0.0f;

#pragma unroll
  for (int slot = 0; slot < TOPK; ++slot) {
    const int r = tok * TOPK + slot;
    const int e = ids[r];
    if (unsigned(e) >= E) continue;
    const __nv_bfloat16* x = intermediate + size_t(r) * K;
    const size_t row0 = size_t(e) * N + m0;
    const size_t row1 = size_t(e) * N + m1;
    const float al = alpha[e];
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    for (int k0 = 0; k0 < K; k0 += 16) {
      const size_t p0 = row0 * (K / 2) + (k0 >> 1) + q;
      const size_t p1 = row1 * (K / 2) + (k0 >> 1) + q;
      const uint32_t a0 = fp4_byte_to_bf16x2(packed[p0]);
      const uint32_t a1 = fp4_byte_to_bf16x2(packed[p1]);
      const uint32_t a2 = fp4_byte_to_bf16x2(packed[p0 + 4]);
      const uint32_t a3 = fp4_byte_to_bf16x2(packed[p1 + 4]);
      uint32_t b0 = 0, b1 = 0;
      if (g == 0) {
        b0 = *reinterpret_cast<const uint32_t*>(x + k0 + q * 2);
        b1 = *reinterpret_cast<const uint32_t*>(x + k0 + 8 + q * 2);
      }
      float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
      mma_m16n8k16_bf16(t0, t1, t2, t3, a0, a1, a2, a3, b0, b1);
      float s0 = 0.0f, s1 = 0.0f;
      if (q == 0) {
        const int kg = k0 >> 4;
        s0 = fp8_e4m3fn_to_float(scales[row0 * KG + kg]) * al;
        s1 = fp8_e4m3fn_to_float(scales[row1 * KG + kg]) * al;
      }
      s0 = __shfl_sync(0xffffffffu, s0, lane & ~3);
      s1 = __shfl_sync(0xffffffffu, s1, lane & ~3);
      c0 = fmaf(t0, s0, c0);
      c1 = fmaf(t1, s0, c1);
      c2 = fmaf(t2, s1, c2);
      c3 = fmaf(t3, s1, c3);
    }
    if (q == 0) {
      const float w = topk_weights[r];
      total0 = fmaf(c0, w, total0);
      total1 = fmaf(c2, w, total1);
    }
  }

  if (q == 0) {
    output[size_t(tok) * N + m0] = __float2bfloat16_rn(total0);
    output[size_t(tok) * N + m1] = __float2bfloat16_rn(total1);
  }
}

template <int K, int N, int KG, int NTILES, int TILE_N, bool FIRST, bool ORDERED = false,
          bool ROUTE_OUTPUT = false, bool GRID_2D = false>
__global__ __launch_bounds__(256, 2) void decode_group16_half_kernel(
    const __nv_bfloat16* __restrict__ input, const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales, const float* __restrict__ alpha,
    const int32_t* __restrict__ ids, const int32_t* __restrict__ route_order,
    const float* __restrict__ topk_weights, int routes, __nv_bfloat16* __restrict__ intermediate,
    float* __restrict__ output_acc) {
  constexpr int STAGE_K = 4096;
  constexpr int OPW = TILE_N / 8;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  int ntile;
  int ridx;
  if constexpr (GRID_2D) {
    ntile = int(blockIdx.y);
    ridx = int(blockIdx.x);
  } else {
    ntile = int(blockIdx.x) / routes;
    ridx = int(blockIdx.x) - ntile * routes;
  }
  const int r = ORDERED ? route_order[ridx] : ridx;
  const int e = ids[r];
  if (unsigned(e) >= E) {
    if constexpr (FIRST) {
      for (int j = threadIdx.x; j < TILE_N; j += blockDim.x) {
        intermediate[size_t(r) * N + ntile * TILE_N + j] = __float2bfloat16_rn(0.0f);
      }
    }
    return;
  }

  __shared__ __align__(16) __half sx[STAGE_K];
  float acc[OPW];
#pragma unroll
  for (int j = 0; j < OPW; ++j) acc[j] = 0.0f;

  const int input_row = FIRST ? (r / TOPK) : r;
  const int nbase = ntile * TILE_N + warp * OPW;
  constexpr int KHALF = K / 2;

  for (int kt = 0; kt < K; kt += STAGE_K) {
    const int this_k = min(STAGE_K, K - kt);
    for (int p = threadIdx.x; p < this_k / 2; p += blockDim.x) {
      const __nv_bfloat162 xb =
          *reinterpret_cast<const __nv_bfloat162*>(input + size_t(input_row) * K + kt + p * 2);
      const int group = p >> 3;
      const int in_group = p & 7;
      const int logical_segment = group * 2 + (in_group >> 2);
      const int physical_segment = logical_segment ^ (group & 7);
      const int physical_pair = physical_segment * 4 + (in_group & 3);
      reinterpret_cast<__half2*>(sx)[physical_pair] = __float22half2_rn(__bfloat1622float2(xb));
    }
    __syncthreads();

    const int groups = this_k / 16;
    for (int group_base = 0; group_base < groups; group_base += 32) {
      const int group = group_base + lane;
      if (group < groups) {
        __half2 xv[8];
#pragma unroll
        for (int p = 0; p < 8; ++p) {
          const int logical_segment = group * 2 + (p >> 2);
          const int physical_segment = logical_segment ^ (group & 7);
          xv[p] = reinterpret_cast<const __half2*>(sx)[physical_segment * 4 + (p & 3)];
        }

#pragma unroll
        for (int j = 0; j < OPW; ++j) {
          const int n = nbase + j;
          const size_t row = size_t(e) * N + n;
          const int kg = (kt >> 4) + group;
          const uint64_t q8 =
              *reinterpret_cast<const uint64_t*>(packed + row * KHALF + size_t(kg) * 8);
          __half2 hsum = __float2half2_rn(0.0f);
#pragma unroll
          for (int p = 0; p < 8; ++p) {
            hsum = __hfma2(xv[p], e2m1x2_to_half2(uint8_t(q8 >> (p * 8))), hsum);
          }
          const float2 pair = __half22float2(hsum);
          const float sf = fp8_e4m3fnx2_to_float2(uint16_t(scales[row * KG + kg])).x;
          acc[j] = fmaf(pair.x + pair.y, sf, acc[j]);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int j = 0; j < OPW; ++j) {
    float v = warp_sum(acc[j]);
    if (lane == 0) {
      const int n = nbase + j;
      v *= alpha[e];
      if constexpr (FIRST) {
        v = fmaxf(v, 0.0f);
        intermediate[size_t(r) * N + n] = __float2bfloat16_rn(v * v);
      } else {
        if constexpr (ROUTE_OUTPUT) {
          intermediate[size_t(r) * N + n] = __float2bfloat16_rn(v);
        } else {
          v *= topk_weights[r];
          atomicAdd(output_acc + size_t(r / TOPK) * H + n, v);
        }
      }
    }
  }
}

template <int NTILES>
__device__ __forceinline__ int find_expert(int block, const int32_t* tiles) {
  int lo = 0;
  int hi = E;
#pragma unroll 8
  for (int it = 0; it < 8; ++it) {
    const int mid = (lo + hi) >> 1;
    if (tiles[mid + 1] <= block)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

template <int K, int N, int KG, int NTILES, bool FIRST>
__global__ __launch_bounds__(128) void grouped_wmma64_kernel(
    const __nv_bfloat16* __restrict__ input, const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales, const float* __restrict__ alpha,
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ tiles,
    const int32_t* __restrict__ route_list, __nv_bfloat16* __restrict__ output) {
  if (int(blockIdx.x) >= tiles[E]) return;
  const int e = find_expert<NTILES>(int(blockIdx.x), tiles);
  const int local = int(blockIdx.x) - tiles[e];
  const int ntile = local % NTILES;
  const int m0 = (local / NTILES) * MMA_M;
  const int valid_m = min(MMA_M, offsets[e + 1] - offsets[e] - m0);
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  __shared__ __align__(16) __nv_bfloat16 as[MMA_M * MMA_K];
  __shared__ __align__(16) __nv_bfloat16 bs[MMA_N * MMA_K];
  __shared__ __align__(16) float cs[MMA_WARPS * MMA_M * 16];
  __shared__ int sroutes[MMA_M];
  __shared__ int srows[MMA_M];
  if (tid < MMA_M) {
    if (tid < valid_m) {
      const int grouped_row = offsets[e] + m0 + tid;
      const int r = route_list[grouped_row];
      sroutes[tid] = r;
      srows[tid] = FIRST ? (r / TOPK) : grouped_row;
    } else {
      sroutes[tid] = -1;
      srows[tid] = -1;
    }
  }
  __syncthreads();

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
  wmma::fill_fragment(cf, 0.0f);

  constexpr int KHALF = K / 2;
  const float al = alpha[e];
  for (int k0 = 0; k0 < K; k0 += MMA_K) {
    for (int x = tid; x < MMA_M * MMA_K; x += blockDim.x) {
      const int m = x / MMA_K;
      const int k = x - m * MMA_K;
      if (m < valid_m) {
        as[x] = input[size_t(srows[m]) * K + k0 + k];
      } else {
        as[x] = __float2bfloat16_rn(0.0f);
      }
    }

    const int pair = lane & 7;
    const int row_in_quad = lane >> 3;
#pragma unroll
    for (int rb = 0; rb < 16; rb += 4) {
      const int nl = warp * 16 + rb + row_in_quad;
      const int n = ntile * MMA_N + nl;
      const size_t row = size_t(e) * N + n;
#pragma unroll
      for (int sg = 0; sg < MMA_K / 16; ++sg) {
        const uint8_t q = packed[row * KHALF + (k0 >> 1) + sg * 8 + pair];
        float sf = 0.0f;
        if (pair == 0) {
          sf = fp8_e4m3fn_to_float(scales[row * KG + (k0 >> 4) + sg]) * al;
        }
        sf = __shfl_sync(0xffffffffu, sf, lane & ~7);
        const int dst = nl * MMA_K + sg * 16 + pair * 2;
        bs[dst] = __float2bfloat16_rn(e2m1_bits_to_float(q & 15) * sf);
        bs[dst + 1] = __float2bfloat16_rn(e2m1_bits_to_float(q >> 4) * sf);
      }
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < MMA_K; kk += 16) {
      wmma::load_matrix_sync(af, as + kk, MMA_K);
      wmma::load_matrix_sync(bf, bs + warp * 16 * MMA_K + kk, MMA_K);
      wmma::mma_sync(cf, af, bf, cf);
    }
    __syncthreads();
  }

  float* warp_cs = cs + warp * MMA_M * 16;
  wmma::store_matrix_sync(warp_cs, cf, 16, wmma::mem_row_major);
  __syncwarp();
  for (int x = lane; x < MMA_M * 16; x += 32) {
    const int m = x >> 4;
    const int n = x & 15;
    if (m < valid_m) {
      const int grouped_row = offsets[e] + m0 + m;
      float v = warp_cs[x];
      if constexpr (FIRST) {
        v = fmaxf(v, 0.0f);
        output[size_t(grouped_row) * N + ntile * MMA_N + warp * 16 + n] =
            __float2bfloat16_rn(v * v);
      } else {
        const int r = sroutes[m];
        output[size_t(r) * N + ntile * MMA_N + warp * 16 + n] = __float2bfloat16_rn(v);
      }
    }
  }
}

template <int K, int N, int KG, int NTILES, int TILE_N, int BLOCK_THREADS, bool FIRST>
__global__ __launch_bounds__(BLOCK_THREADS, 1536 / BLOCK_THREADS) void grouped_mma_bf16_kernel(
    const __nv_bfloat16* __restrict__ input, const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales, const float* __restrict__ alpha,
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ tiles,
    const int32_t* __restrict__ route_list, __nv_bfloat16* __restrict__ output) {
  if (int(blockIdx.x) >= tiles[E]) return;
  const int e = find_expert<NTILES>(int(blockIdx.x), tiles);
  const int local = int(blockIdx.x) - tiles[e];
  const int ntile = local % NTILES;
  const int m0 = (local / NTILES) * MMA_M;
  const int valid_m = min(MMA_M, offsets[e + 1] - offsets[e] - m0);
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int q = lane & 3;

  __shared__ __align__(16) uint32_t as[MMA_M][GMMA_K / 2];
  __shared__ __align__(16) uint32_t bs[TILE_N][GMMA_K / 8];
  __shared__ __align__(16) uint32_t ss[TILE_N][GMMA_K / 64];
  __shared__ int sroutes[MMA_M];

  if (tid < MMA_M) {
    const int grouped_row = offsets[e] + m0 + tid;
    sroutes[tid] = tid < valid_m ? route_list[grouped_row] : -1;
  }
  __syncthreads();

  float acc[2][4];
#pragma unroll
  for (int atom = 0; atom < 2; ++atom) {
#pragma unroll
    for (int x = 0; x < 4; ++x) acc[atom][x] = 0.0f;
  }

  constexpr int KHALF = K / 2;
  for (int k0 = 0; k0 < K; k0 += GMMA_K) {
    const int stage_k = min(GMMA_K, K - k0);
    for (int idx = tid; idx < MMA_M * (GMMA_K / 8); idx += blockDim.x) {
      const int m = idx / (GMMA_K / 8);
      const int chunk = idx - m * (GMMA_K / 8);
      const int r = sroutes[m];
      const void* src = input;
      int src_bytes = 0;
      if (r >= 0 && chunk * 8 < stage_k) {
        const int input_row = FIRST ? (r / TOPK) : (offsets[e] + m0 + m);
        src = input + size_t(input_row) * K + k0 + chunk * 8;
        src_bytes = 16;
      }
      const int pair = chunk * 4;
      cp_async_16(&as[m][pair ^ ((m & 7) << 2)], src, src_bytes);
    }

    for (int idx = tid; idx < TILE_N * (GMMA_K / 32); idx += blockDim.x) {
      const int nr = idx / (GMMA_K / 32);
      const int chunk = idx - nr * (GMMA_K / 32);
      const int n = ntile * TILE_N + nr;
      const size_t row = size_t(e) * N + n;
      const void* src = packed;
      int src_bytes = 0;
      if (chunk * 32 < stage_k) {
        src = packed + row * KHALF + (k0 >> 1) + chunk * 16;
        src_bytes = 16;
      }
      const int physical_segment = chunk ^ ((nr >> 1) & 3);
      cp_async_16(&bs[nr][physical_segment * 4], src, src_bytes);
    }
    cp_async_commit();
    for (int idx = tid; idx < TILE_N * (GMMA_K / 64); idx += blockDim.x) {
      const int nr = idx / (GMMA_K / 64);
      const int word = idx - nr * (GMMA_K / 64);
      const int n = ntile * TILE_N + nr;
      const size_t row = size_t(e) * N + n;
      uint32_t sv = 0;
      if (word * 64 < stage_k) {
        sv = *reinterpret_cast<const uint32_t*>(scales + row * KG + (k0 >> 4) + word * 4);
      }
      ss[nr][word] = sv;
    }
    cp_async_wait();
    __syncthreads();

#pragma unroll 2
    for (int sk = 0; sk < GMMA_K / 16; ++sk) {
      const int pair0 = sk * 8 + q;
      const int pair1 = pair0 + 4;
      const uint32_t a0 = as[g][pair0 ^ ((g & 7) << 2)];
      const uint32_t a1 = as[g + 8][pair0 ^ ((g & 7) << 2)];
      const uint32_t a2 = as[g][pair1 ^ ((g & 7) << 2)];
      const uint32_t a3 = as[g + 8][pair1 ^ ((g & 7) << 2)];

#pragma unroll
      for (int atom = 0; atom < 2; ++atom) {
        const int nr = warp * 16 + atom * 8 + g;
        const int physical_segment = (sk >> 1) ^ ((nr >> 1) & 3);
        const int word = physical_segment * 4 + (sk & 1) * 2;
        const uint32_t p0 = bs[nr][word];
        const uint32_t p1 = bs[nr][word + 1];
        const uint32_t b0 = e2m1x2_to_bf16x2(uint8_t(p0 >> (q * 8)));
        const uint32_t b1 = e2m1x2_to_bf16x2(uint8_t(p1 >> (q * 8)));
        float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
        mma_m16n8k16_bf16(t0, t1, t2, t3, a0, a1, a2, a3, b0, b1);

        const int nr0 = warp * 16 + atom * 8 + q * 2;
        const int shift = (sk & 3) * 8;
        const uint16_t sp = uint16_t(uint8_t(ss[nr0][sk >> 2] >> shift)) |
                            (uint16_t(uint8_t(ss[nr0 + 1][sk >> 2] >> shift)) << 8);
        const float2 sf = fp8_e4m3fnx2_to_float2(sp);
        acc[atom][0] = fmaf(t0, sf.x, acc[atom][0]);
        acc[atom][1] = fmaf(t1, sf.y, acc[atom][1]);
        acc[atom][2] = fmaf(t2, sf.x, acc[atom][2]);
        acc[atom][3] = fmaf(t3, sf.y, acc[atom][3]);
      }
    }
    __syncthreads();
  }

  const float al = alpha[e];
#pragma unroll
  for (int atom = 0; atom < 2; ++atom) {
    const int n0 = ntile * TILE_N + warp * 16 + atom * 8 + q * 2;
    const int rows[2] = {g, g + 8};
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
      const int m = rows[rr];
      if (m < valid_m) {
        const int grouped_row = offsets[e] + m0 + m;
        const int r = sroutes[m];
        float v0 = acc[atom][rr * 2] * al;
        float v1 = acc[atom][rr * 2 + 1] * al;
        if constexpr (FIRST) {
          v0 = fmaxf(v0, 0.0f);
          v1 = fmaxf(v1, 0.0f);
          *reinterpret_cast<__nv_bfloat162*>(output + size_t(grouped_row) * N + n0) =
              __floats2bfloat162_rn(v0 * v0, v1 * v1);
        } else {
          *reinterpret_cast<__nv_bfloat162*>(output + size_t(r) * N + n0) =
              __floats2bfloat162_rn(v0, v1);
        }
      }
    }
  }
}

template <int K, int N, int KG, int NTILES, int TILE_N, bool FIRST, bool ROUTE_OUTPUT = false>
__global__ __launch_bounds__(256, 2) void grouped_group16_half_kernel(
    const __nv_bfloat16* __restrict__ input, const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales, const float* __restrict__ alpha,
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ tiles,
    const int32_t* __restrict__ route_list, const float* __restrict__ topk_weights,
    __nv_bfloat16* __restrict__ intermediate, float* __restrict__ output_acc) {
  constexpr int STAGE_K = 512;
  constexpr int OPW = TILE_N / 8;
  if (int(blockIdx.x) >= tiles[E]) return;
  const int e = find_expert<NTILES>(int(blockIdx.x), tiles);
  const int local = int(blockIdx.x) - tiles[e];
  const int ntile = local % NTILES;
  const int m0 = (local / NTILES) * MT;
  const int valid_m = min(MT, offsets[e + 1] - offsets[e] - m0);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  __shared__ __align__(16) __half sx[MT][STAGE_K];
  __shared__ int sroutes[MT];
  if (threadIdx.x < MT) {
    sroutes[threadIdx.x] = threadIdx.x < valid_m ? route_list[offsets[e] + m0 + threadIdx.x] : -1;
  }
  __syncthreads();

  float acc[OPW][MT];
#pragma unroll
  for (int j = 0; j < OPW; ++j)
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[j][m] = 0.0f;

  const int nbase = ntile * TILE_N + warp * OPW;
  constexpr int KHALF = K / 2;
  for (int kt = 0; kt < K; kt += STAGE_K) {
    const int this_k = min(STAGE_K, K - kt);
    const int pairs = this_k / 2;
    for (int idx = threadIdx.x; idx < MT * pairs; idx += blockDim.x) {
      const int m = idx / pairs;
      const int p = idx - m * pairs;
      const int r = sroutes[m];
      __half2 xh = __float2half2_rn(0.0f);
      if (r >= 0) {
        const int input_row = FIRST ? (r / TOPK) : r;
        const __nv_bfloat162 xb =
            *reinterpret_cast<const __nv_bfloat162*>(input + size_t(input_row) * K + kt + p * 2);
        xh = __float22half2_rn(__bfloat1622float2(xb));
      }
      reinterpret_cast<__half2*>(sx[m])[p] = xh;
    }
    __syncthreads();

    const int groups = this_k / 16;
    if (lane < groups) {
#pragma unroll
      for (int j = 0; j < OPW; ++j) {
        const int n = nbase + j;
        const size_t row = size_t(e) * N + n;
        const int kg = (kt >> 4) + lane;
        const uint64_t q8 =
            *reinterpret_cast<const uint64_t*>(packed + row * KHALF + size_t(kg) * 8);
        __half2 hsum[MT];
#pragma unroll
        for (int m = 0; m < MT; ++m) hsum[m] = __float2half2_rn(0.0f);
#pragma unroll
        for (int p = 0; p < 8; ++p) {
          const __half2 qh = e2m1x2_to_half2(uint8_t(q8 >> (p * 8)));
#pragma unroll
          for (int m = 0; m < MT; ++m) {
            hsum[m] = __hfma2(reinterpret_cast<const __half2*>(sx[m])[lane * 8 + p], qh, hsum[m]);
          }
        }
        const float sf = fp8_e4m3fn_to_float(scales[row * KG + kg]);
#pragma unroll
        for (int m = 0; m < MT; ++m) {
          const float2 pair = __half22float2(hsum[m]);
          acc[j][m] = fmaf(pair.x + pair.y, sf, acc[j][m]);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int j = 0; j < OPW; ++j) {
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      float v = warp_sum(acc[j][m]);
      if (lane == 0 && m < valid_m) {
        const int r = sroutes[m];
        const int n = nbase + j;
        v *= alpha[e];
        if constexpr (FIRST) {
          v = fmaxf(v, 0.0f);
          intermediate[size_t(r) * I + n] = __float2bfloat16_rn(v * v);
        } else if constexpr (ROUTE_OUTPUT) {
          intermediate[size_t(r) * H + n] = __float2bfloat16_rn(v);
        } else {
          v *= topk_weights[r];
          atomicAdd(output_acc + size_t(r / TOPK) * H + n, v);
        }
      }
    }
  }
}

template <int K, int N, int KG, int NTILES, int TILE_N, int BLOCK_THREADS, bool GROUPED,
          bool SECOND>
__global__ __launch_bounds__(BLOCK_THREADS, 2) void moe_fc_kernel(
    const __nv_bfloat16* __restrict__ input, const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales, const float* __restrict__ alpha,
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ tiles,
    const int32_t* __restrict__ route_list, const float* __restrict__ topk_weights, int routes,
    __nv_bfloat16* __restrict__ intermediate, float* __restrict__ output_acc) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  int e;
  int m0;
  int ntile;
  if constexpr (GROUPED) {
    if (int(blockIdx.x) >= tiles[E]) return;
    e = find_expert<NTILES>(int(blockIdx.x), tiles);
    const int local = int(blockIdx.x) - tiles[e];
    ntile = local % NTILES;
    m0 = (local / NTILES) * MT;
  } else {
    const int r = int(blockIdx.x) / NTILES;
    if (r >= routes) return;
    e = reinterpret_cast<const int32_t*>(offsets)[r];
    if (unsigned(e) >= E) return;
    m0 = r;
    ntile = int(blockIdx.x) - r * NTILES;
  }

  __shared__ __align__(16) __nv_bfloat16 sx[MT][KT];
  __shared__ int sroutes[MT];
  constexpr int MVAL = GROUPED ? MT : 1;
  const int valid_m = GROUPED ? min(MT, offsets[e + 1] - offsets[e] - m0) : 1;

  if (threadIdx.x < MVAL) {
    if (threadIdx.x < valid_m) {
      sroutes[threadIdx.x] = GROUPED ? route_list[offsets[e] + m0 + threadIdx.x] : m0;
    } else {
      sroutes[threadIdx.x] = -1;
    }
  }
  __syncthreads();

  constexpr int OPW = TILE_N / (BLOCK_THREADS / 32);
  float acc[OPW][MVAL];
#pragma unroll
  for (int j = 0; j < OPW; ++j)
#pragma unroll
    for (int m = 0; m < MVAL; ++m) acc[j][m] = 0.0f;

  const int nbase = ntile * TILE_N + warp * OPW;
  constexpr int KHALF = K / 2;

  for (int kt = 0; kt < K; kt += KT) {
    const int this_k = min(KT, K - kt);
    for (int idx = threadIdx.x; idx < MVAL * this_k; idx += blockDim.x) {
      const int m = idx / this_k;
      const int k = idx - m * this_k;
      const int r = sroutes[m];
      if (r >= 0) {
        const int row = SECOND ? r : (r / TOPK);
        sx[m][k] = input[size_t(row) * K + kt + k];
      } else {
        sx[m][k] = __float2bfloat16(0.0f);
      }
    }
    __syncthreads();

    const int substeps = this_k / 64;
#pragma unroll
    for (int sub = 0; sub < KT / 64; ++sub) {
      if (sub >= substeps) break;
      const int klocal = sub * 64 + lane * 2;
      const int kb = (kt + klocal) >> 1;
      float2 xv[MVAL];
#pragma unroll
      for (int m = 0; m < MVAL; ++m) {
        xv[m] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&sx[m][klocal]));
      }

#pragma unroll
      for (int j = 0; j < OPW; ++j) {
        const int n = nbase + j;
        const size_t woff = (size_t(e) * N + n) * KHALF + kb;
        const uint32_t q = packed[woff];
        float sf = 0.0f;
        if ((lane & 7) == 0) {
          const int g = (kt + klocal) >> 4;
          sf = fp8_e4m3fn_to_float(scales[(size_t(e) * N + n) * KG + g]);
        }
        sf = __shfl_sync(0xffffffffu, sf, lane & ~7);
        const float ql = e2m1_to_float(q & 15);
        const float qh = e2m1_to_float(q >> 4);
        const float wl = ql * sf;
        const float wh = qh * sf;
#pragma unroll
        for (int m = 0; m < MVAL; ++m) {
          if (m < valid_m) {
            acc[j][m] = fmaf(xv[m].x, wl, acc[j][m]);
            acc[j][m] = fmaf(xv[m].y, wh, acc[j][m]);
          }
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int j = 0; j < OPW; ++j) {
#pragma unroll
    for (int m = 0; m < MVAL; ++m) {
      float v = warp_sum(acc[j][m]);
      if (lane == 0 && m < valid_m) {
        const int n = nbase + j;
        const int r = sroutes[m];
        v *= alpha[e];
        if constexpr (SECOND) {
          const int tok = r / TOPK;
          v *= topk_weights[r];
          atomicAdd(output_acc + size_t(tok) * H + n, v);
        } else {
          v = fmaxf(v, 0.0f);
          intermediate[size_t(r) * I + n] = __float2bfloat16_rn(v * v);
        }
      }
    }
  }
}

__global__ void convert_output_kernel(const float* __restrict__ x, __nv_bfloat16* __restrict__ out,
                                      int n) {
  for (int i = int(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += int(gridDim.x) * blockDim.x) {
    out[i] = __float2bfloat16_rn(x[i]);
  }
}

__global__ void combine_route_output_kernel(const __nv_bfloat16* __restrict__ route_out,
                                            const int32_t* __restrict__ ids,
                                            const float* __restrict__ weights,
                                            __nv_bfloat16* __restrict__ out, int tokens) {
  constexpr int H2 = H / 2;
  const int total = tokens * H2;
  for (int x = int(blockIdx.x) * blockDim.x + threadIdx.x; x < total;
       x += int(gridDim.x) * blockDim.x) {
    const int tok = x / H2;
    const int pair_col = x - tok * H2;
    float2 sum = make_float2(0.0f, 0.0f);
#pragma unroll
    for (int slot = 0; slot < TOPK; ++slot) {
      const int r = tok * TOPK + slot;
      if (unsigned(ids[r]) < E) {
        const float2 v = __bfloat1622float2(
            reinterpret_cast<const __nv_bfloat162*>(route_out)[size_t(r) * H2 + pair_col]);
        const float w = weights[r];
        sum.x = fmaf(v.x, w, sum.x);
        sum.y = fmaf(v.y, w, sum.y);
      }
    }
    reinterpret_cast<__nv_bfloat162*>(out)[x] = __floats2bfloat162_rn(sum.x, sum.y);
  }
}

__device__ __forceinline__ uint32_t e4m3_bcast_bf16x2(uint32_t b) {
  uint32_t bits;
  const uint16_t src = uint16_t((b & 0xFFu) * 0x0101u);
  asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;\n" : "=r"(bits) : "h"(src));
  return bits;
}

__device__ __forceinline__ uint32_t mul_bf16x2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm volatile("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

__device__ __forceinline__ uint32_t fma_bf16x2(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

__device__ __forceinline__ void e2m1x8_to_bf16x2x4(uint32_t w, uint32_t* out) {
#pragma unroll
  for (int p = 0; p < 4; ++p) out[p] = e2m1x2_to_bf16x2(uint8_t(w >> (p * 8)));
}

__device__ __forceinline__ float sum_bf16x2(uint32_t v) {
  return __uint_as_float(v << 16) + __uint_as_float(v & 0xffff0000u);
}

// Grouped BF16 mma.sync GEMM with on-the-fly NVFP4 dequant.  The group scale is
// folded into the B fragment (exact: 2-bit e2m1 x 4-bit e4m3 fits bf16's mantissa),
// so one accumulator chain spans the whole K loop and MM rows share each B tile.
template <int K, int N, int KG, int NTILES, int TILE_N, int BLOCK_THREADS, bool FIRST, int MM>
__global__ __launch_bounds__(BLOCK_THREADS) void grouped_mma2_kernel(
    const __nv_bfloat16* __restrict__ input, const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales, const float* __restrict__ alpha,
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ tiles,
    const int32_t* __restrict__ route_list, __nv_bfloat16* __restrict__ output) {
  if (int(blockIdx.x) >= tiles[E]) return;
  const int e = find_expert<NTILES>(int(blockIdx.x), tiles);
  const int local = int(blockIdx.x) - tiles[e];
  const int ntile = local % NTILES;
  const int m0 = (local / NTILES) * MM;
  const int valid_m = min(MM, offsets[e + 1] - offsets[e] - m0);
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int q = lane & 3;
  constexpr int MATOM = MM / 16;
  constexpr int KHALF = K / 2;

  __shared__ __align__(16) uint32_t as[MM][GMMA_K / 2];
  __shared__ __align__(16) uint32_t bs[TILE_N][GMMA_K / 8];
  __shared__ __align__(16) uint32_t ss[TILE_N][GMMA_K / 64];
  __shared__ int sroutes[MM];

  for (int i = tid; i < MM; i += BLOCK_THREADS) {
    const int grouped_row = offsets[e] + m0 + i;
    sroutes[i] = i < valid_m ? route_list[grouped_row] : -1;
  }
  __syncthreads();

  float acc[MATOM][2][4];
#pragma unroll
  for (int ma = 0; ma < MATOM; ++ma)
#pragma unroll
    for (int atom = 0; atom < 2; ++atom)
#pragma unroll
      for (int x = 0; x < 4; ++x) acc[ma][atom][x] = 0.0f;

  for (int k0 = 0; k0 < K; k0 += GMMA_K) {
    const int stage_k = min(GMMA_K, K - k0);
    for (int idx = tid; idx < MM * (GMMA_K / 8); idx += BLOCK_THREADS) {
      const int m = idx / (GMMA_K / 8);
      const int chunk = idx - m * (GMMA_K / 8);
      const int r = sroutes[m];
      const void* src = input;
      int src_bytes = 0;
      if (r >= 0 && chunk * 8 < stage_k) {
        const int input_row = FIRST ? (r / TOPK) : (offsets[e] + m0 + m);
        src = input + size_t(input_row) * K + k0 + chunk * 8;
        src_bytes = 16;
      }
      const int pair = chunk * 4;
      cp_async_16(&as[m][pair ^ ((m & 7) << 2)], src, src_bytes);
    }
    for (int idx = tid; idx < TILE_N * (GMMA_K / 32); idx += BLOCK_THREADS) {
      const int nr = idx / (GMMA_K / 32);
      const int chunk = idx - nr * (GMMA_K / 32);
      const int n = ntile * TILE_N + nr;
      const size_t row = size_t(e) * N + n;
      const void* src = packed;
      int src_bytes = 0;
      if (chunk * 32 < stage_k) {
        src = packed + row * KHALF + (k0 >> 1) + chunk * 16;
        src_bytes = 16;
      }
      const int physical_segment = chunk ^ ((nr >> 1) & 3);
      cp_async_16(&bs[nr][physical_segment * 4], src, src_bytes);
    }
    cp_async_commit();
    for (int idx = tid; idx < TILE_N * (GMMA_K / 64); idx += BLOCK_THREADS) {
      const int nr = idx / (GMMA_K / 64);
      const int word = idx - nr * (GMMA_K / 64);
      const int n = ntile * TILE_N + nr;
      const size_t row = size_t(e) * N + n;
      uint32_t sv = 0;
      if (word * 64 < stage_k) {
        sv = *reinterpret_cast<const uint32_t*>(scales + row * KG + (k0 >> 4) + word * 4);
      }
      ss[nr][word] = sv;
    }
    cp_async_wait();
    __syncthreads();

#pragma unroll 2
    for (int sk = 0; sk < GMMA_K / 16; ++sk) {
      const int pair0 = sk * 8 + q;
      const int pair1 = pair0 + 4;
      const int sw = (g & 7) << 2;
      uint32_t af[MATOM][4];
#pragma unroll
      for (int ma = 0; ma < MATOM; ++ma) {
        const int mr = ma * 16 + g;
        af[ma][0] = as[mr][pair0 ^ sw];
        af[ma][1] = as[mr + 8][pair0 ^ sw];
        af[ma][2] = as[mr][pair1 ^ sw];
        af[ma][3] = as[mr + 8][pair1 ^ sw];
      }
#pragma unroll
      for (int atom = 0; atom < 2; ++atom) {
        const int nr = warp * 16 + atom * 8 + g;
        const int physical_segment = (sk >> 1) ^ ((nr >> 1) & 3);
        const int word = physical_segment * 4 + (sk & 1) * 2;
        const uint32_t p0 = bs[nr][word];
        const uint32_t p1 = bs[nr][word + 1];
        const uint32_t sfb = e4m3_bcast_bf16x2((ss[nr][sk >> 2] >> ((sk & 3) * 8)) & 0xFFu);
        const uint32_t b0 = mul_bf16x2(e2m1x2_to_bf16x2(uint8_t(p0 >> (q * 8))), sfb);
        const uint32_t b1 = mul_bf16x2(e2m1x2_to_bf16x2(uint8_t(p1 >> (q * 8))), sfb);
#pragma unroll
        for (int ma = 0; ma < MATOM; ++ma)
          mma_m16n8k16_bf16(acc[ma][atom][0], acc[ma][atom][1], acc[ma][atom][2], acc[ma][atom][3],
                            af[ma][0], af[ma][1], af[ma][2], af[ma][3], b0, b1);
      }
    }
    __syncthreads();
  }

  const float al = alpha[e];
#pragma unroll
  for (int ma = 0; ma < MATOM; ++ma) {
#pragma unroll
    for (int atom = 0; atom < 2; ++atom) {
      const int n0 = ntile * TILE_N + warp * 16 + atom * 8 + q * 2;
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        const int m = ma * 16 + g + rr * 8;
        if (m < valid_m) {
          const int grouped_row = offsets[e] + m0 + m;
          const int r = sroutes[m];
          float v0 = acc[ma][atom][rr * 2] * al;
          float v1 = acc[ma][atom][rr * 2 + 1] * al;
          if constexpr (FIRST) {
            v0 = fmaxf(v0, 0.0f);
            v1 = fmaxf(v1, 0.0f);
            *reinterpret_cast<__nv_bfloat162*>(output + size_t(grouped_row) * N + n0) =
                __floats2bfloat162_rn(v0 * v0, v1 * v1);
          } else {
            *reinterpret_cast<__nv_bfloat162*>(output + size_t(r) * N + n0) =
                __floats2bfloat162_rn(v0, v1);
          }
        }
      }
    }
  }
}

// ===================== fast decode path (expert-grouped GEMV) =====================
constexpr int DEC_MAXR = 512;
constexpr int DEC_WARPS = 8;
constexpr int DEC_THREADS = DEC_WARPS * 32;

__device__ __forceinline__ void bf16x8_to_half2x4(const uint4& v, __half2* dst) {
  const uint32_t* w = reinterpret_cast<const uint32_t*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&w[i]);
    dst[i] = __float22half2_rn(__bfloat1622float2(b));
  }
}

// 4 packed e2m1 bytes -> 4 half2 (8 values) with one operand split
__device__ __forceinline__ void e2m1x8_to_half2x4(uint32_t w, __half2* o) {
  uint32_t r0, r1, r2, r3;
  asm volatile(
      "{\n"
      ".reg .b8 b0, b1, b2, b3;\n"
      "mov.b32 {b0, b1, b2, b3}, %4;\n"
      "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
      "cvt.rn.f16x2.e2m1x2 %1, b1;\n"
      "cvt.rn.f16x2.e2m1x2 %2, b2;\n"
      "cvt.rn.f16x2.e2m1x2 %3, b3;\n"
      "}\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(w));
  o[0] = *reinterpret_cast<const __half2*>(&r0);
  o[1] = *reinterpret_cast<const __half2*>(&r1);
  o[2] = *reinterpret_cast<const __half2*>(&r2);
  o[3] = *reinterpret_cast<const __half2*>(&r3);
}

__device__ __forceinline__ __half2 e4m3_bcast_half2(uint32_t b) {
  uint32_t bits;
  const uint16_t src = uint16_t((b & 0xFFu) * 0x0101u);
  asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(bits) : "h"(src));
  return *reinterpret_cast<const __half2*>(&bits);
}

__device__ __forceinline__ void dec_load_x16(const __nv_bfloat16* p, uint32_t* out) {
  asm volatile(
      "ld.global.L1::evict_last.v8.u32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
      : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3]), "=r"(out[4]), "=r"(out[5]),
        "=r"(out[6]), "=r"(out[7])
      : "l"(p));
}

__device__ __forceinline__ void red_bf16x2_v2(__nv_bfloat162* dst, __nv_bfloat162 v0,
                                              __nv_bfloat162 v1) {
  const uint32_t a = *reinterpret_cast<const uint32_t*>(&v0);
  const uint32_t b = *reinterpret_cast<const uint32_t*>(&v1);
  asm volatile("red.global.add.noftz.v2.bf16x2 [%0], {%1, %2};\n"
               :
               : "l"(dst), "r"(a), "r"(b)
               : "memory");
}

// One (route, n-tile) of a decode GEMV: MT=1, activations straight from L1,
// every weight/scale load of the warp hoisted up front.
template <int KK, int NN, int RPW, bool FIRST>
__device__ __forceinline__ void dec_tile_direct(const __nv_bfloat16* __restrict__ xin,
                                                const uint8_t* __restrict__ packed,
                                                const uint8_t* __restrict__ scales, float alph,
                                                const float* __restrict__ tw, int e, int route,
                                                int tile, __nv_bfloat16* __restrict__ interm,
                                                __nv_bfloat16* __restrict__ outp) {
  constexpr int GK = KK / 16;
  constexpr int KHALF = KK / 2;
  constexpr int NT = DEC_WARPS * RPW;
  constexpr int NIT = (GK + 31) / 32;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int xrow = FIRST ? (route / TOPK) : route;
  const int nb = tile * NT + warp * RPW;
  const uint8_t* wp = packed + (size_t(e) * NN + nb) * KHALF;
  const uint8_t* sp = scales + (size_t(e) * NN + nb) * GK;

  uint2 qb[NIT][RPW];
  uint32_t scb[NIT];
#pragma unroll
  for (int it = 0; it < NIT; ++it) {
    const int stl = it * 32 + lane;
    const bool ok = (stl < GK);
    const int st = ok ? stl : (GK - 1);
    uint32_t scpack = 0u;
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      uint2 v = make_uint2(0u, 0u);
      uint8_t sc = 0u;
      if (ok) {
        v = *reinterpret_cast<const uint2*>(wp + size_t(r) * KHALF + st * 8);
        sc = sp[size_t(r) * GK + st];
      }
      qb[it][r] = v;
      scpack |= uint32_t(sc) << (8 * r);
    }
    scb[it] = scpack;
  }

  __half2 acc[RPW];
#pragma unroll
  for (int r = 0; r < RPW; ++r) acc[r] = __halves2half2(__float2half(0.f), __float2half(0.f));

#pragma unroll
  for (int it = 0; it < NIT; ++it) {
    const int st = min(it * 32 + lane, GK - 1);
    __half2 xv[8];
    {
      const uint4* xg = reinterpret_cast<const uint4*>(xin + size_t(xrow) * KK + st * 16);
      const uint4 g0 = __ldg(xg);
      const uint4 g1 = __ldg(xg + 1);
      const uint32_t gw[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        const __nv_bfloat162 bb = *reinterpret_cast<const __nv_bfloat162*>(&gw[c]);
        xv[c] = __float22half2_rn(__bfloat1622float2(bb));
      }
    }
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      const __half2 sf = e4m3_bcast_half2(scb[it] >> (8 * r));
      __half2 wv[4];
      __half2 sacc = __halves2half2(__float2half(0.f), __float2half(0.f));
      e2m1x8_to_half2x4(qb[it][r].x, wv);
#pragma unroll
      for (int p = 0; p < 4; ++p) sacc = __hfma2(wv[p], xv[p], sacc);
      e2m1x8_to_half2x4(qb[it][r].y, wv);
#pragma unroll
      for (int p = 0; p < 4; ++p) sacc = __hfma2(wv[p], xv[p + 4], sacc);
      acc[r] = __hfma2(sacc, sf, acc[r]);
    }
  }

  float red[RPW];
#pragma unroll
  for (int r = 0; r < RPW; ++r)
    red[r] = warp_sum(__low2float(acc[r]) + __high2float(acc[r])) * alph;
  if (lane == 0) {
    if (FIRST) {
#pragma unroll
      for (int r = 0; r < RPW; r += 2) {
        const float v0 = fmaxf(red[r], 0.f);
        const float v1 = fmaxf(red[r + 1], 0.f);
        *reinterpret_cast<__nv_bfloat162*>(interm + size_t(route) * NN + nb + r) =
            __floats2bfloat162_rn(v0 * v0, v1 * v1);
      }
    } else {
      const float w = tw[route];
      __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(outp + size_t(route / TOPK) * NN + nb);
      if constexpr (RPW == 4) {
        const __nv_bfloat162 v0 = __floats2bfloat162_rn(red[0] * w, red[1] * w);
        const __nv_bfloat162 v1 = __floats2bfloat162_rn(red[2] * w, red[3] * w);
        red_bf16x2_v2(op, v0, v1);
      } else {
#pragma unroll
        for (int r = 0; r < RPW; r += 2)
          atomicAdd(op + (r >> 1), __floats2bfloat162_rn(red[r] * w, red[r + 1] * w));
      }
    }
  }
}

// Packed-BF16 arithmetic variant for T=1.  Keeping this separate preserves
// the established FP16 register allocation and instruction stream at T=4.
template <int KK, int NN, int RPW, bool FIRST, bool VECTOR_X = false>
__device__ __forceinline__ void dec_tile_direct_bf16(const __nv_bfloat16* __restrict__ xin,
                                                     const uint8_t* __restrict__ packed,
                                                     const uint8_t* __restrict__ scales, float alph,
                                                     const float* __restrict__ tw, int e, int route,
                                                     int tile, __nv_bfloat16* __restrict__ interm,
                                                     __nv_bfloat16* __restrict__ outp) {
  constexpr int GK = KK / 16;
  constexpr int KHALF = KK / 2;
  constexpr int NT = DEC_WARPS * RPW;
  constexpr int NIT = (GK + 31) / 32;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int xrow = FIRST ? (route / TOPK) : route;
  const int nb = tile * NT + warp * RPW;
  const uint8_t* wp = packed + (size_t(e) * NN + nb) * KHALF;
  const uint8_t* sp = scales + (size_t(e) * NN + nb) * GK;

  uint2 qb[NIT][RPW];
  uint32_t scb[NIT];
#pragma unroll
  for (int it = 0; it < NIT; ++it) {
    const int stl = it * 32 + lane;
    const bool ok = (stl < GK);
    const int st = ok ? stl : (GK - 1);
    uint32_t scpack = 0u;
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      uint2 v = make_uint2(0u, 0u);
      uint8_t sc = 0u;
      if (ok) {
        v = *reinterpret_cast<const uint2*>(wp + size_t(r) * KHALF + st * 8);
        sc = sp[size_t(r) * GK + st];
      }
      qb[it][r] = v;
      scpack |= uint32_t(sc) << (8 * r);
    }
    scb[it] = scpack;
  }

  uint32_t acc[RPW];
#pragma unroll
  for (int r = 0; r < RPW; ++r) acc[r] = 0u;

#pragma unroll
  for (int it = 0; it < NIT; ++it) {
    const int st = min(it * 32 + lane, GK - 1);
    uint32_t xv[8];
    if constexpr (VECTOR_X) {
      dec_load_x16(xin + size_t(xrow) * KK + st * 16, xv);
    } else {
      const uint4* xg = reinterpret_cast<const uint4*>(xin + size_t(xrow) * KK + st * 16);
      const uint4 g0 = __ldg(xg);
      const uint4 g1 = __ldg(xg + 1);
      const uint32_t gw[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
#pragma unroll
      for (int c = 0; c < 8; ++c) xv[c] = gw[c];
    }
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      const uint32_t sf = e4m3_bcast_bf16x2(scb[it] >> (8 * r));
      uint32_t wv[4];
      uint32_t sacc = 0u;
      e2m1x8_to_bf16x2x4(qb[it][r].x, wv);
#pragma unroll
      for (int p = 0; p < 4; ++p) sacc = fma_bf16x2(wv[p], xv[p], sacc);
      e2m1x8_to_bf16x2x4(qb[it][r].y, wv);
#pragma unroll
      for (int p = 0; p < 4; ++p) sacc = fma_bf16x2(wv[p], xv[p + 4], sacc);
      acc[r] = fma_bf16x2(sacc, sf, acc[r]);
    }
  }

  float red[RPW];
#pragma unroll
  for (int r = 0; r < RPW; ++r) red[r] = warp_sum(sum_bf16x2(acc[r])) * alph;
  if (lane == 0) {
    if (FIRST) {
#pragma unroll
      for (int r = 0; r < RPW; r += 2) {
        const float v0 = fmaxf(red[r], 0.f);
        const float v1 = fmaxf(red[r + 1], 0.f);
        *reinterpret_cast<__nv_bfloat162*>(interm + size_t(route) * NN + nb + r) =
            __floats2bfloat162_rn(v0 * v0, v1 * v1);
      }
    } else {
      const float w = tw[route];
      __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(outp + size_t(route / TOPK) * NN + nb);
      if constexpr (RPW == 4) {
        const __nv_bfloat162 v0 = __floats2bfloat162_rn(red[0] * w, red[1] * w);
        const __nv_bfloat162 v1 = __floats2bfloat162_rn(red[2] * w, red[3] * w);
        red_bf16x2_v2(op, v0, v1);
      } else {
#pragma unroll
        for (int r = 0; r < RPW; r += 2)
          atomicAdd(op + (r >> 1), __floats2bfloat162_rn(red[r] * w, red[r + 1] * w));
      }
    }
  }
}

// Device-wide barrier over a persistent grid. The host zeroes the selected
// counter immediately before each ordered launch (stream-ordered, so it is
// also captured into CUDA graphs), which keeps every launch's epoch valid
// regardless of workspace reuse across shapes and prevents the unsigned
// counter from ever wrapping within a launch.
__device__ __forceinline__ unsigned int dec_atomic_add_release(unsigned int* bar,
                                                               unsigned int value) {
  unsigned int old;
  asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;"
               : "=r"(old)
               : "l"(bar), "r"(value)
               : "memory");
  return old;
}

__device__ __forceinline__ unsigned int dec_load_acquire(const unsigned int* bar) {
  unsigned int value;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(value) : "l"(bar) : "memory");
  return value;
}

__device__ __forceinline__ void dec_grid_barrier(unsigned int* bar, unsigned int gsz) {
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned int old = dec_atomic_add_release(bar, 1u);
    const unsigned int target = (old / gsz + 1u) * gsz;
    while (dec_load_acquire(bar) < target) __nanosleep(128);
  }
  __syncthreads();
}

// FC1 + relu^2 + FC2 + weighted combine in a single persistent launch, so the
// decode pays the per-kernel ramp once instead of twice.
template <int RPW, int MINB, bool PREFETCH_FC2, bool LIMIT_PREFETCH = false,
          bool REVERSE_ZERO = false, bool BF16_FC1 = false, bool BF16_FC2 = false,
          bool PRELOAD_IDS = false, bool VECTOR_X = false>
__global__ __launch_bounds__(DEC_THREADS, MINB) void dec_fused_direct_kernel(
    const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ p1,
    const uint8_t* __restrict__ s1, const float* __restrict__ a1, const uint8_t* __restrict__ p2,
    const uint8_t* __restrict__ s2, const float* __restrict__ a2, const int32_t* __restrict__ ids,
    const float* __restrict__ tw, int R, __nv_bfloat16* __restrict__ inter,
    __nv_bfloat16* __restrict__ outp, int out_pairs, unsigned int* __restrict__ bar) {
  constexpr int NT = DEC_WARPS * RPW;
  constexpr int NTILES1 = I / NT;
  constexpr int NTILES2 = H / NT;
  const int tid = threadIdx.x;
  const int block0 = int(blockIdx.x);
  const int n1 = R * NTILES1;
  const int n2 = R * NTILES2;
  int e1_pre = -1;
  int e2_pre = -1;
  if constexpr (PRELOAD_IDS) {
    if (block0 < n1) e1_pre = ids[block0 / NTILES1];
    if (block0 < n2) e2_pre = ids[block0 / NTILES2];
  }

  {
    auto* o2 = reinterpret_cast<__nv_bfloat162*>(outp);
    const __nv_bfloat162 z = __floats2bfloat162_rn(0.f, 0.f);
    const int zero_block = REVERSE_ZERO ? int(gridDim.x) - 1 - int(blockIdx.x) : int(blockIdx.x);
    for (int i = zero_block * DEC_THREADS + tid; i < out_pairs; i += gridDim.x * DEC_THREADS)
      o2[i] = z;
  }

  bool first1 = true;
  for (int item = block0; item < n1; item += int(gridDim.x)) {
    const int route = item / NTILES1;
    const int tile = item - route * NTILES1;
    const int e = PRELOAD_IDS && first1 ? e1_pre : ids[route];
    first1 = false;
    if (unsigned(e) < unsigned(E)) {
      if constexpr (BF16_FC1)
        dec_tile_direct_bf16<H, I, RPW, true, VECTOR_X>(x, p1, s1, a1[e], tw, e, route, tile, inter,
                                                        outp);
      else
        dec_tile_direct<H, I, RPW, true>(x, p1, s1, a1[e], tw, e, route, tile, inter, outp);
    }
  }

  if constexpr (PREFETCH_FC2) {
    if constexpr (LIMIT_PREFETCH) {
      // T4 cooperative launch: all threads arrive at the hardware grid
      // barrier, then warps 1-7 issue one FC2 prefetch tile per SM while the
      // barrier drains.  Waiting only after the prefetch preserves overlap
      // without a global atomic counter or polling traffic.
      cg::grid_group grid = cg::this_grid();
      auto token = grid.barrier_arrive();
      constexpr int prefetch_warp_base = 0;
      constexpr int prefetch_threads = DEC_THREADS - prefetch_warp_base;
      const int prefetch_blocks = min(n2, 376);
      if (tid >= prefetch_warp_base && int(blockIdx.x) < prefetch_blocks) {
        const int route = int(blockIdx.x) / NTILES2;
        const int tile = int(blockIdx.x) - route * NTILES2;
        const int e = ids[route];
        if (unsigned(e) < unsigned(E)) {
          const uint8_t* wp = p2 + (size_t(e) * H + size_t(tile) * NT) * (I / 2);
          constexpr int WB = NT * (I / 2);
          const int prefetch_tid = tid - prefetch_warp_base;
          for (int o = prefetch_tid * 128; o < WB; o += prefetch_threads * 128)
            asm volatile("prefetch.global.L1 [%0];" : : "l"(wp + o));
        }
      }
      grid.barrier_wait(std::move(token));
    } else {
      // T1 also uses a split-phase cooperative barrier: arrive first, issue
      // the established full-grid FC2 prefetch, then wait before consuming
      // intermediates.  This removes the software counter without giving up
      // the barrier-time cache warmup.
      cg::grid_group grid = cg::this_grid();
      auto token = grid.barrier_arrive();
      if (int(blockIdx.x) < min(n2, 160)) {
        const int route = int(blockIdx.x) / NTILES2;
        const int tile = int(blockIdx.x) - route * NTILES2;
        const int e = ids[route];
        if (unsigned(e) < unsigned(E)) {
          const uint8_t* wp = p2 + (size_t(e) * H + size_t(tile) * NT) * (I / 2);
          const uint8_t* sp = s2 + (size_t(e) * H + size_t(tile) * NT) * (I / 16);
          constexpr int WB = NT * (I / 2);
          constexpr int SB = NT * (I / 16);
          for (int o = tid * 128; o < WB; o += DEC_THREADS * 128)
            asm volatile("prefetch.global.L2 [%0];" : : "l"(wp + o));
          // Prioritize the packed-weight stream during the barrier; scales
          // are only one eighth as large and remain demand-fetched.
        }
      }
      grid.barrier_wait(std::move(token));
    }
  } else {
    dec_grid_barrier(bar, gridDim.x);
  }

  bool first2 = true;
  for (int item = block0; item < n2; item += int(gridDim.x)) {
    const int route = item / NTILES2;
    const int tile = item - route * NTILES2;
    const int e = PRELOAD_IDS && first2 ? e2_pre : ids[route];
    first2 = false;
    if (unsigned(e) < unsigned(E)) {
      if constexpr (BF16_FC2)
        dec_tile_direct_bf16<I, H, RPW, false, VECTOR_X>(inter, p2, s2, a2[e], tw, e, route, tile,
                                                         inter, outp);
      else
        dec_tile_direct<I, H, RPW, false>(inter, p2, s2, a2[e], tw, e, route, tile, inter, outp);
    }
  }
}

// One (expert token-group, n-tile) of a decode GEMV with MT grouped tokens.
template <int KK, int NN, int MT, int RPW, bool FIRST>
__device__ __forceinline__ void dec_tile_grp(const __nv_bfloat16* __restrict__ xin,
                                             const uint8_t* __restrict__ packed,
                                             const uint8_t* __restrict__ scales, float alph,
                                             const float* __restrict__ tw, int e, const int* rt,
                                             int mt, int tile, __nv_bfloat16* __restrict__ interm,
                                             __nv_bfloat16* __restrict__ outp) {
  constexpr int GK = KK / 16;
  constexpr int KHALF = KK / 2;
  constexpr int NT = DEC_WARPS * RPW;
  constexpr int NIT = (GK + 31) / 32;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  int xrow[MT];
#pragma unroll
  for (int j = 0; j < MT; ++j) xrow[j] = FIRST ? (rt[j] / TOPK) : rt[j];
  const int nb = tile * NT + warp * RPW;
  const uint8_t* wp = packed + (size_t(e) * NN + nb) * KHALF;
  const uint8_t* sp = scales + (size_t(e) * NN + nb) * GK;

  __half2 acc[RPW][MT];
#pragma unroll
  for (int r = 0; r < RPW; ++r)
#pragma unroll
    for (int j = 0; j < MT; ++j) acc[r][j] = __halves2half2(__float2half(0.f), __float2half(0.f));

#pragma unroll 1
  for (int it = 0; it < NIT; ++it) {
    const int stl = it * 32 + lane;
    const bool ok = (stl < GK);
    const int st = ok ? stl : (GK - 1);
    uint2 qb[RPW];
    uint32_t scb[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      uint2 v = make_uint2(0u, 0u);
      uint32_t sc = 0u;
      if (ok) {
        v = *reinterpret_cast<const uint2*>(wp + size_t(r) * KHALF + st * 8);
        sc = sp[size_t(r) * GK + st];
      }
      qb[r] = v;
      scb[r] = sc;
    }
    __half2 xv[MT][8];
#pragma unroll
    for (int j = 0; j < MT; ++j) {
      const uint4* xg = reinterpret_cast<const uint4*>(xin + size_t(xrow[j]) * KK + st * 16);
      const uint4 g0 = __ldg(xg);
      const uint4 g1 = __ldg(xg + 1);
      const uint32_t gw[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        const __nv_bfloat162 bb = *reinterpret_cast<const __nv_bfloat162*>(&gw[c]);
        xv[j][c] = __float22half2_rn(__bfloat1622float2(bb));
      }
    }
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      const __half2 sf = e4m3_bcast_half2(scb[r]);
      __half2 wv[8];
      e2m1x8_to_half2x4(qb[r].x, wv);
      e2m1x8_to_half2x4(qb[r].y, wv + 4);
#pragma unroll
      for (int j = 0; j < MT; ++j) {
        __half2 sacc = __halves2half2(__float2half(0.f), __float2half(0.f));
#pragma unroll
        for (int p = 0; p < 8; ++p) sacc = __hfma2(wv[p], xv[j][p], sacc);
        acc[r][j] = __hfma2(sacc, sf, acc[r][j]);
      }
    }
  }

  float red[RPW][MT];
#pragma unroll
  for (int r = 0; r < RPW; ++r)
#pragma unroll
    for (int j = 0; j < MT; ++j)
      red[r][j] = warp_sum(__low2float(acc[r][j]) + __high2float(acc[r][j])) * alph;
  if (lane == 0) {
#pragma unroll
    for (int j = 0; j < MT; ++j) {
      if (j >= mt) break;
      const int rr = rt[j];
      if (FIRST) {
#pragma unroll
        for (int r = 0; r < RPW; r += 2) {
          const float v0 = fmaxf(red[r][j], 0.f);
          const float v1 = fmaxf(red[r + 1][j], 0.f);
          *reinterpret_cast<__nv_bfloat162*>(interm + size_t(rr) * NN + nb + r) =
              __floats2bfloat162_rn(v0 * v0, v1 * v1);
        }
      } else {
        const float w = tw[rr];
        __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(outp + size_t(rr / TOPK) * NN + nb);
#pragma unroll
        for (int r = 0; r < RPW; r += 2)
          atomicAdd(op + (r >> 1), __floats2bfloat162_rn(red[r][j] * w, red[r + 1][j] * w));
      }
    }
  }
}

// Table-routed FC1 + FC2 in one persistent launch.
template <int MT, int RPW, int MINB>
__global__ __launch_bounds__(DEC_THREADS, MINB) void dec_fused_grp_kernel(
    const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ p1,
    const uint8_t* __restrict__ s1, const float* __restrict__ a1, const uint8_t* __restrict__ p2,
    const uint8_t* __restrict__ s2, const float* __restrict__ a2, const float* __restrict__ tw,
    const int32_t* __restrict__ g_gpre, const int32_t* __restrict__ g_cnt,
    const int32_t* __restrict__ g_off, const int32_t* __restrict__ g_sorted,
    __nv_bfloat16* __restrict__ inter, __nv_bfloat16* __restrict__ outp, int out_pairs,
    unsigned int* __restrict__ bar) {
  constexpr int NT = DEC_WARPS * RPW;
  constexpr int NTILES1 = I / NT;
  constexpr int NTILES2 = H / NT;
  __shared__ int s_gpre[E + 1];
  const int tid = threadIdx.x;

  {
    auto* o2 = reinterpret_cast<__nv_bfloat162*>(outp);
    const __nv_bfloat162 z = __floats2bfloat162_rn(0.f, 0.f);
    for (int i = blockIdx.x * DEC_THREADS + tid; i < out_pairs; i += gridDim.x * DEC_THREADS)
      o2[i] = z;
  }
  if (tid <= E) s_gpre[tid] = g_gpre[tid];
  __syncthreads();
  const int G = s_gpre[E];

  const int n1 = G * NTILES1;
  for (int item = int(blockIdx.x); item < n1; item += int(gridDim.x)) {
    const int tile = item / G;
    const int gidx = item - tile * G;
    int lo = 0, hi = E;
#pragma unroll
    for (int b = 0; b < 8; ++b) {
      const int mid = (lo + hi) >> 1;
      if (s_gpre[mid] <= gidx)
        lo = mid + 1;
      else
        hi = mid;
    }
    const int e = max(0, lo - 1);
    const int rlo = (gidx - s_gpre[e]) * MT;
    const int mt = min(MT, g_cnt[e] - rlo);
    if (mt <= 0) continue;
    int rt[MT];
    const int base_ = g_off[e] + rlo;
#pragma unroll
    for (int j = 0; j < MT; ++j) rt[j] = g_sorted[base_ + (j < mt ? j : 0)];
    dec_tile_grp<H, I, MT, RPW, true>(x, p1, s1, a1[e], tw, e, rt, mt, tile, inter, outp);
  }

  dec_grid_barrier(bar, gridDim.x);

  const int n2 = G * NTILES2;
  for (int item = int(blockIdx.x); item < n2; item += int(gridDim.x)) {
    const int tile = item / G;
    const int gidx = item - tile * G;
    int lo = 0, hi = E;
#pragma unroll
    for (int b = 0; b < 8; ++b) {
      const int mid = (lo + hi) >> 1;
      if (s_gpre[mid] <= gidx)
        lo = mid + 1;
      else
        hi = mid;
    }
    const int e = max(0, lo - 1);
    const int rlo = (gidx - s_gpre[e]) * MT;
    const int mt = min(MT, g_cnt[e] - rlo);
    if (mt <= 0) continue;
    int rt[MT];
    const int base_ = g_off[e] + rlo;
#pragma unroll
    for (int j = 0; j < MT; ++j) rt[j] = g_sorted[base_ + (j < mt ? j : 0)];
    dec_tile_grp<I, H, MT, RPW, false>(inter, p2, s2, a2[e], tw, e, rt, mt, tile, inter, outp);
  }
}

// Builds the decode routing table once per call (counts, offsets, group prefix,
// routes bucketed by expert) so decode CTAs do not each redo the routing scan.
template <int MT>
__global__ __launch_bounds__(256) void dec_route_kernel(const int32_t* __restrict__ ids, int R,
                                                        int32_t* __restrict__ gpre,
                                                        int32_t* __restrict__ cnt_out,
                                                        int32_t* __restrict__ off_out,
                                                        int32_t* __restrict__ sorted) {
  __shared__ int s_cnt[E];
  __shared__ int s_off[E];
  __shared__ int s_cur[E];
  __shared__ int s_gp[E + 1];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (tid < E) {
    s_cnt[tid] = 0;
    s_cur[tid] = 0;
  }
  __syncthreads();
  for (int i = tid; i < R; i += 256) {
    const int ee = ids[i];
    if (unsigned(ee) < unsigned(E)) atomicAdd(&s_cnt[ee], 1);
  }
  __syncthreads();
  if (warp == 0) {
    int tot = 0, gtot = 0;
#pragma unroll
    for (int base = 0; base < E; base += 32) {
      const int c = s_cnt[base + lane];
      const int g = (c + MT - 1) / MT;
      int xc = c, xg = g;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, xc, d);
        const int z = __shfl_up_sync(0xffffffffu, xg, d);
        if (lane >= d) {
          xc += y;
          xg += z;
        }
      }
      s_off[base + lane] = tot + xc - c;
      s_gp[base + lane] = gtot + xg - g;
      tot += __shfl_sync(0xffffffffu, xc, 31);
      gtot += __shfl_sync(0xffffffffu, xg, 31);
    }
    if (lane == 0) s_gp[E] = gtot;
  }
  __syncthreads();
  for (int i = tid; i < R; i += 256) {
    const int ee = ids[i];
    if (unsigned(ee) < unsigned(E)) {
      const int p = atomicAdd(&s_cur[ee], 1);
      sorted[s_off[ee] + p] = i;
    }
  }
  if (tid < E) {
    cnt_out[tid] = s_cnt[tid];
    off_out[tid] = s_off[tid];
  }
  if (tid <= E) gpre[tid] = s_gp[tid];
}

// Each CTA owns one (expert token-group, n-tile-stripe) and walks its stripe,
// so the staged activations are reused across many output rows.
template <int KK, int NN, int MT, int RPW, bool FIRST, int MINB, int RMODE, bool HOIST, bool XGLOB>
__global__ __launch_bounds__(DEC_THREADS, MINB) void dec_gemv_kernel(
    const __nv_bfloat16* __restrict__ xin, const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales, const float* __restrict__ alpha,
    const int32_t* __restrict__ ids, const float* __restrict__ tw, int R, int splitn,
    const int32_t* __restrict__ g_gpre, const int32_t* __restrict__ g_cnt,
    const int32_t* __restrict__ g_off, const int32_t* __restrict__ g_sorted,
    __nv_bfloat16* __restrict__ interm, __nv_bfloat16* __restrict__ outp, int out_pairs) {
  constexpr int GK = KK / 16;
  constexpr int KHALF = KK / 2;
  constexpr int NSTEP = KK / 16;
  constexpr int NT = DEC_WARPS * RPW;
  constexpr int NTILES = NN / NT;
  constexpr int NIT = (NSTEP + 31) / 32;
  constexpr int SUB = NSTEP * 8;

  __shared__ __half sx[XGLOB ? 4 : (MT * KK)];
  __shared__ int s_rid[DEC_MAXR];
  __shared__ int s_cnt[E];
  __shared__ int s_gpre[E + 1];
  __shared__ int s_rt[MT];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  if (FIRST) {
    auto* o2 = reinterpret_cast<__nv_bfloat162*>(outp);
    const __nv_bfloat162 z = __floats2bfloat162_rn(0.f, 0.f);
    for (int i = blockIdx.x * DEC_THREADS + tid; i < out_pairs; i += gridDim.x * DEC_THREADS)
      o2[i] = z;
  }

  int e, rlo, mt;
  int gidx, sidx;
  int rt[MT];
  constexpr bool DIRECT = (RMODE == 2);
  constexpr bool TABLE = (RMODE == 1);
  if constexpr (DIRECT) {
    gidx = int(blockIdx.x) / splitn;
    sidx = int(blockIdx.x) - gidx * splitn;
    e = ids[gidx];
    rlo = 0;
    mt = 1;
    rt[0] = gidx;
  } else if constexpr (TABLE) {
    gidx = int(blockIdx.x) / splitn;
    sidx = int(blockIdx.x) - gidx * splitn;
    if (gidx >= g_gpre[E]) return;
    if (tid <= E) s_gpre[tid] = g_gpre[tid];
    __syncthreads();
    int lo = 0, hi = E;
#pragma unroll
    for (int b = 0; b < 8; ++b) {
      const int mid = (lo + hi) >> 1;
      if (s_gpre[mid] <= gidx)
        lo = mid + 1;
      else
        hi = mid;
    }
    e = max(0, lo - 1);
    rlo = (gidx - s_gpre[e]) * MT;
    mt = min(MT, g_cnt[e] - rlo);
    if (mt <= 0) return;
    const int base_ = g_off[e] + rlo;
#pragma unroll
    for (int j = 0; j < MT; ++j) rt[j] = g_sorted[base_ + (j < mt ? j : 0)];
  } else {
    for (int i = tid; i < R; i += DEC_THREADS) s_rid[i] = ids[i];
    if (tid < E) s_cnt[tid] = 0;
    __syncthreads();
    for (int i = tid; i < R; i += DEC_THREADS) {
      const int ee = s_rid[i];
      if (unsigned(ee) < unsigned(E)) atomicAdd(&s_cnt[ee], 1);
    }
    __syncthreads();
    if (warp == 0) {
      int total = 0;
#pragma unroll
      for (int base = 0; base < E; base += 32) {
        const int c = s_cnt[base + lane];
        const int v = (c + MT - 1) / MT;
        int x = v;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
          const int y = __shfl_up_sync(0xffffffffu, x, d);
          if (lane >= d) x += y;
        }
        s_gpre[base + lane] = total + x - v;
        total += __shfl_sync(0xffffffffu, x, 31);
      }
      if (lane == 0) s_gpre[E] = total;
    }
    __syncthreads();
    gidx = int(blockIdx.x) / splitn;
    sidx = int(blockIdx.x) - gidx * splitn;
    if (gidx >= s_gpre[E]) return;
    int lo = 0, hi = E;
#pragma unroll
    for (int b = 0; b < 8; ++b) {
      const int mid = (lo + hi) >> 1;
      if (s_gpre[mid] <= gidx)
        lo = mid + 1;
      else
        hi = mid;
    }
    e = max(0, lo - 1);
    rlo = (gidx - s_gpre[e]) * MT;
    mt = min(MT, s_cnt[e] - rlo);
    if (mt <= 0) return;
    if (warp == 0) {
      int found = 0;
      for (int base = 0; base < R; base += 32) {
        const int r = base + lane;
        const bool m = (r < R) && (s_rid[r] == e);
        const unsigned mask = __ballot_sync(0xffffffffu, m);
        if (m) {
          const int rank = found + __popc(mask & ((1u << lane) - 1u));
          if (rank >= rlo && rank < rlo + MT) s_rt[rank - rlo] = r;
        }
        found += __popc(mask);
        if (found >= rlo + MT) break;
      }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < MT; ++j) rt[j] = s_rt[j < mt ? j : 0];
  }

  int xrow[MT];
#pragma unroll
  for (int j = 0; j < MT; ++j) xrow[j] = FIRST ? (rt[j] / TOPK) : rt[j];
  if constexpr (!XGLOB) {
#pragma unroll
    for (int j = 0; j < MT; ++j) {
      const int rr = rt[j];
      const int row = FIRST ? (rr / TOPK) : rr;
      const uint4* src = reinterpret_cast<const uint4*>(xin + size_t(row) * KK);
      __half* dst = sx + j * KK;
      for (int i = tid; i < KK / 8; i += DEC_THREADS) {
        const uint4 v = src[i];
        const uint32_t vw[4] = {v.x, v.y, v.z, v.w};
        __half2 h[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&vw[c]);
          h[c] = __float22half2_rn(__bfloat1622float2(b));
        }
        *reinterpret_cast<uint4*>(dst + (i & 1) * SUB + (i >> 1) * 8) =
            *reinterpret_cast<const uint4*>(h);
      }
    }
  }
  if constexpr (!XGLOB) __syncthreads();
  if constexpr (DIRECT) {
    if (unsigned(e) >= unsigned(E)) return;
  }

  const float alph = alpha[e];
  const uint8_t* wbase = packed + size_t(e) * NN * KHALF;
  const uint8_t* sbase = scales + size_t(e) * NN * GK;

  for (int tile = sidx; tile < NTILES; tile += splitn) {
    const int nb = tile * NT + warp * RPW;
    const uint8_t* wp = wbase + size_t(nb) * KHALF;
    const uint8_t* sp = sbase + size_t(nb) * GK;

    __half2 acc[RPW][MT];
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
      for (int j = 0; j < MT; ++j) acc[r][j] = __halves2half2(__float2half(0.f), __float2half(0.f));

    constexpr int HB = HOIST ? NIT : 1;
    uint2 qb[HB][RPW];
    uint32_t scb[HB][RPW];
    if constexpr (HOIST) {
#pragma unroll
      for (int it = 0; it < NIT; ++it) {
        const int stl = it * 32 + lane;
        const bool ok = (stl < NSTEP);
        const int st = ok ? stl : (NSTEP - 1);
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
          uint2 v = make_uint2(0u, 0u);
          uint32_t sc = 0u;
          if (ok) {
            v = *reinterpret_cast<const uint2*>(wp + size_t(r) * KHALF + st * 8);
            sc = sp[size_t(r) * GK + st];
          }
          qb[it][r] = v;
          scb[it][r] = sc;
        }
      }
    }

#pragma unroll HOIST ? 0 : 1
    for (int it = 0; it < NIT; ++it) {
      const int stl = it * 32 + lane;
      const bool ok = (stl < NSTEP);
      const int st = ok ? stl : (NSTEP - 1);
      const int hb = HOIST ? it : 0;
      if constexpr (!HOIST) {
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
          uint2 v = make_uint2(0u, 0u);
          uint32_t sc = 0u;
          if (ok) {
            v = *reinterpret_cast<const uint2*>(wp + size_t(r) * KHALF + st * 8);
            sc = sp[size_t(r) * GK + st];
          }
          qb[0][r] = v;
          scb[0][r] = sc;
        }
      }
      // activations are read once per K step and reused across all RPW rows
      __half2 xv[MT][8];
      if constexpr (XGLOB) {
#pragma unroll
        for (int j = 0; j < MT; ++j) {
          const uint4* xg = reinterpret_cast<const uint4*>(xin + size_t(xrow[j]) * KK + st * 16);
          const uint4 g0 = __ldg(xg);
          const uint4 g1 = __ldg(xg + 1);
          const uint32_t gw[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
#pragma unroll
          for (int c = 0; c < 8; ++c) {
            const __nv_bfloat162 bb = *reinterpret_cast<const __nv_bfloat162*>(&gw[c]);
            xv[j][c] = __float22half2_rn(__bfloat1622float2(bb));
          }
        }
      } else {
#pragma unroll
        for (int j = 0; j < MT; ++j) {
          const __half* xb = sx + j * KK + st * 8;
          *reinterpret_cast<uint4*>(&xv[j][0]) = *reinterpret_cast<const uint4*>(xb);
          *reinterpret_cast<uint4*>(&xv[j][4]) = *reinterpret_cast<const uint4*>(xb + SUB);
        }
      }
#pragma unroll
      for (int r = 0; r < RPW; ++r) {
        const __half2 sf = e4m3_bcast_half2(scb[hb][r]);
        __half2 wv[8];
        e2m1x8_to_half2x4(qb[hb][r].x, wv);
        e2m1x8_to_half2x4(qb[hb][r].y, wv + 4);
#pragma unroll
        for (int j = 0; j < MT; ++j) {
          __half2 sacc = __halves2half2(__float2half(0.f), __float2half(0.f));
#pragma unroll
          for (int p = 0; p < 8; ++p) sacc = __hfma2(wv[p], xv[j][p], sacc);
          acc[r][j] = __hfma2(sacc, sf, acc[r][j]);
        }
      }
    }

    float red[RPW][MT];
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
      for (int j = 0; j < MT; ++j)
        red[r][j] = warp_sum(__low2float(acc[r][j]) + __high2float(acc[r][j])) * alph;
    if (lane == 0) {
#pragma unroll
      for (int j = 0; j < MT; ++j) {
        if (j >= mt) break;
        const int rr = rt[j];
        if (FIRST) {
          if constexpr (RPW >= 2) {
#pragma unroll
            for (int r = 0; r < RPW; r += 2) {
              const float v0 = fmaxf(red[r][j], 0.f);
              const float v1 = fmaxf(red[r + 1][j], 0.f);
              *reinterpret_cast<__nv_bfloat162*>(interm + size_t(rr) * NN + nb + r) =
                  __floats2bfloat162_rn(v0 * v0, v1 * v1);
            }
          } else {
            const float v0 = fmaxf(red[0][j], 0.f);
            interm[size_t(rr) * NN + nb] = __float2bfloat16_rn(v0 * v0);
          }
        } else {
          const float w = tw[rr];
          if constexpr (RPW >= 2) {
            __nv_bfloat162* op =
                reinterpret_cast<__nv_bfloat162*>(outp + size_t(rr / TOPK) * NN + nb);
#pragma unroll
            for (int r = 0; r < RPW; r += 2)
              atomicAdd(op + (r >> 1), __floats2bfloat162_rn(red[r][j] * w, red[r + 1][j] * w));
          } else {
            atomicAdd(outp + size_t(rr / TOPK) * NN + nb, __float2bfloat16_rn(red[0][j] * w));
          }
        }
      }
    }
  }
}

}  // namespace

void RunPersistentDirect(ffi::TensorView hidden_states, ffi::TensorView w1_packed,
                         ffi::TensorView w1_gs, ffi::TensorView w1_alpha, ffi::TensorView w2_packed,
                         ffi::TensorView w2_gs, ffi::TensorView w2_alpha, ffi::TensorView topk_ids,
                         ffi::TensorView topk_weights, ffi::TensorView out, int device,
                         cudaStream_t stream) {
  const int T = int(hidden_states.size(0));
  const int routes = T * TOPK;
  size_t bytes = 0;
  const size_t inter_off = bytes;
  bytes += align_up(size_t(routes) * I * sizeof(__nv_bfloat16));
  const size_t output_scratch_off = bytes;
  bytes += align_up(size_t(routes) * H * sizeof(__nv_bfloat16));
  const size_t routes_off = bytes;
  bytes += align_up(size_t(routes) * sizeof(int32_t));
  const size_t counts_off = bytes;
  bytes += align_up((E + 1) * sizeof(int32_t));
  const size_t cursor_off = bytes;
  bytes += align_up((E + 1) * sizeof(int32_t));
  const size_t offsets_off = bytes;
  bytes += align_up((E + 1) * sizeof(int32_t));
  const size_t tiles1_off = bytes;
  bytes += align_up((E + 1) * sizeof(int32_t));
  const size_t tiles2_off = bytes;
  bytes += align_up((E + 1) * sizeof(int32_t));
  const size_t bar_off = bytes;
  bytes += align_up(4 * sizeof(unsigned int));

  char* workspace = static_cast<char*>(ensure_workspace(bytes, device));
  auto* inter = reinterpret_cast<__nv_bfloat16*>(workspace + inter_off);
  auto* output_acc = reinterpret_cast<float*>(workspace + output_scratch_off);
  auto* route_out = reinterpret_cast<__nv_bfloat16*>(workspace + output_scratch_off);
  auto* route_list = reinterpret_cast<int32_t*>(workspace + routes_off);
  auto* counts = reinterpret_cast<int32_t*>(workspace + counts_off);
  auto* cursor = reinterpret_cast<int32_t*>(workspace + cursor_off);
  auto* offsets = reinterpret_cast<int32_t*>(workspace + offsets_off);
  auto* tiles1 = reinterpret_cast<int32_t*>(workspace + tiles1_off);
  auto* tiles2 = reinterpret_cast<int32_t*>(workspace + tiles2_off);
  auto* bars = reinterpret_cast<unsigned int*>(workspace + bar_off);
  const int bar_slot = T == 1 ? 0 : (T == 4 ? 1 : (T == 16 ? 2 : 3));
  auto* bar = bars + bar_slot;

  const auto* x = static_cast<const __nv_bfloat16*>(hidden_states.data_ptr());
  const auto* p1 = static_cast<const uint8_t*>(w1_packed.data_ptr());
  const auto* s1 = static_cast<const uint8_t*>(w1_gs.data_ptr());
  const auto* a1 = static_cast<const float*>(w1_alpha.data_ptr());
  const auto* p2 = static_cast<const uint8_t*>(w2_packed.data_ptr());
  const auto* s2 = static_cast<const uint8_t*>(w2_gs.data_ptr());
  const auto* a2 = static_cast<const float*>(w2_alpha.data_ptr());
  const auto* ids = static_cast<const int32_t*>(topk_ids.data_ptr());
  const auto* tw = static_cast<const float*>(topk_weights.data_ptr());
  auto* y = static_cast<__nv_bfloat16*>(out.data_ptr());

  if (T <= 64) {
    const int out_pairs = T * (H / 2);
#define DEC_LAUNCH(MTV, RPW1, RPW2, MB, RMD, HST, XG)                                             \
  {                                                                                               \
    constexpr int MT_ = MTV;                                                                      \
    const int act = routes < E ? routes : E;                                                      \
    const int nb_ = act + (routes - act + MT_ - 1) / MT_;                                         \
    int numg = routes < nb_ ? routes : nb_;                                                       \
    if ((RMD) == 2) numg = routes;                                                                \
    constexpr int NTIL1 = I / (DEC_WARPS * RPW1);                                                 \
    constexpr int NTIL2 = H / (DEC_WARPS * RPW2);                                                 \
    if ((RMD) == 1)                                                                               \
      dec_route_kernel<MT_>                                                                       \
          <<<1, 256, 0, stream>>>(ids, routes, tiles1, counts, offsets, route_list);              \
    dec_gemv_kernel<H, I, MT_, RPW1, true, MB, RMD, HST, XG>                                      \
        <<<numg * NTIL1, DEC_THREADS, 0, stream>>>(x, p1, s1, a1, ids, tw, routes, NTIL1, tiles1, \
                                                   counts, offsets, route_list, inter, y,         \
                                                   out_pairs);                                    \
    dec_gemv_kernel<I, H, MT_, RPW2, false, MB, RMD, HST, XG>                                     \
        <<<numg * NTIL2, DEC_THREADS, 0, stream>>>(inter, p2, s2, a2, ids, tw, routes, NTIL2,     \
                                                   tiles1, counts, offsets, route_list, inter, y, \
                                                   out_pairs);                                    \
  }
    if (T <= 4) {
      constexpr int FRPW = 4;
      cudaMemsetAsync(bar, 0, sizeof(unsigned int), stream);
      if (T == 1) {
        static int grid_t1_by_device[kMaxCachedDevices] = {};
        int grid_t1 = (device >= 0 && device < kMaxCachedDevices) ? grid_t1_by_device[device] : 0;
        if (grid_t1 == 0) {
          int nb = 0;
          int sms = 1;
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &nb,
              (const void*)
                  dec_fused_direct_kernel<FRPW, 3, true, false, true, true, true, false, true>,
              DEC_THREADS, 0);
          cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
          grid_t1 = nb * sms;
          if (grid_t1 < 1) grid_t1 = 1;
          if (grid_t1 > 376) grid_t1 = 376;
          if (device >= 0 && device < kMaxCachedDevices) grid_t1_by_device[device] = grid_t1;
        }
        int routes_arg = routes;
        int out_pairs_arg = out_pairs;
        void* args[] = {
            &x,  &p1, &s1, &a1, &p2, &s2, &a2, &ids, &tw, &routes_arg, &inter, &y, &out_pairs_arg,
            &bar};
        cudaError_t lerr = cudaLaunchCooperativeKernel(
            (const void*)
                dec_fused_direct_kernel<FRPW, 3, true, false, true, true, true, false, true>,
            dim3(grid_t1), dim3(DEC_THREADS), args, 0, stream);
        TVM_FFI_ICHECK(lerr == cudaSuccess) << "specialized MoE T=1 cooperative launch (grid "
                                            << grid_t1 << ") failed: " << cudaGetErrorString(lerr);
      } else {
        static int occupancy_t4_by_device[kMaxCachedDevices] = {};
        int occupancy_t4 =
            (device >= 0 && device < kMaxCachedDevices) ? occupancy_t4_by_device[device] : 0;
        if (occupancy_t4 == 0) {
          int nb = 0;
          int sms = 1;
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &nb,
              (const void*)dec_fused_direct_kernel<FRPW, 3, true, true, false, true, true, true>,
              DEC_THREADS, 0);
          cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
          occupancy_t4 = nb * sms;
          if (occupancy_t4 < 1) occupancy_t4 = 1;
          if (device >= 0 && device < kMaxCachedDevices)
            occupancy_t4_by_device[device] = occupancy_t4;
        }
        int grid_t4 = occupancy_t4;
        const int maxitems = routes * (H / (DEC_WARPS * FRPW));
        if (grid_t4 > maxitems) grid_t4 = maxitems;
        int routes_arg = routes;
        int out_pairs_arg = out_pairs;
        void* args[] = {
            &x,  &p1, &s1, &a1, &p2, &s2, &a2, &ids, &tw, &routes_arg, &inter, &y, &out_pairs_arg,
            &bar};
        cudaError_t lerr = cudaLaunchCooperativeKernel(
            (const void*)dec_fused_direct_kernel<FRPW, 3, true, true, false, true, true, true>,
            dim3(grid_t4), dim3(DEC_THREADS), args, 0, stream);
        TVM_FFI_ICHECK(lerr == cudaSuccess) << "specialized MoE T=4 cooperative launch (grid "
                                            << grid_t4 << ") failed: " << cudaGetErrorString(lerr);
      }
    } else {
#define FUSED_GRP(MTV)                                                                           \
  {                                                                                              \
    constexpr int MT_ = MTV;                                                                     \
    static int fg_by_device[kMaxCachedDevices] = {};                                             \
    int fg = (device >= 0 && device < kMaxCachedDevices) ? fg_by_device[device] : 0;             \
    if (fg == 0) {                                                                               \
      int nb = 0;                                                                                \
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(                                             \
          &nb, (const void*)dec_fused_grp_kernel<MT_, 4, 3>, DEC_THREADS, 0);                    \
      int sms = 1;                                                                               \
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);                      \
      fg = nb * sms;                                                                             \
      if (fg < 1) fg = 1;                                                                        \
      if (device >= 0 && device < kMaxCachedDevices) fg_by_device[device] = fg;                  \
    }                                                                                            \
    cudaMemsetAsync(bar, 0, sizeof(unsigned int), stream);                                       \
    dec_route_kernel<MT_>                                                                        \
        <<<1, 256, 0, stream>>>(ids, routes, tiles1, counts, offsets, route_list);               \
    dec_fused_grp_kernel<MT_, 4, 3>                                                              \
        <<<fg, DEC_THREADS, 0, stream>>>(x, p1, s1, a1, p2, s2, a2, tw, tiles1, counts, offsets, \
                                         route_list, inter, y, out_pairs, bar);                  \
  }
      if (T <= 16) {
        FUSED_GRP(2)
      } else {
        FUSED_GRP(2)
      }
#undef FUSED_GRP
    }
#undef DEC_LAUNCH
  } else {
    cudaMemsetAsync(counts, 0, E * sizeof(int32_t), stream);
    cudaMemsetAsync(cursor, 0, E * sizeof(int32_t), stream);
    const int route_blocks =
        ((routes + THREADS - 1) / THREADS < 256) ? (routes + THREADS - 1) / THREADS : 256;
    count_routes_kernel<<<route_blocks, THREADS, 0, stream>>>(ids, routes, counts);
    prefix_mma_routes_kernel<<<1, 1, 0, stream>>>(counts, offsets, cursor, tiles1, tiles2);
    scatter_routes_kernel<<<route_blocks, THREADS, 0, stream>>>(ids, routes, offsets, cursor,
                                                                route_list);
    const int max_mtiles = (routes + PF_M - 1) / PF_M + E;
    grouped_mma2_kernel<H, I, H / 16, I / GMMA_N1, GMMA_N1, 128, true, PF_M>
        <<<max_mtiles*(I / GMMA_N1), 128, 0, stream>>>(x, p1, s1, a1, offsets, tiles1, route_list,
                                                       inter);
    grouped_mma2_kernel<I, H, I / 16, H / GMMA_N2, GMMA_N2, 256, false, PF_M>
        <<<max_mtiles*(H / GMMA_N2), 256, 0, stream>>>(inter, p2, s2, a2, offsets, tiles2,
                                                       route_list, route_out);
    const int out_elems = T * (H / 2);
    const int out_blocks =
        ((out_elems + THREADS - 1) / THREADS < 4096) ? (out_elems + THREADS - 1) / THREADS : 4096;
    combine_route_output_kernel<<<out_blocks, THREADS, 0, stream>>>(route_out, ids, tw, y, T);
  }
}
