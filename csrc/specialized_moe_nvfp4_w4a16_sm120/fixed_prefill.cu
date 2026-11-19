#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace ffi = tvm::ffi;
namespace wmma = nvcuda::wmma;

void RunFastDirect(ffi::TensorView hidden_states, ffi::TensorView w1_packed, ffi::TensorView w1_gs,
                   ffi::TensorView w1_alpha, ffi::TensorView w2_packed, ffi::TensorView w2_gs,
                   ffi::TensorView w2_alpha, ffi::TensorView topk_ids, ffi::TensorView topk_weights,
                   ffi::TensorView out, cudaStream_t stream);

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

inline size_t align_up(size_t x, size_t a = 256) { return (x + a - 1) & ~(a - 1); }

// Persistent-grid sizes depend on the launch device's SM count and occupancy;
// cache them per device so a value computed on one GPU is never reused on a
// smaller one (where it could exceed the resident-block capacity and leave
// blocks waiting at the grid barrier).
constexpr int kMaxCachedDevices = 64;

// Buffers retired by workspace growth stay alive for the process lifetime:
// a CUDA graph captured earlier may still replay kernels holding the old
// address. One workspace slot per device (a device switch retires nothing),
// and growth at least doubles a slot's capacity, so every retired buffer is
// at most half its replacement and the retained total stays below the live
// per-device capacity across arbitrary admitted token-count sequences.
std::vector<void*> g_retired_workspaces;
Workspace g_workspaces_by_device[kMaxCachedDevices];

void* ensure_workspace(size_t bytes, int device) {
  // Devices beyond the per-device table share one slot with the old
  // retire-on-switch policy (bounded by how rarely such devices occur).
  static Workspace overflow;
  Workspace& ws =
      (device >= 0 && device < kMaxCachedDevices) ? g_workspaces_by_device[device] : overflow;
  if (ws.ptr != nullptr && (ws.bytes < bytes || ws.device != device)) {
    // Geometric growth: never grow a slot by less than 2x, so each retired
    // buffer is at most half its replacement and the retained total is
    // bounded by the live capacity.
    if (ws.device == device && bytes < ws.bytes * 2) bytes = ws.bytes * 2;
    g_retired_workspaces.push_back(ws.ptr);
    ws.ptr = nullptr;
    ws.bytes = 0;
  }
  if (ws.ptr == nullptr) {
    cudaSetDevice(device);
    cudaError_t err = cudaMalloc(&ws.ptr, bytes);
    TVM_FFI_ICHECK(err == cudaSuccess && ws.ptr != nullptr)
        << "specialized MoE workspace cudaMalloc(" << bytes
        << ") failed: " << cudaGetErrorString(err);
    ws.bytes = bytes;
    ws.device = device;
  }
  return ws.ptr;
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

__device__ __forceinline__ void ldmatrix_m8n8x4_b16(uint32_t& a0, uint32_t& a1, uint32_t& a2,
                                                    uint32_t& a3, const void* smem_ptr) {
  const uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 "
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
      : "r"(smem));
}

__device__ __forceinline__ void cp_async_16(void* dst, const void* src, int src_bytes) {
  const uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16, %2;\n"
               :
               : "r"(smem), "l"(src), "r"(src_bytes));
}

__device__ __forceinline__ void cp_async_16_l2_256(void* dst, const void* src, int src_bytes) {
  const uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16, %2;\n"
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
    const unsigned active = __activemask();
    const unsigned peers = __match_any_sync(active, e);
    const int leader = __ffs(peers) - 1;
    if (unsigned(e) < E && (threadIdx.x & 31) == leader) atomicAdd(counts + e, __popc(peers));
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

template <int MM>
__global__ __launch_bounds__(128) void prefix_mma_routes_parallel_kernel(
    const int32_t* __restrict__ counts, int32_t* __restrict__ offsets, int32_t* __restrict__ cursor,
    int32_t* __restrict__ tiles1, int32_t* __restrict__ tiles2) {
  const int e = threadIdx.x;
  const int lane = e & 31;
  const int warp = e >> 5;
  const int count = counts[e];
  const int mtiles = (count + MM - 1) / MM;
  int rs = count;
  int t1s = mtiles * (I / GMMA_N1);
  int t2s = mtiles * (H / GMMA_N2);
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const int r = __shfl_up_sync(0xffffffffu, rs, d);
    const int a = __shfl_up_sync(0xffffffffu, t1s, d);
    const int b = __shfl_up_sync(0xffffffffu, t2s, d);
    if (lane >= d) {
      rs += r;
      t1s += a;
      t2s += b;
    }
  }
  __shared__ int warp_sums[4][3];
  if (lane == 31) {
    warp_sums[warp][0] = rs;
    warp_sums[warp][1] = t1s;
    warp_sums[warp][2] = t2s;
  }
  __syncthreads();
  int rbase = 0;
  int t1base = 0;
  int t2base = 0;
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    if (w < warp) {
      rbase += warp_sums[w][0];
      t1base += warp_sums[w][1];
      t2base += warp_sums[w][2];
    }
  }
  offsets[e] = rbase + rs - count;
  cursor[e] = 0;
  tiles1[e] = t1base + t1s - mtiles * (I / GMMA_N1);
  tiles2[e] = t2base + t2s - mtiles * (H / GMMA_N2);
  if (e == E - 1) {
    offsets[E] = rbase + rs;
    tiles1[E] = t1base + t1s;
    tiles2[E] = t2base + t2s;
  }
}

template <int MM>
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
    const int mtiles = (counts[e] + MM - 1) / MM;
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
      const unsigned active = __activemask();
      const unsigned peers = __match_any_sync(active, e);
      const int lane = threadIdx.x & 31;
      const int leader = __ffs(peers) - 1;
      int base = 0;
      if (lane == leader) base = atomicAdd(cursor + e, __popc(peers));
      base = __shfl_sync(peers, base, leader);
      const unsigned lane_lt = lane ? ((1u << lane) - 1u) : 0u;
      route_list[offsets[e] + base + __popc(peers & lane_lt)] = r;
    }
  }
}

// Count and scatter in one pass. Each expert owns a fixed-capacity bin, so
// the compact prefix is not needed before routes can be written.
__global__ void bin_routes_kernel(const int32_t* __restrict__ ids, int routes,
                                  int32_t* __restrict__ counts, int32_t* __restrict__ route_list) {
  for (int r = int(blockIdx.x) * blockDim.x + threadIdx.x; r < routes;
       r += int(gridDim.x) * blockDim.x) {
    const int e = ids[r];
    if (unsigned(e) < E) {
      const unsigned active = __activemask();
      const unsigned peers = __match_any_sync(active, e);
      const int lane = threadIdx.x & 31;
      const int leader = __ffs(peers) - 1;
      int base = 0;
      if (lane == leader) base = atomicAdd(counts + e, __popc(peers));
      base = __shfl_sync(peers, base, leader);
      const unsigned lane_lt = lane ? ((1u << lane) - 1u) : 0u;
      route_list[size_t(e) * routes + base + __popc(peers & lane_lt)] = r;
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
  // Keep a 64-byte row even for K64 so the four-way segment swizzle remains
  // bank-conflict-free; only two of the four segments are populated per stage.
  __shared__ __align__(16) uint32_t bs[TILE_N][16];
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
      cp_async_16_l2_256(&bs[nr][physical_segment * 4], src, src_bytes);
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

__device__ __forceinline__ void red_bf16x2_v2(__nv_bfloat162* dst, uint32_t a, uint32_t b) {
  asm volatile("red.global.add.noftz.v2.bf16x2 [%0], {%1, %2};\n"
               :
               : "l"(dst), "r"(a), "r"(b)
               : "memory");
}

// Grouped BF16 mma.sync GEMM with on-the-fly NVFP4 dequant.  The group scale is
// folded into the B fragment (exact: 2-bit e2m1 x 4-bit e4m3 fits bf16's mantissa),
// so one accumulator chain spans the whole K loop and MM rows share each B tile.
template <int K, int N, int KG, int NTILES, int TILE_N, int BLOCK_THREADS, bool FIRST, int MM,
          int STAGE_K, bool ATOMIC_COMBINE = false>
__global__ __launch_bounds__(BLOCK_THREADS) void grouped_mma2_kernel(
    const __nv_bfloat16* __restrict__ input, const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales, const float* __restrict__ alpha,
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ tiles,
    const int32_t* __restrict__ route_list, const float* __restrict__ topk_weights,
    __nv_bfloat16* __restrict__ zero_output, int zero_vecs, __nv_bfloat16* __restrict__ output,
    int route_stride) {
  const int actual_tiles = tiles[E];
  if constexpr (FIRST) {
    const int pad_blocks = int(gridDim.x) - actual_tiles;
    if (int(blockIdx.x) >= actual_tiles) {
      auto* zero4 = reinterpret_cast<uint4*>(zero_output);
      const int worker = int(blockIdx.x) - actual_tiles;
      for (int x = worker * blockDim.x + threadIdx.x; x < zero_vecs; x += pad_blocks * blockDim.x)
        zero4[x] = make_uint4(0u, 0u, 0u, 0u);
    }
  }
  if (int(blockIdx.x) >= actual_tiles) return;
  const int e = find_expert<NTILES>(int(blockIdx.x), tiles);
  const int local = int(blockIdx.x) - tiles[e];
  const int ntile = local % NTILES;
  const int m0 = (local / NTILES) * MM;
  const int valid_m = min(MM, offsets[e + 1] - offsets[e] - m0);
  const int active_atoms = (valid_m + 15) >> 4;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int q = lane & 3;
  constexpr int MATOM = MM / 16;
  constexpr int NWARPS = BLOCK_THREADS / 32;
  constexpr int NATOM = TILE_N / NWARPS / 8;
  constexpr int KHALF = K / 2;
  constexpr int SCALE_WORDS = STAGE_K / 64;
  constexpr int SCALE_ITEMS = TILE_N * SCALE_WORDS;
  constexpr int SCALE_TASKS = (SCALE_ITEMS + BLOCK_THREADS - 1) / BLOCK_THREADS;
  static_assert(TILE_N % (NWARPS * 8) == 0);
  uint32_t scale_prefetch[SCALE_TASKS] = {};

  __shared__ __align__(16) uint32_t as[MM][STAGE_K / 2];
  // Keep a 64-byte row even for K64 so the four-way segment swizzle remains
  // bank-conflict-free; only two of the four segments are populated per stage.
  __shared__ __align__(16) uint32_t bs[TILE_N][16];
  __shared__ __align__(16) uint16_t sbf[STAGE_K / 16][TILE_N];
  __shared__ int sroutes[MM];

  for (int i = tid; i < MM; i += BLOCK_THREADS) {
    const int grouped_row = offsets[e] + m0 + i;
    const int route_index = route_stride ? e * route_stride + m0 + i : grouped_row;
    sroutes[i] = i < valid_m ? route_list[route_index] : -1;
  }
  __syncthreads();

  float acc[MATOM][NATOM][4];
#pragma unroll
  for (int ma = 0; ma < MATOM; ++ma)
#pragma unroll
    for (int atom = 0; atom < NATOM; ++atom)
#pragma unroll
      for (int x = 0; x < 4; ++x) acc[ma][atom][x] = 0.0f;

  for (int k0 = 0; k0 < K; k0 += STAGE_K) {
    const int stage_k = min(STAGE_K, K - k0);
    for (int idx = tid; idx < MM * (STAGE_K / 8); idx += BLOCK_THREADS) {
      const int m = idx / (STAGE_K / 8);
      const int chunk = idx - m * (STAGE_K / 8);
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
    for (int idx = tid; idx < TILE_N * (STAGE_K / 32); idx += BLOCK_THREADS) {
      const int nr = idx / (STAGE_K / 32);
      const int chunk = idx - nr * (STAGE_K / 32);
      const int n = ntile * TILE_N + nr;
      const size_t row = size_t(e) * N + n;
      const void* src = packed;
      int src_bytes = 0;
      if (chunk * 32 < stage_k) {
        src = packed + row * KHALF + (k0 >> 1) + chunk * 16;
        src_bytes = 16;
      }
      const int physical_segment = chunk ^ ((nr >> 1) & 3);
      cp_async_16_l2_256(&bs[nr][physical_segment * 4], src, src_bytes);
    }
    cp_async_commit();
#pragma unroll
    for (int si = 0; si < SCALE_TASKS; ++si) {
      const int scale_idx = tid + si * BLOCK_THREADS;
      if (scale_idx < SCALE_ITEMS) {
        const int scale_nr = scale_idx / SCALE_WORDS;
        const int scale_word = scale_idx - scale_nr * SCALE_WORDS;
        const uint8_t* scale_row = scales + (size_t(e) * N + ntile * TILE_N + scale_nr) * KG;
        uint32_t sv = 0;
        if constexpr (!FIRST) sv = scale_prefetch[si];
        if ((FIRST || k0 == 0) && scale_word * 64 < stage_k) {
          sv = *reinterpret_cast<const uint32_t*>(scale_row + (k0 >> 4) + scale_word * 4);
        }
#pragma unroll
        for (int b = 0; b < 4; ++b)
          sbf[scale_word * 4 + b][scale_nr] = uint16_t(e4m3_bcast_bf16x2(sv >> (b * 8)));

        if constexpr (!FIRST) {
          scale_prefetch[si] = 0;
          const int next_k0 = k0 + STAGE_K;
          if (next_k0 < K) {
            const int next_stage_k = min(STAGE_K, K - next_k0);
            if (scale_word * 64 < next_stage_k) {
              scale_prefetch[si] =
                  *reinterpret_cast<const uint32_t*>(scale_row + (next_k0 >> 4) + scale_word * 4);
            }
          }
        }
      }
    }
    cp_async_wait();
    __syncthreads();

#pragma unroll 2
    for (int sk = 0; sk < STAGE_K / 16; ++sk) {
      uint32_t af[MATOM][4];
#pragma unroll
      for (int ma = 0; ma < MATOM; ++ma) {
        if (ma < active_atoms) {
          const int row = ma * 16 + (lane & 15);
          const int logical_segment = sk * 2 + (lane >> 4);
          const int physical_segment = logical_segment ^ (row & 7);
          ldmatrix_m8n8x4_b16(af[ma][0], af[ma][1], af[ma][2], af[ma][3],
                              &as[row][physical_segment * 4]);
        }
      }
#pragma unroll
      for (int atom = 0; atom < NATOM; ++atom) {
        const int nr = warp * (NATOM * 8) + atom * 8 + g;
        const int physical_segment = (sk >> 1) ^ ((nr >> 1) & 3);
        const int word = physical_segment * 4 + (sk & 1) * 2;
        const uint32_t p0 = bs[nr][word];
        const uint32_t p1 = bs[nr][word + 1];
        const uint32_t sf16 = sbf[sk][nr];
        const uint32_t sfb = sf16 | (sf16 << 16);
        const uint32_t b0 = mul_bf16x2(e2m1x2_to_bf16x2(uint8_t(p0 >> (q * 8))), sfb);
        const uint32_t b1 = mul_bf16x2(e2m1x2_to_bf16x2(uint8_t(p1 >> (q * 8))), sfb);
#pragma unroll
        for (int ma = 0; ma < MATOM; ++ma) {
          if (ma < active_atoms)
            mma_m16n8k16_bf16(acc[ma][atom][0], acc[ma][atom][1], acc[ma][atom][2],
                              acc[ma][atom][3], af[ma][0], af[ma][1], af[ma][2], af[ma][3], b0, b1);
        }
      }
    }
    __syncthreads();
  }

  const float al = alpha[e];
#pragma unroll
  for (int ma = 0; ma < MATOM; ++ma) {
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
      const int m = ma * 16 + g + rr * 8;
      if (m < valid_m) {
        const int grouped_row = offsets[e] + m0 + m;
        const int r = sroutes[m];
        float w = 0.0f;
        if constexpr (!FIRST && ATOMIC_COMBINE) w = topk_weights[r];
#pragma unroll
        for (int atom = 0; atom < NATOM; ++atom) {
          const int n0 = ntile * TILE_N + warp * (NATOM * 8) + atom * 8 + q * 2;
          float v0 = acc[ma][atom][rr * 2] * al;
          float v1 = acc[ma][atom][rr * 2 + 1] * al;
          if constexpr (FIRST) {
            v0 = fmaxf(v0, 0.0f);
            v1 = fmaxf(v1, 0.0f);
            *reinterpret_cast<__nv_bfloat162*>(output + size_t(grouped_row) * N + n0) =
                __floats2bfloat162_rn(v0 * v0, v1 * v1);
          } else {
            if constexpr (ATOMIC_COMBINE) {
              const __nv_bfloat162 pair = __floats2bfloat162_rn(v0 * w, v1 * w);
              const uint32_t bits = *reinterpret_cast<const uint32_t*>(&pair);
              const unsigned mask = __activemask();
              const uint32_t next = __shfl_down_sync(mask, bits, 1);
              if ((q & 1) == 0) {
                auto* dst = reinterpret_cast<__nv_bfloat162*>(output + size_t(r / TOPK) * N + n0);
                red_bf16x2_v2(dst, bits, next);
              }
            } else {
              *reinterpret_cast<__nv_bfloat162*>(output + size_t(r) * N + n0) =
                  __floats2bfloat162_rn(v0, v1);
            }
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
  uint32_t scb[NIT][RPW];
#pragma unroll
  for (int it = 0; it < NIT; ++it) {
    const int stl = it * 32 + lane;
    const bool ok = (stl < GK);
    const int st = ok ? stl : (GK - 1);
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
      const __half2 sf = e4m3_bcast_half2(scb[it][r]);
      __half2 wv[8];
      e2m1x8_to_half2x4(qb[it][r].x, wv);
      e2m1x8_to_half2x4(qb[it][r].y, wv + 4);
      __half2 sacc = __halves2half2(__float2half(0.f), __float2half(0.f));
#pragma unroll
      for (int p = 0; p < 8; ++p) sacc = __hfma2(wv[p], xv[p], sacc);
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
#pragma unroll
      for (int r = 0; r < RPW; r += 2)
        atomicAdd(op + (r >> 1), __floats2bfloat162_rn(red[r] * w, red[r + 1] * w));
    }
  }
}

// Device-wide barrier over a persistent grid. The host zeroes the selected
// counter immediately before each ordered launch (stream-ordered, so it is
// also captured into CUDA graphs), which keeps every launch's epoch valid
// regardless of workspace reuse across shapes and prevents the unsigned
// counter from ever wrapping within a launch.
__device__ __forceinline__ void dec_grid_barrier(unsigned int* bar, unsigned int gsz) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    const unsigned int old = atomicAdd(bar, 1u);
    const unsigned int target = (old / gsz + 1u) * gsz;
    while (atomicAdd(bar, 0u) < target) __nanosleep(128);
  }
  __syncthreads();
  __threadfence();
}

// FC1 + relu^2 + FC2 + weighted combine in a single persistent launch, so the
// decode pays the per-kernel ramp once instead of twice.
template <int RPW, int MINB>
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

  {
    auto* o2 = reinterpret_cast<__nv_bfloat162*>(outp);
    const __nv_bfloat162 z = __floats2bfloat162_rn(0.f, 0.f);
    for (int i = blockIdx.x * DEC_THREADS + tid; i < out_pairs; i += gridDim.x * DEC_THREADS)
      o2[i] = z;
  }

  const int n1 = R * NTILES1;
  for (int item = int(blockIdx.x); item < n1; item += int(gridDim.x)) {
    const int route = item / NTILES1;
    const int tile = item - route * NTILES1;
    const int e = ids[route];
    if (unsigned(e) < unsigned(E))
      dec_tile_direct<H, I, RPW, true>(x, p1, s1, a1[e], tw, e, route, tile, inter, outp);
  }

  dec_grid_barrier(bar, gridDim.x);

  const int n2 = R * NTILES2;
  for (int item = int(blockIdx.x); item < n2; item += int(gridDim.x)) {
    const int route = item / NTILES2;
    const int tile = item - route * NTILES2;
    const int e = ids[route];
    if (unsigned(e) < unsigned(E))
      dec_tile_direct<I, H, RPW, false>(inter, p2, s2, a2[e], tw, e, route, tile, inter, outp);
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

void RunFixedM64Prefill(ffi::TensorView hidden_states, ffi::TensorView w1_packed,
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
  const size_t route_capacity = T == 1024 ? size_t(E) * routes : size_t(routes);
  bytes += align_up(route_capacity * sizeof(int32_t));
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
  // Persistent decode grids differ for T=1..4, so each shape owns a slot.
  // bar_off is shape-dependent (the workspace is reused across shapes), so
  // the selected slot is zeroed before every ordered launch below; a slot is
  // never trusted to hold a prior epoch.
  bytes += align_up(5 * sizeof(unsigned int));

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

  if (T == 64) {
    RunFastDirect(hidden_states, w1_packed, w1_gs, w1_alpha, w2_packed, w2_gs, w2_alpha, topk_ids,
                  topk_weights, out, stream);
    return;
  }
  if (T == 16) {
    decode_group16_half_kernel<H, I, H / 16, I / DECODE_NT, DECODE_NT, true>
        <<<routes*(I / DECODE_NT), THREADS, 0, stream>>>(x, p1, s1, a1, ids, nullptr, tw, routes,
                                                         inter, output_acc);
    decode_group16_half_kernel<I, H, I / 16, H / DECODE_NT, DECODE_NT, false, false, true>
        <<<routes*(H / DECODE_NT), THREADS, 0, stream>>>(inter, p2, s2, a2, ids, nullptr, tw,
                                                         routes, route_out, output_acc);
    const int out_elems = T * (H / 2);
    const int out_blocks = (out_elems + THREADS - 1) / THREADS;
    combine_route_output_kernel<<<out_blocks, THREADS, 0, stream>>>(route_out, ids, tw, y, T);
    return;
  }

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
      static int fused_grid_by_device[kMaxCachedDevices] = {};
      int fused_grid =
          (device >= 0 && device < kMaxCachedDevices) ? fused_grid_by_device[device] : 0;
      if (fused_grid == 0) {
        int nb = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &nb, (const void*)dec_fused_direct_kernel<FRPW, 3>, DEC_THREADS, 0);
        int sms = 1;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
        fused_grid = nb * sms;
        if (fused_grid < 1) fused_grid = 1;
        if (device >= 0 && device < kMaxCachedDevices) fused_grid_by_device[device] = fused_grid;
      }
      unsigned int* bar = bars + (T - 1);
      cudaMemsetAsync(bar, 0, sizeof(unsigned int), stream);
      int grid = fused_grid;
      const int maxitems = routes * (H / (DEC_WARPS * FRPW));
      if (grid > maxitems) grid = maxitems;
      dec_fused_direct_kernel<FRPW, 3><<<grid, DEC_THREADS, 0, stream>>>(
          x, p1, s1, a1, p2, s2, a2, ids, tw, routes, inter, y, out_pairs, bar);
    } else {
      // All grouped fallback shapes use the same occupancy-capped grid, and
      // the slot is zeroed before each launch, so no epoch is shared with
      // any of the four direct-decode grids.
      unsigned int* bar = bars + 4;
      cudaMemsetAsync(bar, 0, sizeof(unsigned int), stream);
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
    const int route_blocks =
        ((routes + THREADS - 1) / THREADS < 256) ? (routes + THREADS - 1) / THREADS : 256;
    if (T != 1024) count_routes_kernel<<<route_blocks, THREADS, 0, stream>>>(ids, routes, counts);
    if (T >= 8192) {
      constexpr int PM = 128;
      prefix_mma_routes_kernel<PM><<<1, 1, 0, stream>>>(counts, offsets, cursor, tiles1, tiles2);
      scatter_routes_kernel<<<route_blocks, THREADS, 0, stream>>>(ids, routes, offsets, cursor,
                                                                  route_list);
      const int max_mtiles = (routes + PM - 1) / PM + E;
      grouped_mma2_kernel<H, I, H / 16, I / GMMA_N1, GMMA_N1, 128, true, PM, 128>
          <<<max_mtiles*(I / GMMA_N1), 128, 0, stream>>>(x, p1, s1, a1, offsets, tiles1, route_list,
                                                         tw, nullptr, 0, inter, 0);
      grouped_mma2_kernel<I, H, I / 16, H / GMMA_N2, GMMA_N2, 256, false, PM, 128>
          <<<max_mtiles*(H / GMMA_N2), 256, 0, stream>>>(inter, p2, s2, a2, offsets, tiles2,
                                                         route_list, tw, nullptr, 0, route_out, 0);
    } else {
#define LAUNCH_FIXED_PREFILL(PFV, SK1V, SK2V)                                                      \
  do {                                                                                             \
    constexpr int PM = PFV;                                                                        \
    constexpr int SK1 = SK1V;                                                                      \
    constexpr int SK2 = SK2V;                                                                      \
    prefix_mma_routes_kernel<PM><<<1, 1, 0, stream>>>(counts, offsets, cursor, tiles1, tiles2);    \
    scatter_routes_kernel<<<route_blocks, THREADS, 0, stream>>>(ids, routes, offsets, cursor,      \
                                                                route_list);                       \
    const int max_mtiles = (routes + PM - 1) / PM + E;                                             \
    grouped_mma2_kernel<H, I, H / 16, I / GMMA_N1, GMMA_N1, 128, true, PM, SK1>                    \
        <<<max_mtiles*(I / GMMA_N1), 128, 0, stream>>>(x, p1, s1, a1, offsets, tiles1, route_list, \
                                                       tw, nullptr, 0, inter, 0);                  \
    grouped_mma2_kernel<I, H, I / 16, H / GMMA_N2, GMMA_N2, 256, false, PM, SK2>                   \
        <<<max_mtiles*(H / GMMA_N2), 256, 0, stream>>>(inter, p2, s2, a2, offsets, tiles2,         \
                                                       route_list, tw, nullptr, 0, route_out, 0);  \
  } while (0)
      if (T == 1024) {
        constexpr int PM = 32;
        bin_routes_kernel<<<route_blocks, THREADS, 0, stream>>>(ids, routes, counts, route_list);
        prefix_mma_routes_parallel_kernel<PM>
            <<<1, E, 0, stream>>>(counts, offsets, cursor, tiles1, tiles2);
        const int max_mtiles = (routes + PM - 1) / PM + E;
        grouped_mma2_kernel<H, I, H / 16, I / GMMA_N1, GMMA_N1, 128, true, PM, 128>
            <<<max_mtiles*(I / GMMA_N1), 128, 0, stream>>>(
                x, p1, s1, a1, offsets, tiles1, route_list, tw, y, (T * H) / 8, inter, routes);
        grouped_mma2_kernel<I, H, I / 16, H / GMMA_N2, GMMA_N2, 128, false, PM, 128, true>
            <<<max_mtiles*(H / GMMA_N2), 128, 0, stream>>>(inter, p2, s2, a2, offsets, tiles2,
                                                           route_list, tw, nullptr, 0, y, routes);
      } else {
        LAUNCH_FIXED_PREFILL(64, 128, 128);
      }
#undef LAUNCH_FIXED_PREFILL
    }
    const int out_elems = T * (H / 2);
    const int out_blocks =
        ((out_elems + THREADS - 1) / THREADS < 4096) ? (out_elems + THREADS - 1) / THREADS : 4096;
    if (T != 1024)
      combine_route_output_kernel<<<out_blocks, THREADS, 0, stream>>>(route_out, ids, tw, y, T);
  }
}
