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
void RunPersistentDirect(ffi::TensorView hidden_states, ffi::TensorView w1_packed,
                         ffi::TensorView w1_gs, ffi::TensorView w1_alpha, ffi::TensorView w2_packed,
                         ffi::TensorView w2_gs, ffi::TensorView w2_alpha, ffi::TensorView topk_ids,
                         ffi::TensorView topk_weights, ffi::TensorView out, int device,
                         cudaStream_t stream);
void RunFixedM64Prefill(ffi::TensorView hidden_states, ffi::TensorView w1_packed,
                        ffi::TensorView w1_gs, ffi::TensorView w1_alpha, ffi::TensorView w2_packed,
                        ffi::TensorView w2_gs, ffi::TensorView w2_alpha, ffi::TensorView topk_ids,
                        ffi::TensorView topk_weights, ffi::TensorView out, int device,
                        cudaStream_t stream);
void RunWidePrefill(ffi::TensorView hidden_states, ffi::TensorView w1_packed, ffi::TensorView w1_gs,
                    ffi::TensorView w1_alpha, ffi::TensorView w2_packed, ffi::TensorView w2_gs,
                    ffi::TensorView w2_alpha, ffi::TensorView topk_ids,
                    ffi::TensorView topk_weights, ffi::TensorView out, int device,
                    cudaStream_t stream);

namespace {

constexpr int E = 128;
constexpr int TOPK = 6;
constexpr int H = 2688;
constexpr int I = 1856;
constexpr int THREADS = 256;
constexpr int NT = 32;
constexpr int DECODE_NT = 32;
constexpr int DECODE_NT_MID = 48;
constexpr int DECODE_NT_SMALL = 16;
constexpr int DECODE_NT_LARGE = 64;
constexpr int DECODE_THREADS_SMALL = 128;
constexpr int KT = 256;
constexpr int MT = 8;
constexpr int W1_NTILES = I / NT;
constexpr int W2_NTILES = H / NT;
constexpr int MMA_M = 16;
constexpr int MMA_N = 64;
constexpr int MMA_K = 64;
constexpr int GMMA_K = 128;
constexpr int GMMA_N1_SMALL = 64;
constexpr int GMMA_N1_LARGE = 128;
constexpr int GMMA_N2_SMALL = 64;
constexpr int GMMA_N2_LARGE = 128;
constexpr int MMA_WARPS = MMA_N / 16;

struct Workspace {
  void* ptr = nullptr;
  size_t bytes = 0;
  int device = -1;
  std::mutex mutex;
};

Workspace g_workspace;
std::mutex g_run_mutex;
cudaStream_t g_last_stream = nullptr;
int g_last_stream_device = -1;
// The default stream is a valid previous stream, so track "have we run"
// separately instead of using nullptr as the sentinel.
bool g_have_last_stream = false;
// Buffers retired by workspace growth or a device change. A CUDA graph
// captured earlier may still replay kernels holding the old address, so
// retired buffers are kept alive for the process lifetime. Growth is
// monotone along the warmup ladder, so the retained total stays bounded
// (under 2x the final capacity per device).
std::vector<void*> g_retired_workspaces;

inline size_t align_up(size_t x, size_t a = 256) { return (x + a - 1) & ~(a - 1); }

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

__device__ __forceinline__ uint32_t fp8_e4m3fn_broadcast_bf16x2(uint8_t x) {
  const uint16_t packed = uint16_t(x) | (uint16_t(x) << 8);
  uint32_t bits;
  asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;\n" : "=r"(bits) : "h"(packed));
  return bits;
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

__device__ __forceinline__ void e2m1x8_to_half2x4(uint32_t q, __half2 (&out)[4]) {
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
      : "r"(q));
  out[0] = *reinterpret_cast<const __half2*>(&r0);
  out[1] = *reinterpret_cast<const __half2*>(&r1);
  out[2] = *reinterpret_cast<const __half2*>(&r2);
  out[3] = *reinterpret_cast<const __half2*>(&r3);
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

__device__ __forceinline__ void red_bf16x2_v2(__nv_bfloat162* dst, __nv_bfloat162 v0,
                                              __nv_bfloat162 v1) {
  const uint32_t a = *reinterpret_cast<const uint32_t*>(&v0);
  const uint32_t b = *reinterpret_cast<const uint32_t*>(&v1);
  asm volatile("red.global.add.noftz.v2.bf16x2 [%0], {%1, %2};\n"
               :
               : "l"(dst), "r"(a), "r"(b)
               : "memory");
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
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :
               : "r"(smem), "l"(src), "r"(src_bytes));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }

__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group 0;\n"); }

__device__ __forceinline__ uint64_t load_cg_l2_256_u64(const void* p) {
  uint64_t v;
  asm volatile("ld.global.cg.L2::256B.u64 %0, [%1];" : "=l"(v) : "l"(p));
  return v;
}

__device__ __forceinline__ uint4 load_cg_l2_256_u128(const void* p) {
  uint4 v;
  asm volatile("ld.global.cg.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

struct U64x2 {
  uint64_t x;
  uint64_t y;
};

__device__ __forceinline__ U64x2 load_cg_u128(const void* p) {
  U64x2 v;
  asm volatile("ld.global.cg.v2.u64 {%0, %1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p));
  return v;
}

__device__ __forceinline__ void load_bf16x16_as_half2x8(const __nv_bfloat16* p, __half2* out) {
  uint32_t w[8];
  asm volatile(
      "ld.global.L1::evict_last.v8.u32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
      : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3]), "=r"(w[4]), "=r"(w[5]), "=r"(w[6]),
        "=r"(w[7])
      : "l"(p));
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&w[i]);
    out[i] = __float22half2_rn(__bfloat1622float2(b));
  }
}

__device__ __forceinline__ void load_bf16x16_u32x8(const __nv_bfloat16* p, uint32_t* out) {
  asm volatile(
      "ld.global.L1::evict_last.v8.u32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
      : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3]), "=r"(out[4]), "=r"(out[5]),
        "=r"(out[6]), "=r"(out[7])
      : "l"(p));
}

__device__ __forceinline__ uint32_t fma_bf16x2(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

__device__ __forceinline__ float sum_bf16x2(uint32_t v) {
  return __uint_as_float(v << 16) + __uint_as_float(v & 0xffff0000u);
}

__device__ __forceinline__ void e2m1x8_to_bf16x2x4(uint32_t q, uint32_t* out) {
  asm volatile(
      "{\n"
      ".reg .b8 b0, b1, b2, b3;\n"
      "mov.b32 {b0, b1, b2, b3}, %4;\n"
      "cvt.rn.bf16x2.e2m1x2 %0, b0;\n"
      "cvt.rn.bf16x2.e2m1x2 %1, b1;\n"
      "cvt.rn.bf16x2.e2m1x2 %2, b2;\n"
      "cvt.rn.bf16x2.e2m1x2 %3, b3;\n"
      "}\n"
      : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
      : "r"(q));
}

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

template <int MM, int NT1, int NT2>
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
    tile1_sum += mtiles * NT1;
    tile2_sum += mtiles * NT2;
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
          bool ROUTE_OUTPUT = false, bool GRID_2D = false, bool BF16_ATOMIC = false,
          bool WIDE = false, bool PDL_WAIT = false, bool PDL_SIGNAL = false,
          int PDL_WARM_BLOCKS = 752>
__global__ __launch_bounds__(256, 4) void decode_group16_half_kernel(
    const __nv_bfloat16* __restrict__ input, const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales, const uint8_t* __restrict__ next_packed,
    const uint8_t* __restrict__ next_scales, const float* __restrict__ alpha,
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
  // Issue the routing lookup before the output-zeroing stores so the
  // ids -> weight-address -> first weight load chain overlaps the zero pass
  // instead of starting after it.  For the PDL consumer the same lookup (and
  // a warm-up of this block's own weight tile) runs ahead of the dependency
  // wait: neither topk_ids nor the packed weights are written by the producer.
  const int r = ORDERED ? route_order[ridx] : ridx;
  const int e = ids[r];
  if constexpr (PDL_WAIT) {
    if (unsigned(e) < E) {
      const int blk = int(blockIdx.x) + int(gridDim.x) * int(blockIdx.y);
      if (blk < PDL_WARM_BLOCKS) {
        const uint8_t* wp = packed + (size_t(e) * N + size_t(ntile) * TILE_N) * (K / 2);
        constexpr int WB = TILE_N * (K / 2);
        for (int o = int(threadIdx.x) * 128; o < WB; o += int(blockDim.x) * 128)
          asm volatile("prefetch.global.L2 [%0];" : : "l"(wp + o));
      }
    }
    cudaGridDependencySynchronize();
  }
  if constexpr (FIRST && BF16_ATOMIC) {
    const int block = int(blockIdx.x) + int(gridDim.x) * int(blockIdx.y);
    const int blocks = int(gridDim.x) * int(gridDim.y);
    const int total = (routes / TOPK) * (H / 2);
    auto* out32 = reinterpret_cast<uint32_t*>(output_acc);
    for (int x = block * blockDim.x + threadIdx.x; x < total; x += blocks * blockDim.x) {
      out32[x] = 0u;
    }
  }
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
    if constexpr (!WIDE) {
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
    }

    const int groups = this_k / 16;
    if constexpr (WIDE) {
      for (int group_base = 0; group_base < groups; group_base += 64) {
        const int group = group_base + lane * 2;
        if (group + 1 < groups) {
          uint32_t xv0[8];
          uint32_t xv1[8];
          const __nv_bfloat16* xg = input + size_t(input_row) * K + kt + group * 16;
          load_bf16x16_u32x8(xg, xv0);
          load_bf16x16_u32x8(xg + 16, xv1);
#pragma unroll
          for (int j = 0; j < OPW; ++j) {
            const int n = nbase + j;
            const size_t row = size_t(e) * N + n;
            const int kg = (kt >> 4) + group;
            const U64x2 q16 = load_cg_u128(packed + row * KHALF + size_t(kg) * 8);
            const uint64_t qlo = q16.x;
            const uint64_t qhi = q16.y;
            uint32_t hsum0 = 0u;
            uint32_t hsum1 = 0u;
            uint32_t wv[4];
            e2m1x8_to_bf16x2x4(uint32_t(qlo), wv);
#pragma unroll
            for (int p = 0; p < 4; ++p) hsum0 = fma_bf16x2(xv0[p], wv[p], hsum0);
            e2m1x8_to_bf16x2x4(uint32_t(qlo >> 32), wv);
#pragma unroll
            for (int p = 0; p < 4; ++p) hsum0 = fma_bf16x2(xv0[p + 4], wv[p], hsum0);
            e2m1x8_to_bf16x2x4(uint32_t(qhi), wv);
#pragma unroll
            for (int p = 0; p < 4; ++p) hsum1 = fma_bf16x2(xv1[p], wv[p], hsum1);
            e2m1x8_to_bf16x2x4(uint32_t(qhi >> 32), wv);
#pragma unroll
            for (int p = 0; p < 4; ++p) hsum1 = fma_bf16x2(xv1[p + 4], wv[p], hsum1);
            const float2 sf =
                fp8_e4m3fnx2_to_float2(*reinterpret_cast<const uint16_t*>(scales + row * KG + kg));
            acc[j] = fmaf(sum_bf16x2(hsum0), sf.x, acc[j]);
            acc[j] = fmaf(sum_bf16x2(hsum1), sf.y, acc[j]);
          }
        }
      }
    } else {
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
            const void* qptr = packed + row * KHALF + size_t(kg) * 8;
            const uint64_t q8 =
                TILE_N >= 32 ? load_cg_l2_256_u64(qptr) : *reinterpret_cast<const uint64_t*>(qptr);
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
    }
    if constexpr (!WIDE) __syncthreads();
  }

  if constexpr (!FIRST && BF16_ATOMIC) {
    float red[OPW];
#pragma unroll
    for (int j = 0; j < OPW; ++j) red[j] = warp_sum(acc[j]);
    if (lane == 0) {
      const float scale = alpha[e] * topk_weights[r];
      auto* dst =
          reinterpret_cast<__nv_bfloat162*>(output_acc) + size_t(r / TOPK) * (H / 2) + nbase / 2;
#pragma unroll
      for (int j = 0; j < OPW; j += 4) {
        const __nv_bfloat162 v0 = __floats2bfloat162_rn(red[j] * scale, red[j + 1] * scale);
        const __nv_bfloat162 v1 = __floats2bfloat162_rn(red[j + 2] * scale, red[j + 3] * scale);
        red_bf16x2_v2(dst + j / 2, v0, v1);
      }
    }
  } else {
    float red[OPW];
#pragma unroll
    for (int j = 0; j < OPW; ++j) red[j] = warp_sum(acc[j]);
    if (lane == 0) {
      const float al = alpha[e];
      if constexpr (FIRST && OPW == 4) {
        float v0 = fmaxf(red[0] * al, 0.0f);
        float v1 = fmaxf(red[1] * al, 0.0f);
        float v2 = fmaxf(red[2] * al, 0.0f);
        float v3 = fmaxf(red[3] * al, 0.0f);
        union PackBF16x2 {
          __nv_bfloat162 bf;
          uint32_t u;
        } p01, p23;
        p01.bf = __floats2bfloat162_rn(v0 * v0, v1 * v1);
        p23.bf = __floats2bfloat162_rn(v2 * v2, v3 * v3);
        *reinterpret_cast<uint2*>(intermediate + size_t(r) * N + nbase) = make_uint2(p01.u, p23.u);
        // Release this warp's intermediate tile to device scope as soon as it
        // is written, so the PDL completion signal below is ordered after it
        // without a block-wide fence on the critical path.
        if constexpr (PDL_SIGNAL) __threadfence();
      } else if constexpr (!FIRST && ROUTE_OUTPUT && OPW == 4) {
        union PackBF16x2 {
          __nv_bfloat162 bf;
          uint32_t u;
        } p01, p23;
        p01.bf = __floats2bfloat162_rn(red[0] * al, red[1] * al);
        p23.bf = __floats2bfloat162_rn(red[2] * al, red[3] * al);
        *reinterpret_cast<uint2*>(intermediate + size_t(r) * N + nbase) = make_uint2(p01.u, p23.u);
      } else {
#pragma unroll
        for (int j = 0; j < OPW; ++j) {
          float v = red[j] * al;
          const int n = nbase + j;
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
  }

  if constexpr (FIRST && WIDE) {
    const int block = int(blockIdx.x) + int(gridDim.x) * int(blockIdx.y);
    const int blocks = int(gridDim.x) * int(gridDim.y);
    constexpr int PREFETCH_BLOCKS = 150;
    if (block >= blocks - PREFETCH_BLOCKS) {
      const int item = block - (blocks - PREFETCH_BLOCKS);
      const int proute = item % routes;
      const int ptile = item / routes;
      const int pe = ids[proute];
      if (unsigned(pe) < E && ptile < H / TILE_N) {
        const uint8_t* wp = next_packed + (size_t(pe) * H + size_t(ptile) * TILE_N) * (I / 2);
        constexpr int WB = TILE_N * (I / 2);
        for (int o = int(threadIdx.x) * 128; o < WB; o += int(blockDim.x) * 128)
          asm volatile("prefetch.global.L2 [%0];" : : "l"(wp + o));
        const uint8_t* sp = next_scales + (size_t(pe) * H + size_t(ptile) * TILE_N) * (I / 16);
        constexpr int SB = TILE_N * (I / 16);
        for (int o = int(threadIdx.x) * 128; o < SB; o += int(blockDim.x) * 128)
          asm volatile("prefetch.global.L2 [%0];" : : "l"(sp + o));
      }
    }
  }

  if constexpr (PDL_SIGNAL) {
    __syncthreads();
    if (threadIdx.x == 0) cudaTriggerProgrammaticLaunchCompletion();
  }
}

template <int K, int N, int KG, int TILE_N>
__global__ __launch_bounds__(256, 2) void decode_fc1_pair_half_kernel(
    const __nv_bfloat16* __restrict__ input, const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales, const float* __restrict__ alpha,
    const int32_t* __restrict__ ids, __nv_bfloat16* __restrict__ intermediate,
    __nv_bfloat16* __restrict__ output) {
  constexpr int OPW = TILE_N / 8;
  constexpr int KHALF = K / 2;
  static_assert(OPW == 4);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int r0 = int(blockIdx.x) * 2;
  const int ntile = int(blockIdx.y);
  const int nbase = ntile * TILE_N + warp * OPW;
  const int e0 = ids[r0];
  const int e1 = ids[r0 + 1];
  const __nv_bfloat16* x = input + size_t(r0 / TOPK) * K;
  const int block = int(blockIdx.x) + int(gridDim.x) * int(blockIdx.y);
  const int blocks = int(gridDim.x) * int(gridDim.y);
  auto* out32 = reinterpret_cast<uint32_t*>(output);
  for (int i = block * blockDim.x + threadIdx.x; i < 16 * (H / 2); i += blocks * blockDim.x) {
    out32[i] = 0u;
  }
  float acc0[OPW];
  float acc1[OPW];
#pragma unroll
  for (int j = 0; j < OPW; ++j) {
    acc0[j] = 0.0f;
    acc1[j] = 0.0f;
  }

  constexpr int GROUPS = K / 16;
  for (int group_base = 0; group_base < GROUPS; group_base += 64) {
    const int group = group_base + lane * 2;
    if (group + 1 < GROUPS) {
      __half2 xv0[8];
      __half2 xv1[8];
      const __nv_bfloat16* xg = x + group * 16;
      load_bf16x16_as_half2x8(xg, xv0);
      load_bf16x16_as_half2x8(xg + 16, xv1);
#pragma unroll
      for (int j = 0; j < OPW; ++j) {
        const int n = nbase + j;
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
          const int e = rr == 0 ? e0 : e1;
          if (unsigned(e) < E) {
            const size_t row = size_t(e) * N + n;
            const int kg = group;
            const U64x2 q16 = load_cg_u128(packed + row * KHALF + size_t(kg) * 8);
            __half2 hsum0 = __float2half2_rn(0.0f);
            __half2 hsum1 = __float2half2_rn(0.0f);
            __half2 wv[4];
            e2m1x8_to_half2x4(uint32_t(q16.x), wv);
#pragma unroll
            for (int p = 0; p < 4; ++p) hsum0 = __hfma2(xv0[p], wv[p], hsum0);
            e2m1x8_to_half2x4(uint32_t(q16.x >> 32), wv);
#pragma unroll
            for (int p = 0; p < 4; ++p) hsum0 = __hfma2(xv0[p + 4], wv[p], hsum0);
            e2m1x8_to_half2x4(uint32_t(q16.y), wv);
#pragma unroll
            for (int p = 0; p < 4; ++p) hsum1 = __hfma2(xv1[p], wv[p], hsum1);
            e2m1x8_to_half2x4(uint32_t(q16.y >> 32), wv);
#pragma unroll
            for (int p = 0; p < 4; ++p) hsum1 = __hfma2(xv1[p + 4], wv[p], hsum1);
            const float2 pair0 = __half22float2(hsum0);
            const float2 pair1 = __half22float2(hsum1);
            const float2 sf =
                fp8_e4m3fnx2_to_float2(*reinterpret_cast<const uint16_t*>(scales + row * KG + kg));
            float& dst = rr == 0 ? acc0[j] : acc1[j];
            dst = fmaf(pair0.x + pair0.y, sf.x, dst);
            dst = fmaf(pair1.x + pair1.y, sf.y, dst);
          }
        }
      }
    }
  }

  float red0[OPW];
  float red1[OPW];
#pragma unroll
  for (int j = 0; j < OPW; ++j) {
    red0[j] = warp_sum(acc0[j]);
    red1[j] = warp_sum(acc1[j]);
  }
  if (lane == 0) {
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
      const int r = r0 + rr;
      const int e = rr == 0 ? e0 : e1;
      const float* red = rr == 0 ? red0 : red1;
      float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
      if (unsigned(e) < E) {
        const float al = alpha[e];
        v0 = fmaxf(red[0] * al, 0.0f);
        v1 = fmaxf(red[1] * al, 0.0f);
        v2 = fmaxf(red[2] * al, 0.0f);
        v3 = fmaxf(red[3] * al, 0.0f);
      }
      union PackBF16x2 {
        __nv_bfloat162 bf;
        uint32_t u;
      } p01, p23;
      p01.bf = __floats2bfloat162_rn(v0 * v0, v1 * v1);
      p23.bf = __floats2bfloat162_rn(v2 * v2, v3 * v3);
      *reinterpret_cast<uint2*>(intermediate + size_t(r) * N + nbase) = make_uint2(p01.u, p23.u);
    }
  }
}

template <int K, int N, int KG, int NTILES, int TILE_N>
__global__ __launch_bounds__(256, 2) void decode_fc2_token_kernel(
    const __nv_bfloat16* __restrict__ input, const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales, const float* __restrict__ alpha,
    const int32_t* __restrict__ ids, const float* __restrict__ topk_weights, int tokens,
    __nv_bfloat16* __restrict__ output) {
  constexpr int OPW = TILE_N / 8;
  constexpr int KHALF = K / 2;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int ntile = int(blockIdx.x) / tokens;
  const int tok = int(blockIdx.x) - ntile * tokens;
  const int nbase = ntile * TILE_N + warp * OPW;
  __shared__ __align__(16) __half sx[K];
  float total[OPW];
#pragma unroll
  for (int j = 0; j < OPW; ++j) total[j] = 0.0f;

#pragma unroll
  for (int slot = 0; slot < TOPK; ++slot) {
    const int r = tok * TOPK + slot;
    const int e = ids[r];
    if (unsigned(e) < E) {
      for (int p = threadIdx.x; p < K / 2; p += blockDim.x) {
        const __nv_bfloat162 xb =
            *reinterpret_cast<const __nv_bfloat162*>(input + size_t(r) * K + p * 2);
        const int group = p >> 3;
        const int in_group = p & 7;
        const int logical_segment = group * 2 + (in_group >> 2);
        const int physical_segment = logical_segment ^ (group & 7);
        const int physical_pair = physical_segment * 4 + (in_group & 3);
        reinterpret_cast<__half2*>(sx)[physical_pair] = __float22half2_rn(__bfloat1622float2(xb));
      }
      __syncthreads();

      float acc[OPW];
#pragma unroll
      for (int j = 0; j < OPW; ++j) acc[j] = 0.0f;
      constexpr int GROUPS = K / 16;
      for (int group_base = 0; group_base < GROUPS; group_base += 32) {
        const int group = group_base + lane;
        if (group < GROUPS) {
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
            const uint64_t q8 =
                *reinterpret_cast<const uint64_t*>(packed + row * KHALF + size_t(group) * 8);
            __half2 hsum = __float2half2_rn(0.0f);
#pragma unroll
            for (int p = 0; p < 8; ++p) {
              hsum = __hfma2(xv[p], e2m1x2_to_half2(uint8_t(q8 >> (p * 8))), hsum);
            }
            const float2 pair = __half22float2(hsum);
            const float sf = fp8_e4m3fnx2_to_float2(uint16_t(scales[row * KG + group])).x;
            acc[j] = fmaf(pair.x + pair.y, sf, acc[j]);
          }
        }
      }
      const float route_scale = alpha[e] * topk_weights[r];
#pragma unroll
      for (int j = 0; j < OPW; ++j) {
        const float v = warp_sum(acc[j]);
        if (lane == 0) total[j] = fmaf(v, route_scale, total[j]);
      }
      __syncthreads();
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int j = 0; j < OPW; j += 2) {
      *reinterpret_cast<__nv_bfloat162*>(output + size_t(tok) * N + nbase + j) =
          __floats2bfloat162_rn(total[j], total[j + 1]);
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

template <int K, int N, int KG, int NTILES, int TILE_N, int BLOCK_THREADS, int MM, int NS,
          bool FOLD_SCALE, bool FIRST, bool ATOMIC_COMBINE>
__global__ __launch_bounds__(BLOCK_THREADS, 2) void grouped_mma_bf16_kernel(
    const __nv_bfloat16* __restrict__ input, const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales, const float* __restrict__ alpha,
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ tiles,
    const int32_t* __restrict__ route_list, const float* __restrict__ topk_weights,
    __nv_bfloat16* __restrict__ output) {
  if (int(blockIdx.x) >= tiles[E]) return;
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

  __shared__ __align__(16) uint32_t as[NS][MM][GMMA_K / 2];
  __shared__ __align__(16) uint32_t bs[NS][TILE_N][GMMA_K / 8];
  __shared__ __align__(16) uint32_t ss[NS][TILE_N][GMMA_K / 64];
  __shared__ int sroutes[MM];

  for (int tt = tid; tt < MM; tt += BLOCK_THREADS) {
    const int grouped_row = offsets[e] + m0 + tt;
    sroutes[tt] = tt < valid_m ? route_list[grouped_row] : -1;
  }
  __syncthreads();

  constexpr int MA = MM / 16;
  float acc[MA][2][4];
#pragma unroll
  for (int mi = 0; mi < MA; ++mi)
#pragma unroll
    for (int atom = 0; atom < 2; ++atom)
#pragma unroll
      for (int x = 0; x < 4; ++x) acc[mi][atom][x] = 0.0f;

  constexpr int KHALF = K / 2;
  constexpr int NKS = (K + GMMA_K - 1) / GMMA_K;

#define STAGE_KCHUNK(KI, BUF)                                                              \
  {                                                                                        \
    const int k0 = (KI) * GMMA_K;                                                          \
    const int stage_k = min(GMMA_K, K - k0);                                               \
    for (int idx = tid; idx < MM * (GMMA_K / 8); idx += BLOCK_THREADS) {                   \
      const int m = idx / (GMMA_K / 8);                                                    \
      const int chunk = idx - m * (GMMA_K / 8);                                            \
      const int r = sroutes[m];                                                            \
      const void* src = input;                                                             \
      int src_bytes = 0;                                                                   \
      if (r >= 0 && chunk * 8 < stage_k) {                                                 \
        const int input_row = FIRST ? (r / TOPK) : (offsets[e] + m0 + m);                  \
        src = input + size_t(input_row) * K + k0 + chunk * 8;                              \
        src_bytes = 16;                                                                    \
      }                                                                                    \
      const int pair = chunk * 4;                                                          \
      cp_async_16(&as[BUF][m][pair ^ ((m & 7) << 2)], src, src_bytes);                     \
    }                                                                                      \
    for (int idx = tid; idx < TILE_N * (GMMA_K / 32); idx += BLOCK_THREADS) {              \
      const int nr = idx / (GMMA_K / 32);                                                  \
      const int chunk = idx - nr * (GMMA_K / 32);                                          \
      const int n = ntile * TILE_N + nr;                                                   \
      const size_t row = size_t(e) * N + (n < N ? n : (N - 1));                            \
      const void* src = packed;                                                            \
      int src_bytes = 0;                                                                   \
      if (chunk * 32 < stage_k && n < N) {                                                 \
        src = packed + row * KHALF + (k0 >> 1) + chunk * 16;                               \
        src_bytes = 16;                                                                    \
      }                                                                                    \
      const int physical_segment = chunk ^ ((nr >> 1) & 3);                                \
      cp_async_16(&bs[BUF][nr][physical_segment * 4], src, src_bytes);                     \
    }                                                                                      \
    cp_async_commit();                                                                     \
    for (int idx = tid; idx < TILE_N * (GMMA_K / 64); idx += BLOCK_THREADS) {              \
      const int nr = idx / (GMMA_K / 64);                                                  \
      const int word = idx - nr * (GMMA_K / 64);                                           \
      const int n = ntile * TILE_N + nr;                                                   \
      const size_t row = size_t(e) * N + (n < N ? n : (N - 1));                            \
      uint32_t sv = 0;                                                                     \
      if (word * 64 < stage_k && n < N) {                                                  \
        sv = *reinterpret_cast<const uint32_t*>(scales + row * KG + (k0 >> 4) + word * 4); \
      }                                                                                    \
      ss[BUF][nr][word] = sv;                                                              \
    }                                                                                      \
  }

  STAGE_KCHUNK(0, 0)

  for (int ki = 0; ki < NKS; ++ki) {
    const int buf = (NS == 2) ? (ki & 1) : 0;
    if (NS == 2 && ki + 1 < NKS) {
      STAGE_KCHUNK(ki + 1, (NS == 2) ? ((ki + 1) & 1) : 0)
      asm volatile("cp.async.wait_group 1;\n");
    } else {
      asm volatile("cp.async.wait_group 0;\n");
    }
    __syncthreads();

#pragma unroll 1
    for (int sk = 0; sk < GMMA_K / 16; ++sk) {
      if constexpr (FOLD_SCALE) {
        uint32_t bb0[2], bb1[2];
#pragma unroll
        for (int atom = 0; atom < 2; ++atom) {
          const int nr = warp * 16 + atom * 8 + g;
          const int physical_segment = (sk >> 1) ^ ((nr >> 1) & 3);
          const int word = physical_segment * 4 + (sk & 1) * 2;
          const uint32_t p0 = bs[buf][nr][word];
          const uint32_t p1 = bs[buf][nr][word + 1];
          uint32_t b0 = e2m1x2_to_bf16x2(uint8_t(p0 >> (q * 8)));
          uint32_t b1 = e2m1x2_to_bf16x2(uint8_t(p1 >> (q * 8)));
          const uint8_t sbyte = uint8_t(ss[buf][nr][sk >> 2] >> ((sk & 3) * 8));
          const uint32_t sbits = fp8_e4m3fn_broadcast_bf16x2(sbyte);
          asm("mul.rn.bf16x2 %0, %0, %1;" : "+r"(b0) : "r"(sbits));
          asm("mul.rn.bf16x2 %0, %0, %1;" : "+r"(b1) : "r"(sbits));
          bb0[atom] = b0;
          bb1[atom] = b1;
        }
#pragma unroll
        for (int mi = 0; mi < MA; ++mi) {
          if (mi < active_atoms) {
            const int row = mi * 16 + (lane & 15);
            const int logical_segment = sk * 2 + (lane >> 4);
            const int physical_segment = logical_segment ^ (row & 7);
            uint32_t a0, a1, a2, a3;
            ldmatrix_m8n8x4_b16(a0, a1, a2, a3, &as[buf][row][physical_segment * 4]);
#pragma unroll
            for (int atom = 0; atom < 2; ++atom) {
              mma_m16n8k16_bf16(acc[mi][atom][0], acc[mi][atom][1], acc[mi][atom][2],
                                acc[mi][atom][3], a0, a1, a2, a3, bb0[atom], bb1[atom]);
            }
          }
        }
      } else {
#pragma unroll
        for (int atom = 0; atom < 2; ++atom) {
          const int nr = warp * 16 + atom * 8 + g;
          const int physical_segment = (sk >> 1) ^ ((nr >> 1) & 3);
          const int word = physical_segment * 4 + (sk & 1) * 2;
          const uint32_t p0 = bs[buf][nr][word];
          const uint32_t p1 = bs[buf][nr][word + 1];
          const uint32_t b0 = e2m1x2_to_bf16x2(uint8_t(p0 >> (q * 8)));
          const uint32_t b1 = e2m1x2_to_bf16x2(uint8_t(p1 >> (q * 8)));
          const int nr0 = warp * 16 + atom * 8 + q * 2;
          const int shift = (sk & 3) * 8;
          const uint16_t sps = uint16_t(uint8_t(ss[buf][nr0][sk >> 2] >> shift)) |
                               (uint16_t(uint8_t(ss[buf][nr0 + 1][sk >> 2] >> shift)) << 8);
          const float2 sf = fp8_e4m3fnx2_to_float2(sps);
#pragma unroll
          for (int mi = 0; mi < MA; ++mi) {
            if (mi < active_atoms) {
              const int row = mi * 16 + (lane & 15);
              const int logical_segment = sk * 2 + (lane >> 4);
              const int physical_segment = logical_segment ^ (row & 7);
              uint32_t a0, a1, a2, a3;
              ldmatrix_m8n8x4_b16(a0, a1, a2, a3, &as[buf][row][physical_segment * 4]);
              float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
              mma_m16n8k16_bf16(t0, t1, t2, t3, a0, a1, a2, a3, b0, b1);
              acc[mi][atom][0] = fmaf(t0, sf.x, acc[mi][atom][0]);
              acc[mi][atom][1] = fmaf(t1, sf.y, acc[mi][atom][1]);
              acc[mi][atom][2] = fmaf(t2, sf.x, acc[mi][atom][2]);
              acc[mi][atom][3] = fmaf(t3, sf.y, acc[mi][atom][3]);
            }
          }
        }
      }
    }
    __syncthreads();
    if (NS == 1 && ki + 1 < NKS) {
      STAGE_KCHUNK(ki + 1, 0)
    }
  }
#undef STAGE_KCHUNK

  const float al = alpha[e];
#pragma unroll
  for (int mi = 0; mi < MA; ++mi)
#pragma unroll
    for (int atom = 0; atom < 2; ++atom) {
      const int n0 = ntile * TILE_N + warp * 16 + atom * 8 + q * 2;
      const int rows[2] = {mi * 16 + g, mi * 16 + g + 8};
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        const int m = rows[rr];
        if (m < valid_m && n0 < N) {
          const int grouped_row = offsets[e] + m0 + m;
          const int r = sroutes[m];
          float v0 = acc[mi][atom][rr * 2] * al;
          float v1 = acc[mi][atom][rr * 2 + 1] * al;
          if constexpr (FIRST) {
            v0 = fmaxf(v0, 0.0f);
            v1 = fmaxf(v1, 0.0f);
            *reinterpret_cast<__nv_bfloat162*>(output + size_t(grouped_row) * N + n0) =
                __floats2bfloat162_rn(v0 * v0, v1 * v1);
          } else {
            if constexpr (ATOMIC_COMBINE) {
              const float w = topk_weights[r];
              atomicAdd(reinterpret_cast<__nv_bfloat162*>(output + size_t(r / TOPK) * N + n0),
                        __floats2bfloat162_rn(v0 * w, v1 * w));
            } else {
              *reinterpret_cast<__nv_bfloat162*>(output + size_t(r) * N + n0) =
                  __floats2bfloat162_rn(v0, v1);
            }
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

}  // namespace

void Run(ffi::TensorView hidden_states, ffi::TensorView w1_packed, ffi::TensorView w1_gs,
         ffi::TensorView w1_alpha, ffi::TensorView w2_packed, ffi::TensorView w2_gs,
         ffi::TensorView w2_alpha, ffi::TensorView topk_ids, ffi::TensorView topk_weights,
         ffi::TensorView out) {
  const int device = hidden_states.device().device_id;
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, device));
  // All specialized paths reuse process-global scratch. Serialize their host
  // launch sequences and order transitions between current CUDA streams.
  std::lock_guard<std::mutex> run_lock(g_run_mutex);
  if (g_have_last_stream && (g_last_stream != stream || g_last_stream_device != device)) {
    cudaSetDevice(g_last_stream_device);
    cudaStreamSynchronize(g_last_stream);
    cudaSetDevice(device);
  }
  g_last_stream = stream;
  g_last_stream_device = device;
  g_have_last_stream = true;

  const int T = int(hidden_states.size(0));
  if (T == 1) {
    RunPersistentDirect(hidden_states, w1_packed, w1_gs, w1_alpha, w2_packed, w2_gs, w2_alpha,
                        topk_ids, topk_weights, out, device, stream);
    return;
  }
  if (T == 64) {
    RunFastDirect(hidden_states, w1_packed, w1_gs, w1_alpha, w2_packed, w2_gs, w2_alpha, topk_ids,
                  topk_weights, out, stream);
    return;
  }
  if (T == 1024) {
    RunFixedM64Prefill(hidden_states, w1_packed, w1_gs, w1_alpha, w2_packed, w2_gs, w2_alpha,
                       topk_ids, topk_weights, out, device, stream);
    return;
  }
  if (T > 64) {
    RunWidePrefill(hidden_states, w1_packed, w1_gs, w1_alpha, w2_packed, w2_gs, w2_alpha, topk_ids,
                   topk_weights, out, device, stream);
    return;
  }
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
    // T=4 now shares the T=16 GRID_2D + PDL launch; keep the older narrow-tile
    // path compiled but unselected.
    constexpr bool kDecodeT4Legacy = false;
    if (kDecodeT4Legacy && T == 4) {
      decode_group16_half_kernel<H, I, H / 16, I / DECODE_NT_SMALL, DECODE_NT_SMALL, true, false,
                                 false, false, true>
          <<<routes*(I / DECODE_NT_SMALL), THREADS, 0, stream>>>(x, p1, s1, nullptr, nullptr, a1,
                                                                 ids, nullptr, tw, routes, inter,
                                                                 reinterpret_cast<float*>(y));
      decode_group16_half_kernel<I, H, I / 16, H / DECODE_NT_SMALL, DECODE_NT_SMALL, false, false,
                                 false, false, true>
          <<<routes*(H / DECODE_NT_SMALL), THREADS, 0, stream>>>(
              inter, p2, s2, nullptr, nullptr, a2, ids, nullptr, tw, routes, route_out,
              reinterpret_cast<float*>(y));
    } else if (T == 16 || T == 4) {
      cudaLaunchAttribute attr{};
      attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr.val.programmaticStreamSerializationAllowed = 1;
      cudaLaunchConfig_t cfg{};
      cfg.blockDim = dim3(THREADS);
      cfg.dynamicSmemBytes = 0;
      cfg.stream = stream;
      cfg.attrs = &attr;
      cfg.numAttrs = 1;
      cfg.gridDim = dim3(routes, I / DECODE_NT);
      cudaLaunchKernelEx(&cfg,
                         decode_group16_half_kernel<H, I, H / 16, I / DECODE_NT, DECODE_NT, true,
                                                    false, false, true, true, true, false, true>,
                         x, p1, s1, p2, s2, a1, ids, nullptr, tw, routes, inter,
                         reinterpret_cast<float*>(y));
      cfg.gridDim = dim3(routes, H / DECODE_NT);
      cudaLaunchKernelEx(&cfg,
                         decode_group16_half_kernel<I, H, I / 16, H / DECODE_NT, DECODE_NT, false,
                                                    false, false, true, true, true, true, false>,
                         inter, p2, s2, nullptr, nullptr, a2, ids, nullptr, tw, routes, route_out,
                         reinterpret_cast<float*>(y));
    } else if (T == 64) {
      decode_group16_half_kernel<H, I, H / 16, I / DECODE_NT_LARGE, DECODE_NT_LARGE, true, false,
                                 false, true>
          <<<dim3(routes, I / DECODE_NT_LARGE), THREADS, 0, stream>>>(
              x, p1, s1, nullptr, nullptr, a1, ids, nullptr, tw, routes, inter, output_acc);
      decode_group16_half_kernel<I, H, I / 16, H / DECODE_NT_LARGE, DECODE_NT_LARGE, false, false,
                                 true, true>
          <<<dim3(routes, H / DECODE_NT_LARGE), THREADS, 0, stream>>>(
              inter, p2, s2, nullptr, nullptr, a2, ids, nullptr, tw, routes, route_out, output_acc);
    } else {
      decode_group16_half_kernel<H, I, H / 16, I / DECODE_NT, DECODE_NT, true, false, false, false,
                                 false, true><<<routes*(I / DECODE_NT), THREADS, 0, stream>>>(
          x, p1, s1, p2, s2, a1, ids, nullptr, tw, routes, inter, output_acc);
      decode_group16_half_kernel<I, H, I / 16, H / DECODE_NT, DECODE_NT, false, false, true, false,
                                 false, true><<<routes*(H / DECODE_NT), THREADS, 0, stream>>>(
          inter, p2, s2, nullptr, nullptr, a2, ids, nullptr, tw, routes, route_out, output_acc);
    }
    if (T != 4 && T != 16) {
      const int out_elems = T * (H / 2);
      const int out_blocks = (out_elems + THREADS - 1) / THREADS;
      combine_route_output_kernel<<<out_blocks, THREADS, 0, stream>>>(route_out, ids, tw, y, T);
    }
  } else {
    cudaMemsetAsync(counts, 0, E * sizeof(int32_t), stream);
    cudaMemsetAsync(y, 0, size_t(T) * H * sizeof(__nv_bfloat16), stream);
    const int route_blocks =
        ((routes + THREADS - 1) / THREADS < 256) ? (routes + THREADS - 1) / THREADS : 256;
    count_routes_kernel<<<route_blocks, THREADS, 0, stream>>>(ids, routes, counts);
#define PREFILL_LAUNCH(GMM, GN1, GN2, FOLD)                                                      \
  do {                                                                                           \
    prefix_mma_routes_kernel<GMM, (I + GN1 - 1) / GN1, (H + GN2 - 1) / GN2>                      \
        <<<1, 1, 0, stream>>>(counts, offsets, cursor, tiles1, tiles2);                          \
    scatter_routes_kernel<<<route_blocks, THREADS, 0, stream>>>(ids, routes, offsets, cursor,    \
                                                                route_list);                     \
    const int max_mtiles = (routes + GMM - 1) / GMM + E;                                         \
    constexpr int NT1C = (I + GN1 - 1) / GN1;                                                    \
    constexpr int BT1C = (GN1 == 64 ? 128 : 256);                                                \
    grouped_mma_bf16_kernel<H, I, H / 16, NT1C, GN1, BT1C, GMM, 1, FOLD, true, false>            \
        <<<max_mtiles * NT1C, BT1C, 0, stream>>>(x, p1, s1, a1, offsets, tiles1, route_list,     \
                                                 nullptr, inter);                                \
    constexpr int NT2C = (H + GN2 - 1) / GN2;                                                    \
    constexpr int BT2C = (GN2 == 64 ? 128 : 256);                                                \
    grouped_mma_bf16_kernel<I, H, I / 16, NT2C, GN2, BT2C, GMM, 1, FOLD, false, true>            \
        <<<max_mtiles * NT2C, BT2C, 0, stream>>>(inter, p2, s2, a2, offsets, tiles2, route_list, \
                                                 tw, y);                                         \
  } while (0)

    if (T < 2048) {
      PREFILL_LAUNCH(32, GMMA_N1_SMALL, GMMA_N2_LARGE, false);
    } else if (T < 8192) {
      PREFILL_LAUNCH(96, GMMA_N1_LARGE, GMMA_N2_LARGE, true);
    } else {
      PREFILL_LAUNCH(128, GMMA_N1_LARGE, GMMA_N2_LARGE, true);
    }
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, Run);
