// DSV4.1 sparse MLA decode (bf16 dense KV pools), fused gather + flash attention.
// Orientation: S^T[col][head] = KV(m=col,k=d) x Q^T(k=d,n=head) so the KV tile is the
// mma A-operand (read once per tile) and Q^T lives in registers.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#ifdef ENABLE_G4
#include <cuda.h>
#endif
#include <cooperative_groups.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/device_guard.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cfloat>
#include <cstdint>
#include <mutex>

namespace ffi = tvm::ffi;
namespace cg = cooperative_groups;

#ifdef ENABLE_G4
#define G4_PARAM \
  , const __grid_constant__ CUtensorMap swa_tm, const __grid_constant__ CUtensorMap ckv_tm
#define G4_ARG , swa_tm, ckv_tm
#else
#define G4_PARAM
#define G4_ARG
#endif

#define HEADS 16
#define DIM 512
#define WIDTH 640
#define LOG2E 1.4426950408889634f
#define QKSCALE (0.044194173824159216f * LOG2E)

#define SM_Q 0
#define SM_KV 16384
#define SCORE_QSTRIDE (32 * 18)
#define SM_S_FOR(T_) (SM_KV + 2 * (T_) * 1024)
#define SCORE_BYTES_FOR(T_) (4 * (((T_) * 18 > SCORE_QSTRIDE) ? (T_) * 18 : SCORE_QSTRIDE) * 4)
#define SM_P_FOR(T_) (SM_S_FOR(T_) + SCORE_BYTES_FOR(T_))
#define SM_ALPHA_FOR(T_) (SM_P_FOR(T_) + (T_) * 48)
#define SM_ML_FOR(T_) (SM_ALPHA_FOR(T_) + 64)
#define SM_IDX_FOR(T_) (SM_ML_FOR(T_) + 128)
#define SM_NORM_FOR(T_) (SM_IDX_FOR(T_) + WIDTH * 4)
#define SM_BYTES_FOR(T_) (SM_NORM_FOR(T_) + 64)

__device__ __forceinline__ float sink_log2(float x) {
  const float y = x * LOG2E;
  return isinf(y) && isfinite(x) ? copysignf(FLT_MAX, x) : y;
}

__device__ __forceinline__ float2 sum_qk4(const float* p, int stride) {
  const float2 a = *(const float2*)p;
  const float2 b = *(const float2*)(p + stride);
  const float2 c = *(const float2*)(p + 2 * stride);
  const float2 d = *(const float2*)(p + 3 * stride);
  return make_float2((a.x + b.x) + (c.x + d.x), (a.y + b.y) + (c.y + d.y));
}

__device__ __forceinline__ uint32_t sa(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ uint2 dsm_ld2_c5(const void* local, unsigned rank) {
  uint32_t addr;
  uint2 value;
  asm volatile(
      "mapa.shared::cluster.u32 %0, %3, %4;\n\t"
      "ld.shared::cluster.v2.b32 {%1,%2}, [%0];\n"
      : "=&r"(addr), "=r"(value.x), "=r"(value.y)
      : "r"(sa(local)), "r"(rank)
      : "memory");
  return value;
}
__device__ __forceinline__ int swz(int row, int chunk) {
  return (chunk & ~7) | ((chunk & 7) ^ (row & 7));
}
#ifdef ENABLE_G4
// TMA gather4 packs four independently indexed 128-byte row segments next to
// one another. Eight such segments form the 32x512 tile consumed by ldmatrix.
__device__ __forceinline__ int g4off(int row, int chunk) {
  const int rg = row >> 2;
  const int ri = row & 3;
  const int seg = chunk >> 3;
  const int ci = chunk & 7;
  return rg * 4096 + seg * 512 + ri * 128 + (ci ^ ri ^ ((seg & 1) << 2)) * 16;
}
__device__ __forceinline__ void g4_bar_init(uint64_t* bar) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 8;\n" ::"r"(sa(bar)) : "memory");
}
__device__ __forceinline__ void g4_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 4096;\n" ::"r"(sa(bar))
               : "memory");
}
__device__ __forceinline__ void g4_wait(uint64_t* bar, int phase) {
  if ((threadIdx.x & 31) == 0) {
    uint32_t done;
    do {
      asm volatile(
          "{ .reg .pred p; "
          "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2; "
          "selp.u32 %0, 1, 0, p; }\n"
          : "=r"(done)
          : "r"(sa(bar)), "r"(phase)
          : "memory");
    } while (!done);
  }
  __syncwarp();
}
__device__ __forceinline__ void g4_issue(const CUtensorMap* tm, char* dst, int col, int r0, int r1,
                                         int r2, int r3, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.tile::gather4.shared::cta.global."
      "mbarrier::complete_tx::bytes.cta_group::1 "
      "[%0], [%1, {%2,%3,%4,%5,%6}], [%7];\n" ::"r"(sa(dst)),
      "l"(tm), "r"(col), "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(sa(bar))
      : "memory");
}
#endif
__device__ __forceinline__ int score_off(int row, int hp) { return row * 18 + 2 * hp; }
__device__ __forceinline__ void ldm4(uint32_t a, uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                     uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(a));
}
__device__ __forceinline__ void ldm4t(uint32_t a, uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                      uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(a));
}
__device__ __forceinline__ void ldm2(uint32_t a, uint32_t& r0, uint32_t& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r0), "=r"(r1)
               : "r"(a));
}
__device__ __forceinline__ void ldm2t(uint32_t a, uint32_t& r0, uint32_t& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r0), "=r"(r1)
               : "r"(a));
}
__device__ __forceinline__ void stm2t(uint32_t a, uint32_t r0, uint32_t r1) {
  asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1,%2};\n" ::"r"(a), "r"(r0),
               "r"(r1)
               : "memory");
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint64_t tc_enc(uint32_t x) { return (uint64_t(x & 0x3ffffu) >> 4); }
__device__ __forceinline__ uint64_t tc_desc128(const void* p, uint32_t lbo, uint32_t sbo) {
  return tc_enc(sa(p)) | (tc_enc(lbo) << 16) | (tc_enc(sbo) << 32) | (1ull << 46) | (2ull << 61);
}
__device__ __forceinline__ void tc_alloc64(uint32_t* slot) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 64;\n" ::"r"(sa(slot))
               : "memory");
}
__device__ __forceinline__ void tc_dealloc64(uint32_t taddr) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 64;\n" ::"r"(taddr) : "memory");
}
__device__ __forceinline__ void tc_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void tc_mma(uint32_t taddr, uint64_t ad, uint64_t bd, uint32_t idesc,
                                       bool accumulate) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 "
      "[%0], %1, %2, %3, p;\n\t"
      "}\n" ::"r"(taddr),
      "l"(ad), "l"(bd), "r"(idesc), "r"(uint32_t(accumulate))
      : "memory");
}
__device__ __forceinline__ void tc_commit(uint64_t* bar) {
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one."
      "shared::cluster.b64 [%0];\n" ::"r"(sa(bar))
      : "memory");
}
__device__ __forceinline__ void tc_fence_after_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
}
__device__ __forceinline__ void tc_fence_before_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
}
__device__ __forceinline__ void tc_smem_async_fence() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}
__device__ __forceinline__ void tc_ld_score(uint32_t taddr, uint32_t (&x)[8]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.16x256b.x2.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
      : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3]), "=r"(x[4]), "=r"(x[5]), "=r"(x[6]),
        "=r"(x[7])
      : "r"(taddr)
      : "memory");
  asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void tc_ld_pv(uint32_t taddr, uint32_t (&x)[8]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
      : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3]), "=r"(x[4]), "=r"(x[5]), "=r"(x[6]),
        "=r"(x[7])
      : "r"(taddr)
      : "memory");
  asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ int tc_swz_hk(int h, int k) {
  return h * 64 + (((k >> 3) ^ (h & 7)) * 8) + (k & 7);
}
// Packed FP32 multiply (FMUL2).  Used where a thread holds many output
// accumulators, so the flash-attention rescale halves its instruction count.
__device__ __forceinline__ void mul_f32x2(float& x, float& y, float a, float b) {
  asm("{\n\t.reg .b64 ds, ss, ms;\n\t"
      "mov.b64 ss, {%0, %1};\n\t"
      "mov.b64 ms, {%2, %3};\n\t"
      "mul.f32x2 ds, ss, ms;\n\t"
      "mov.b64 {%0, %1}, ds;\n\t}\n"
      : "+f"(x), "+f"(y)
      : "f"(a), "f"(b));
}
__device__ __forceinline__ float warp_max_f32(float x) {
  float y;
  asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;\n" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ uint32_t add_bf16x2(uint32_t a, uint32_t b) {
  uint32_t y;
  asm volatile("add.rn.bf16x2 %0, %1, %2;\n" : "=r"(y) : "r"(a), "r"(b));
  return y;
}
__device__ __forceinline__ uint32_t fma_bf16x2(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t y;
  asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(y) : "r"(a), "r"(b), "r"(c));
  return y;
}

// FP32 merge with a single final BF16 conversion.
struct MergePair {
  float x, y;
  __device__ __forceinline__ MergePair(int = 0) : x(0.f), y(0.f) {}
  __device__ __forceinline__ operator uint32_t() const {
    const __nv_bfloat162 b = __floats2bfloat162_rn(x, y);
    return *reinterpret_cast<const uint32_t*>(&b);
  }
};
__device__ __forceinline__ MergePair merge_f32x2(uint32_t a, uint32_t wbits, MergePair c) {
  const float x = __uint_as_float(a << 16);
  const float y = __uint_as_float(a & 0xffff0000u);
  const float w = __uint_as_float(wbits);
  asm("{ .reg .b64 av, bv, cv, dv;\n\t"
      "mov.b64 av, {%2, %3};\n\t"
      "mov.b64 bv, {%4, %4};\n\t"
      "mov.b64 cv, {%0, %1};\n\t"
      "fma.rn.f32x2 dv, av, bv, cv;\n\t"
      "mov.b64 {%0, %1}, dv; }\n"
      : "+f"(c.x), "+f"(c.y)
      : "f"(x), "f"(y), "f"(w));
  return c;
}
__device__ __forceinline__ uint32_t ex2_bf16x2(float x0, float x1) {
  uint32_t y;
  asm volatile(
      "{\n\t"
      ".reg .b32 x;\n\t"
      "cvt.rn.bf16x2.f32 x, %2, %1;\n\t"
      "ex2.approx.ftz.bf16x2 %0, x;\n\t"
      "}\n"
      : "=r"(y)
      : "f"(x0), "f"(x1));
  return y;
}
template <int WIDTH_>
__device__ __forceinline__ uint32_t warp_add_u32(uint32_t x) {
  if constexpr (WIDTH_ == 32) {
    uint32_t y;
    asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;\n" : "=r"(y) : "r"(x));
    return y;
  } else {
#pragma unroll
    for (int off = WIDTH_ / 2; off > 0; off >>= 1) x += __shfl_xor_sync(0xffffffff, x, off, WIDTH_);
    return x;
  }
}

__device__ __forceinline__ void ws_mbar_init(uint64_t* bar) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(sa(bar)));
}
__device__ __forceinline__ void ws_mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n" ::"r"(sa(bar)) : "memory");
}
__device__ __forceinline__ void ws_mbar_wait(uint64_t* bar, int phase) {
  uint32_t done;
  do {
    asm volatile(
        "{ .reg .pred p; "
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2; "
        "selp.u32 %0, 1, 0, p; }\n"
        : "=r"(done)
        : "r"(sa(bar)), "r"(phase)
        : "memory");
  } while (!done);
}
__device__ __forceinline__ void ws_qk_sync() { asm volatile("bar.sync 1, 128;\n" ::: "memory"); }
__device__ __forceinline__ void ws_pv_sync() { asm volatile("bar.sync 2, 128;\n" ::: "memory"); }

// Four QK/softmax producer warps overlap a two-slot queue with four PV
// consumer warps.  This path is specialized for TILE32 / cluster size 4.
#define WS_P SM_P_FOR(32)
#define WS_ALPHA (WS_P + 2 * 32 * 48)
#define WS_ML (WS_ALPHA + 2 * HEADS * 4)
#define WS_IDX (WS_ML + HEADS * 8)
#define WS_BYTES (WS_IDX + 5 * 32 * 4)

__global__ __launch_bounds__(256, 2) void mla_ws_c4(
    const uint8_t* __restrict__ qp, const uint8_t* __restrict__ swap_,
    const uint8_t* __restrict__ ckvp, const int* __restrict__ sidx, const int* __restrict__ slen,
    const float* __restrict__ sinks, __nv_bfloat16* __restrict__ out) {
  extern __shared__ __align__(16) char smem[];
  __shared__ __align__(8) uint64_t handoff[4];
  char* Qs = smem + SM_Q;
  char* KVs = smem + SM_KV;
  __nv_bfloat16* Sb = (__nv_bfloat16*)(smem + SM_S_FOR(32));
  char* Pbase = smem + WS_P;
  float* Abase = (float*)(smem + WS_ALPHA);
  float2* MLsh = (float2*)(smem + WS_ML);
  int* Idx = (int*)(smem + WS_IDX);

  const int t = blockIdx.y;
  const int sp = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int tile0 = sp * 5;
  const int len = slen[t];
  const int* myidx = sidx + (size_t)t * WIDTH;

  if (tid == 0) {
    ws_mbar_init(&handoff[0]);
    ws_mbar_init(&handoff[1]);
    ws_mbar_init(&handoff[2]);
    ws_mbar_init(&handoff[3]);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    ws_mbar_arrive(&handoff[2]);
    ws_mbar_arrive(&handoff[3]);
  }

  // Stage Q once; cache the five rank-local index tiles.
  {
    const uint8_t* src = qp + (size_t)t * HEADS * DIM * 2;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int ci = tid + r * 256;
      const int row = ci >> 6;
      const int ch = ci & 63;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(
                       sa(Qs + row * 1024 + swz(row, ch) * 16)),
                   "l"(src + (size_t)ci * 16));
    }
    asm volatile("cp.async.commit_group;\n");
  }
  for (int i = tid; i < 5 * 32; i += 256) {
    const int gcol = tile0 * 32 + i;
    const int v = myidx[gcol];
    Idx[i] = (gcol < len && v >= 0) ? v : -1;
  }
  __syncthreads();
  asm volatile("cp.async.wait_group 0;\n");
  __syncthreads();

  if (warp < 4) {
    const int qw = warp;
    const int qrow = (lane & 7) | ((lane & 16) >> 1);
    const int qjo = (lane >> 3) & 1;
    uint32_t qf[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int j0 = (qw * 128 + i * 16) >> 3;
      ldm4(sa(Qs + qrow * 1024 + swz(qrow, j0 + qjo) * 16), qf[i][0], qf[i][1], qf[i][2], qf[i][3]);
    }
    float mr[2][2] = {{-INFINITY, -INFINITY}, {-INFINITY, -INFINITY}};
    float lr[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
#pragma unroll
    for (int tl = 0; tl < 5; ++tl) {
      const int slot = tl & 1;
      if (tid == 0) ws_mbar_wait(&handoff[2 + slot], (tl >> 1) & 1);
      ws_qk_sync();

      // Four producer warps cooperatively gather one 32x512 KV tile.
      const int base = (tile0 + tl) * 32;
      const uint8_t* pool = base < 128 ? swap_ : ckvp;
      char* kvb = KVs + slot * 32 * 1024;
      const int ldrow = tid >> 6;
      const int ldch = tid & 63;
#pragma unroll
      for (int r = 0; r < 16; ++r) {
        const int row = ldrow + r * 2;
        const int off = Idx[tl * 32 + row];
        const int cpsz = off >= 0 ? 16 : 0;  // zero-fill invalid rows
        const uint8_t* src = pool + (size_t)(off >= 0 ? off : 0) * 1024 + ldch * 16;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(
                         sa(kvb + row * 1024 + swz(row, ldch) * 16)),
                     "l"(src), "r"(cpsz));
      }
      asm volatile(
          "cp.async.commit_group;\n"
          "cp.async.wait_group 0;\n" ::
              : "memory");
      ws_qk_sync();

      // Each producer warp accumulates one D128 slice for both key halves.
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int col0 = half * 16;
        const int arow = col0 + (lane & 15);
        const int ajo = lane >> 4;
        float s0[4] = {0.f, 0.f, 0.f, 0.f};
        float s1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int j0 = (qw * 128 + i * 16) >> 3;
          uint32_t a[4];
          ldm4(sa(kvb + arow * 1024 + swz(arow, j0 + ajo) * 16), a[0], a[1], a[2], a[3]);
          mma_bf16(s0, a, qf[i][0], qf[i][1]);
          mma_bf16(s1, a, qf[i][2], qf[i][3]);
        }
        __nv_bfloat16* sb = Sb + qw * SCORE_QSTRIDE;
        const int srow = col0 + (lane >> 2);
        const int hp = lane & 3;
        const int sx = (srow ^ (srow >> 2)) & 7;
        const int so0 = srow * HEADS + 2 * (hp ^ sx);
        const int so1 = (srow + 8) * HEADS + 2 * (hp ^ sx ^ 2);
        const __nv_bfloat162 a0 = __floats2bfloat162_rn(s0[0], s0[1]);
        const __nv_bfloat162 a1 = __floats2bfloat162_rn(s0[2], s0[3]);
        const __nv_bfloat162 b0 = __floats2bfloat162_rn(s1[0], s1[1]);
        const __nv_bfloat162 b1 = __floats2bfloat162_rn(s1[2], s1[3]);
        *(uint32_t*)(sb + so0) = *(const uint32_t*)&a0;
        *(uint32_t*)(sb + so1) = *(const uint32_t*)&a1;
        *(uint32_t*)(sb + (so0 ^ 8)) = *(const uint32_t*)&b0;
        *(uint32_t*)(sb + (so1 ^ 8)) = *(const uint32_t*)&b1;
      }
      ws_qk_sync();

      char* pb = Pbase + slot * 32 * 48;
      float* ab = Abase + slot * HEADS;
#pragma unroll
      for (int pair = 0; pair < 2; ++pair) {
        const int h0 = 4 * qw + 2 * pair;
        const int soff = score_off(lane, h0 >> 1);
        const uint32_t sp0 = *(const uint32_t*)(Sb + soff);
        const uint32_t sp1 = *(const uint32_t*)(Sb + SCORE_QSTRIDE + soff);
        const uint32_t sp2 = *(const uint32_t*)(Sb + 2 * SCORE_QSTRIDE + soff);
        const uint32_t sp3 = *(const uint32_t*)(Sb + 3 * SCORE_QSTRIDE + soff);
        const uint32_t score_pair = add_bf16x2(add_bf16x2(sp0, sp1), add_bf16x2(sp2, sp3));
        __nv_bfloat162 sbf;
        *reinterpret_cast<uint32_t*>(&sbf) = score_pair;
        const float2 acc = __bfloat1622float2(sbf);
        const bool ok = Idx[tl * 32 + lane] >= 0;
        const float x0 = ok ? acc.x * QKSCALE : -INFINITY;
        const float x1 = ok ? acc.y * QKSCALE : -INFINITY;
        const float mt0 = warp_max_f32(x0);
        const float mt1 = warp_max_f32(x1);
        const float mn0 = fmaxf(mr[pair][0], mt0);
        const float mn1 = fmaxf(mr[pair][1], mt1);
        const float g0 = mn0 == -INFINITY ? 0.f : mn0;
        const float g1 = mn1 == -INFINITY ? 0.f : mn1;
        const float al0 = exp2f(mr[pair][0] - g0);
        const float al1 = exp2f(mr[pair][1] - g1);
        const float p0 = exp2f(x0 - g0);
        const float p1 = exp2f(x1 - g1);
        const __nv_bfloat162 pk = __floats2bfloat162_rn(p0, p1);
        *(uint32_t*)(pb + lane * 48 + h0 * 2) = *(const uint32_t*)&pk;
        constexpr float kProbScale = 67108864.f;
        constexpr float kInvProbScale = 1.f / kProbScale;
        const uint32_t ps0 = warp_add_u32<32>(__float2uint_rn(p0 * kProbScale));
        const uint32_t ps1 = warp_add_u32<32>(__float2uint_rn(p1 * kProbScale));
        lr[pair][0] = lr[pair][0] * al0 + float(ps0) * kInvProbScale;
        lr[pair][1] = lr[pair][1] * al1 + float(ps1) * kInvProbScale;
        mr[pair][0] = mn0;
        mr[pair][1] = mn1;
        if (lane == 0) {
          ab[h0] = al0;
          ab[h0 + 1] = al1;
        }
      }
      ws_qk_sync();
      if (tid == 0) ws_mbar_arrive(&handoff[slot]);
    }

    if (lane == 0) {
#pragma unroll
      for (int pair = 0; pair < 2; ++pair) {
        const int h0 = 4 * qw + 2 * pair;
        MLsh[h0] = make_float2(mr[pair][0], lr[pair][0]);
        MLsh[h0 + 1] = make_float2(mr[pair][1], lr[pair][1]);
      }
    }
  } else {
    const int pw = warp - 4;
    float o[8][2][4];
#pragma unroll
    for (int mi = 0; mi < 8; ++mi)
#pragma unroll
      for (int n = 0; n < 2; ++n)
#pragma unroll
        for (int r = 0; r < 4; ++r) o[mi][n][r] = 0.f;

    const int vrow_b = (lane & 7) + ((lane & 16) >> 1);
    const int vjo = (lane >> 3) & 1;
    const int shd = 2 * (lane & 3);
#pragma unroll
    for (int tl = 0; tl < 5; ++tl) {
      const int slot = tl & 1;
      if (tid == 128) ws_mbar_wait(&handoff[slot], (tl >> 1) & 1);
      ws_pv_sync();

      char* kvb = KVs + slot * 32 * 1024;
      char* pb = Pbase + slot * 32 * 48;
      float* ab = Abase + slot * HEADS;
      float av[2][2];
      const float2 av0 = *reinterpret_cast<const float2*>(ab + shd);
      const float2 av1 = *reinterpret_cast<const float2*>(ab + 8 + shd);
      av[0][0] = av0.x;
      av[0][1] = av0.y;
      av[1][0] = av1.x;
      av[1][1] = av1.y;
#pragma unroll
      for (int mi = 0; mi < 8; ++mi)
#pragma unroll
        for (int n = 0; n < 2; ++n)
#pragma unroll
          for (int r = 0; r < 4; ++r) o[mi][n][r] *= av[n][r & 1];

#pragma unroll
      for (int ki = 0; ki < 2; ++ki) {
        uint32_t b[4];
        ldm4t(sa(pb + (ki * 16 + (lane & 15)) * 48 + (lane >> 4) * 16), b[0], b[1], b[2], b[3]);
        const int vrow = ki * 16 + vrow_b;
#pragma unroll
        for (int mi = 0; mi < 8; mi += 2) {
          const int j0 = (pw * 128 + mi * 16) >> 3;
          const int j1 = (pw * 128 + (mi + 1) * 16) >> 3;
          uint32_t a0[4], a1[4];
          ldm4t(sa(kvb + vrow * 1024 + swz(vrow, j0 + vjo) * 16), a0[0], a0[1], a0[2], a0[3]);
          ldm4t(sa(kvb + vrow * 1024 + swz(vrow, j1 + vjo) * 16), a1[0], a1[1], a1[2], a1[3]);
          mma_bf16(o[mi][0], a0, b[0], b[1]);
          mma_bf16(o[mi][1], a0, b[2], b[3]);
          mma_bf16(o[mi + 1][0], a1, b[0], b[1]);
          mma_bf16(o[mi + 1][1], a1, b[2], b[3]);
        }
      }
      ws_pv_sync();
      if (tid == 128) ws_mbar_arrive(&handoff[2 + slot]);
    }

    __nv_bfloat16* Osh = (__nv_bfloat16*)(smem + SM_Q);
    constexpr int kDsmXor = 8;
#pragma unroll
    for (int mi = 0; mi < 8; ++mi) {
      const int dbase = pw * 128 + mi * 16;
#pragma unroll
      for (int n = 0; n < 2; ++n) {
        const int h = 8 * n + (lane & 7);
        const int d8 = dbase + (((lane >> 3) & 1) * 8);
        const int dx = kDsmXor * (h & 7);
        const __nv_bfloat162 r0 = __floats2bfloat162_rn(o[mi][n][0], o[mi][n][1]);
        const __nv_bfloat162 r1 = __floats2bfloat162_rn(o[mi][n][2], o[mi][n][3]);
        stm2t(sa(Osh + h * DIM + (d8 ^ dx)), *reinterpret_cast<const uint32_t*>(&r0),
              *reinterpret_cast<const uint32_t*>(&r1));
      }
    }
  }
  __syncthreads();

  // Fixed cluster-4 DSM reduction; each rank owns four heads and two output
  // chunks per thread.
  cg::cluster_group clu = cg::this_cluster();
  const unsigned rank = clu.block_rank();
  __nv_bfloat16* Osh = (__nv_bfloat16*)(smem + SM_Q);
  constexpr int kDsmXor = 8;
  clu.sync();
  uint32_t* W2sh = (uint32_t*)(smem + SM_S_FOR(32));
  if (tid < HEADS && rank == (unsigned)((tid >> 1) & 3)) {
    const int h = tid;
    const float sk = sinks[h] * LOG2E;
    float2 mlv[4];
    float M = sk;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      mlv[p] = ((float2*)clu.map_shared_rank(MLsh, p))[h];
      M = fmaxf(M, mlv[p].x);
    }
    float den = exp2f(sk - M);
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      mlv[p].x = exp2f(mlv[p].x - M);
      den += mlv[p].y * mlv[p].x;
    }
    const float inv = 1.f / den;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const __nv_bfloat162 wb = __floats2bfloat162_rn(mlv[p].x * inv, mlv[p].x * inv);
      W2sh[p * HEADS + h] = *reinterpret_cast<const uint32_t*>(&wb);
    }
  }

  const int c0 = (int)rank * 256 + tid;
  const int c1 = c0 + 1024;
  const int h0 = c0 >> 7;
  const int h1 = c1 >> 7;
  const int d40 = (c0 & 127) * 4;
  const int d41 = (c1 & 127) * 4;
  const int sd40 = d40 ^ (kDsmXor * (h0 & 7));
  const int sd41 = d41 ^ (kDsmXor * (h1 & 7));
  uint2 v0[4], v1[4];
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    v0[p] = *(const uint2*)clu.map_shared_rank(Osh + h0 * DIM + sd40, p);
    v1[p] = *(const uint2*)clu.map_shared_rank(Osh + h1 * DIM + sd41, p);
  }
  __syncthreads();

  uint32_t a00 = 0, a01 = 0, a10 = 0, a11 = 0;
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    const uint32_t w0 = W2sh[p * HEADS + h0];
    const uint32_t w1 = W2sh[p * HEADS + h1];
    a00 = fma_bf16x2(v0[p].x, w0, a00);
    a01 = fma_bf16x2(v0[p].y, w0, a01);
    a10 = fma_bf16x2(v1[p].x, w1, a10);
    a11 = fma_bf16x2(v1[p].y, w1, a11);
  }
  *(uint2*)(out + ((size_t)t * HEADS + h0) * DIM + d40) = make_uint2(a00, a01);
  *(uint2*)(out + ((size_t)t * HEADS + h1) * DIM + d41) = make_uint2(a10, a11);
  asm volatile(
      "barrier.cluster.arrive.relaxed.aligned;\n\t"
      "barrier.cluster.wait.aligned;\n" ::
          : "memory");
}

template <int TILE, int CSZ, bool GROUPED = false, bool OWNER = false, bool XOR_SCORE = true,
          bool BAL7 = false>
__global__ __launch_bounds__((TILE == 32 ? 256 : 128), (TILE == 32 ? 2 : 3)) void mla_main(
    const uint8_t* __restrict__ qp, const uint8_t* __restrict__ swap_,
    const uint8_t* __restrict__ ckvp, const int* __restrict__ sidx, const int* __restrict__ slen,
    const float* __restrict__ sinks, __nv_bfloat16* __restrict__ out,
    int tiles_per_split G4_PARAM) {
#ifdef ENABLE_G4
  extern __shared__ __align__(1024) char g4_smem[];
  char* smem = g4_smem;
  __shared__ __align__(8) uint64_t g4bar[2];
#else
  extern __shared__ __align__(16) char smem[];
#endif
  char* Qs = smem + SM_Q;
  char* KVs = smem + SM_KV;
  float* Sb = (float*)(smem + SM_S_FOR(TILE));
  char* Pb = smem + SM_P_FOR(TILE);
  float* Alp = (float*)(smem + SM_ALPHA_FOR(TILE));
  int* Idx = (int*)(smem + SM_IDX_FOR(TILE));
  float* Norm = (float*)(smem + SM_NORM_FOR(TILE));

  const int t = blockIdx.y;
  const int sp = blockIdx.x;
  constexpr int TOTAL_TILES = WIDTH / TILE;
  constexpr int TILES_PER_SPLIT = (TOTAL_TILES + CSZ - 1) / CSZ;
  const int tile0 = sp * TILES_PER_SPLIT;
  int ntile = TILES_PER_SPLIT;
  if constexpr (TOTAL_TILES % CSZ != 0) {
    ntile = TOTAL_TILES - tile0;
    if (ntile > TILES_PER_SPLIT) ntile = TILES_PER_SPLIT;
  }

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  constexpr int NTHREAD = TILE == 32 ? 256 : 128;
  constexpr int NWARP = NTHREAD / 32;
  constexpr int MIO = 512 / (NWARP * 16);
#ifdef ENABLE_G4
  constexpr bool kUseG4 = TILE == 32 && CSZ == 4 && !GROUPED && !OWNER;
#else
  constexpr bool kUseG4 = false;
#endif
  constexpr bool kC4DirectTile0 =
      TILE == 32 && (CSZ == 4 || CSZ == 5 || (CSZ == 7 && BAL7)) && !GROUPED && !OWNER && !kUseG4;
  constexpr bool kC4OverlapQReload =
      TILE == 32 && (CSZ == 4 || CSZ == 5 || (CSZ == 7 && BAL7)) && !GROUPED && !OWNER && !kUseG4;
  const int len = slen[t];
  const int* myidx = sidx + (size_t)t * WIDTH;

  auto issue_tile = [&](int tl) {
    const int base = (tile0 + tl) * TILE;
    char* dst = KVs + (tl & 1) * (TILE * 1024);
    const uint8_t* pool = (base < 128) ? swap_ : ckvp;
    if constexpr (kUseG4) {
#ifdef ENABLE_G4
      if (lane == 0) {
        const int row0 = 4 * warp;
        int v0 = Idx[tl * TILE + row0];
        int v1 = Idx[tl * TILE + row0 + 1];
        int v2 = Idx[tl * TILE + row0 + 2];
        int v3 = Idx[tl * TILE + row0 + 3];
        v0 = v0 < 0 ? 0 : v0;
        v1 = v1 < 0 ? 0 : v1;
        v2 = v2 < 0 ? 0 : v2;
        v3 = v3 < 0 ? 0 : v3;
        g4_arrive(&g4bar[tl & 1]);
        const CUtensorMap* tm = base < 128 ? &swa_tm : &ckv_tm;
#pragma unroll
        for (int seg = 0; seg < 8; ++seg)
          g4_issue(tm, dst + warp * 4096 + seg * 512, seg * 64, v0, v1, v2, v3, &g4bar[tl & 1]);
      }
#endif
    } else if constexpr (TILE == 32 && (CSZ == 4 || CSZ == 5 || CSZ == 7)) {
      constexpr int ROWS_PER_WARP = TILE / NWARP;
      int iv[ROWS_PER_WARP];
#pragma unroll
      for (int r = 0; r < ROWS_PER_WARP; ++r) iv[r] = Idx[tl * TILE + warp + r * NWARP];
#pragma unroll
      for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const int row = warp + r * NWARP;
        const int off = iv[r];
        // Invalid rows (negative index, or tail columns masked at staging)
        // are ZERO-FILLED via the cp.async src-size operand: no global read
        // at all, so garbage/NaN pool contents can never enter the PV MMA.
        const int cpsz = off >= 0 ? 16 : 0;
        const uint8_t* row_src = pool + (size_t)(off >= 0 ? off : 0) * 1024;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int ch = lane + half * 32;
          const uint8_t* s = row_src + ch * 16;
          uint32_t d = sa(dst + row * 1024 + swz(row, ch) * 16);
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(s),
                       "r"(cpsz));
        }
      }
    } else {
      const int ldrow = tid >> 6;
      const int ldch = tid & 63;
      int iv[8];
#pragma unroll
      for (int r = 0; r < 8; ++r) iv[r] = Idx[tl * TILE + ldrow + r * (NTHREAD / 64)];
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        const int row = ldrow + r * (NTHREAD / 64);
        const int off = iv[r];
        const int cpsz = off >= 0 ? 16 : 0;  // zero-fill invalid rows
        const uint8_t* s = pool + (size_t)(off >= 0 ? off : 0) * 1024 + ldch * 16;
        uint32_t d = sa(dst + row * 1024 + swz(row, ldch) * 16);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(s), "r"(cpsz));
      }
    }
  };

  if constexpr (kUseG4) {
#ifdef ENABLE_G4
    if (tid == 0) {
      g4_bar_init(&g4bar[0]);
      g4_bar_init(&g4bar[1]);
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();
#endif
  }

  // Q -> smem
  {
    const uint8_t* src = qp + (size_t)t * (HEADS * DIM * 2);
#pragma unroll
    for (int r = 0; r < 1024 / NTHREAD; ++r) {
      int ci = tid + r * NTHREAD;
      int row = ci >> 6, ch = ci & 63;
      uint32_t d = sa(Qs + row * 1024 + swz(row, ch) * 16);
      const uint8_t* s = src + (size_t)ci * 16;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(s));
    }
    asm volatile("cp.async.commit_group;\n");
  }
  {
    const int ncols = ntile * TILE;
    constexpr int FIRST_STAGED_COL = kC4DirectTile0 ? TILE : 0;
    for (int i = FIRST_STAGED_COL + tid; i < ncols; i += NTHREAD) {
      const int gcol = tile0 * TILE + i;
      // Mask tail columns past the per-token length to -1 BEFORE staging:
      // inactive entries carry no offset-validity contract, so they must
      // never reach a gather (the softmax length mask still applies).
      const int v = (gcol < len) ? myidx[gcol] : -1;
      Idx[i] = v;
    }
  }
  if constexpr (kC4DirectTile0) {
    // Lane 0 of each warp broadcasts its four tile-0 offsets while the other
    // lanes retire the remaining index-table stores. This avoids staging and
    // redundantly loading the first 32 offsets before the initial gather.
    constexpr int ROWS_PER_WARP = TILE / NWARP;
    const int base = tile0 * TILE;
    const uint8_t* pool = (base < 128) ? swap_ : ckvp;
    int iv[ROWS_PER_WARP];
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
      const int row = warp + r * NWARP;
      const int gcol = base + row;
      int valid_v = -1;
      if (lane == 0) {
        // length-masked at staging (see the staged-Idx comment above)
        const int v = (gcol < len) ? myidx[gcol] : -1;
        valid_v = v;
        Idx[row] = valid_v;
      }
      valid_v = __shfl_sync(0xffffffff, valid_v, 0);
      iv[r] = valid_v;  // raw: negative means zero-fill at the gather
    }
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
      const int row = warp + r * NWARP;
      const int off = iv[r];
      const int cpsz = off >= 0 ? 16 : 0;  // zero-fill invalid rows
      const uint8_t* row_src = pool + (size_t)(off >= 0 ? off : 0) * 1024;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int ch = lane + half * 32;
        const uint8_t* s = row_src + ch * 16;
        const uint32_t d = sa(KVs + row * 1024 + swz(row, ch) * 16);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(s), "r"(cpsz));
      }
    }
  } else {
    __syncthreads();
    issue_tile(0);
  }
  if constexpr (kUseG4) {
#ifdef ENABLE_G4
    g4_wait(&g4bar[0], 0);
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#endif
  } else {
    asm volatile(
        "cp.async.commit_group;\n"
        "cp.async.wait_group 1;\n" ::
            : "memory");
  }
  __syncthreads();

  const int kk = TILE == 32 ? warp >> 1 : warp;
  const int col0 = TILE == 32 ? (warp & 1) * 16 : 0;
  const int qrow = (lane & 7) | ((lane & 16) >> 1);
  const int qjo = (lane >> 3) & 1;
  const uint32_t q0_addr = sa(Qs) + (uint32_t)(qrow * 1024 + kk * 256 + ((qjo ^ (qrow & 7)) * 16));
  uint32_t qf[8][4];
  {
    if constexpr (TILE == 32 && CSZ == 4 && !GROUPED && !OWNER) {
      const uint32_t qbase = sa(Qs) + (uint32_t)(qrow * 1024 + kk * 256);
      const int qx = qrow & 7;
      uint32_t qoff[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) qoff[j] = (uint32_t)((((2 * j + qjo) & 7) ^ qx) * 16);
#pragma unroll
      for (int i = 0; i < 8; ++i)
        ldm4(qbase + qoff[i & 3] + (uint32_t)((i >> 2) * 128), qf[i][0], qf[i][1], qf[i][2],
             qf[i][3]);
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        int j0 = (kk * 128 + i * 16) >> 3;
        ldm4(sa(Qs + qrow * 1024 + swz(qrow, j0 + qjo) * 16), qf[i][0], qf[i][1], qf[i][2],
             qf[i][3]);
      }
    }
  }
  constexpr bool kAnchor = false;
  float mrun0 = kAnchor ? 0.f : -INFINITY, mrun1 = kAnchor ? 0.f : -INFINITY, lrun0 = 0.f,
        lrun1 = 0.f;
  // The live path uses a running maximum and floating-point running sums.
  // Retain the compile-time-disabled anchor branch's accumulator declarations.
  uint32_t lfix0 = 0, lfix1 = 0;
  float o[MIO][2][4];
#pragma unroll
  for (int i = 0; i < MIO; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) o[i][j][k] = 0.f;

  const int soft_lane = TILE == 32 ? lane : lane & 15;
  const int h0 = TILE == 32 ? 2 * warp : 4 * warp + 2 * (lane >> 4);
  const int arow = col0 + (lane & 15);  // QK A-operand row
  const int ajo = lane >> 4;
  const int vrow_b = (lane & 7) + ((lane & 16) >> 1);  // PV A-operand row base
  const int vjo = (lane >> 3) & 1;
  const int srow = col0 + (lane >> 2);
  const int shd = 2 * (lane & 3);

  // Swizzled smem offsets: swz(row, chunk) splits into a compile-time part and
  // a lane-dependent XOR, so each ldmatrix address becomes base + const + e/g.
  const uint32_t kv_s0 = sa(KVs);
  const int ax = arow & 7;
  const int vx = vrow_b & 7;
  uint32_t qk_base = 0, pv_base = 0;
  uint32_t eoff[4] = {0, 0, 0, 0}, goff[4] = {0, 0, 0, 0};
  if constexpr (TILE == 32) {
    qk_base = kv_s0 + (uint32_t)(arow * 1024 + kk * 256);
    pv_base = kv_s0 + (uint32_t)(vrow_b * 1024 + warp * 128);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      eoff[j] = (uint32_t)((((2 * j + ajo) & 7) ^ ax) * 16);
      goff[j] = (uint32_t)((((2 * j + vjo) & 7) ^ vx) * 16);
    }
  }
  constexpr bool kPvPrefetch = TILE == 32 && !GROUPED && (CSZ == 4 || CSZ == 5);

  for (int tl = 0; tl < ntile; ++tl) {
    if (tl + 1 < ntile) {
      issue_tile(tl + 1);
      if constexpr (!kUseG4) {
        asm volatile("cp.async.commit_group;\n");
        if constexpr (kC4OverlapQReload) {
          if (tl > 0) {
            ldm4(q0_addr, qf[0][0], qf[0][1], qf[0][2], qf[0][3]);
          }
        }
        asm volatile("cp.async.wait_group 1;\n");
      }
    } else {
      if constexpr (!kUseG4) {
        if constexpr (kC4OverlapQReload) {
          if (tl > 0) {
            ldm4(q0_addr, qf[0][0], qf[0][1], qf[0][2], qf[0][3]);
          }
        }
        asm volatile("cp.async.wait_group 0;\n");
      }
    }
    if constexpr (kUseG4) {
#ifdef ENABLE_G4
      if (tl > 0) g4_wait(&g4bar[tl & 1], (tl >> 1) & 1);
#endif
    }
    __syncthreads();
    char* kvb = KVs + (tl & 1) * (TILE * 1024);

    // ---- QK: S^T[16 cols][16 heads] partial over this warp's 128 d
    float s[2][4];
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) s[j][k] = 0.f;
    if constexpr (GROUPED) {
#pragma unroll
      for (int g = 0; g < 2; ++g) {
        uint32_t a[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          int j0 = (kk * 128 + (g * 4 + i) * 16) >> 3;
          ldm4(sa(kvb + arow * 1024 + swz(arow, j0 + ajo) * 16), a[i][0], a[i][1], a[i][2],
               a[i][3]);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          mma_bf16(s[0], a[i], qf[g * 4 + i][0], qf[g * 4 + i][1]);
          mma_bf16(s[1], a[i], qf[g * 4 + i][2], qf[g * 4 + i][3]);
        }
      }
    } else {
      if constexpr (TILE == 16 && CSZ == 5) {
        uint32_t a[4];
        int j0 = (kk * 128) >> 3;
        ldm4(sa(kvb + arow * 1024 + swz(arow, j0 + ajo) * 16), a[0], a[1], a[2], a[3]);
        mma_bf16(s[0], a, qf[0][0], qf[0][1]);
        mma_bf16(s[1], a, qf[0][2], qf[0][3]);
        j0 = (kk * 128 + 16) >> 3;
        ldm4(sa(kvb + arow * 1024 + swz(arow, j0 + ajo) * 16), qf[0][0], qf[0][1], qf[0][2],
             qf[0][3]);
#pragma unroll
        for (int i = 1; i < 8; ++i) {
          if (i + 1 < 8) {
            j0 = (kk * 128 + (i + 1) * 16) >> 3;
            if (i & 1)
              ldm4(sa(kvb + arow * 1024 + swz(arow, j0 + ajo) * 16), a[0], a[1], a[2], a[3]);
            else
              ldm4(sa(kvb + arow * 1024 + swz(arow, j0 + ajo) * 16), qf[0][0], qf[0][1], qf[0][2],
                   qf[0][3]);
          }
          if (i & 1) {
            mma_bf16(s[0], qf[0], qf[i][0], qf[i][1]);
            mma_bf16(s[1], qf[0], qf[i][2], qf[i][3]);
          } else {
            mma_bf16(s[0], a, qf[i][0], qf[i][1]);
            mma_bf16(s[1], a, qf[i][2], qf[i][3]);
          }
        }
      } else if constexpr (TILE == 32) {
        const uint32_t qb = kUseG4 ? sa(kvb) : qk_base + (uint32_t)((tl & 1) * (TILE * 1024));
        if constexpr ((CSZ == 4 || CSZ == 5 || (CSZ == 7 && BAL7)) && !GROUPED && !OWNER &&
                      !kUseG4) {
          uint32_t a[4];
          ldm4(qb + eoff[0], a[0], a[1], a[2], a[3]);
          mma_bf16(s[0], a, qf[0][0], qf[0][1]);
          mma_bf16(s[1], a, qf[0][2], qf[0][3]);

          // qf[0] is dead after i=0; alternate it with a as a rolling
          // A-fragment buffer.  The next ldmatrix now overlaps the second
          // independent accumulator chain of the current K slice.
          ldm4(qb + eoff[1], qf[0][0], qf[0][1], qf[0][2], qf[0][3]);
#pragma unroll
          for (int i = 1; i < 8; ++i) {
            const uint32_t next = qb + eoff[(i + 1) & 3] + (uint32_t)(((i + 1) >> 2) * 128);
            if (i & 1) {
              mma_bf16(s[0], qf[0], qf[i][0], qf[i][1]);
              if (i + 1 < 8) ldm4(next, a[0], a[1], a[2], a[3]);
              mma_bf16(s[1], qf[0], qf[i][2], qf[i][3]);
            } else {
              mma_bf16(s[0], a, qf[i][0], qf[i][1]);
              if (i + 1 < 8) ldm4(next, qf[0][0], qf[0][1], qf[0][2], qf[0][3]);
              mma_bf16(s[1], a, qf[i][2], qf[i][3]);
            }
          }
        } else {
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            uint32_t a[4];
            if constexpr (kUseG4) {
#ifdef ENABLE_G4
              const int j0 = (kk * 128 + i * 16) >> 3;
              ldm4(qb + (uint32_t)g4off(arow, j0 + ajo), a[0], a[1], a[2], a[3]);
#endif
            } else {
              ldm4(qb + eoff[i & 3] + (uint32_t)((i >> 2) * 128), a[0], a[1], a[2], a[3]);
            }
            mma_bf16(s[0], a, qf[i][0], qf[i][1]);
            mma_bf16(s[1], a, qf[i][2], qf[i][3]);
          }
        }
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          int j0 = (kk * 128 + i * 16) >> 3;
          uint32_t a[4];
          ldm4(sa(kvb + arow * 1024 + swz(arow, j0 + ajo) * 16), a[0], a[1], a[2], a[3]);
          mma_bf16(s[0], a, qf[i][0], qf[i][1]);
          mma_bf16(s[1], a, qf[i][2], qf[i][3]);
        }
      }
    }
    {
      float* sb = Sb + kk * (XOR_SCORE ? SCORE_QSTRIDE : TILE * 18);
      const float2 s00 = make_float2(s[0][0], s[0][1]);
      const float2 s01 = make_float2(s[0][2], s[0][3]);
      const float2 s10 = make_float2(s[1][0], s[1][1]);
      const float2 s11 = make_float2(s[1][2], s[1][3]);
      if constexpr (XOR_SCORE) {
        const int hp = lane & 3;
        const int sx = (srow ^ (srow >> 2)) & 7;
        const int so0 = score_off(srow, hp);
        const int so1 = score_off(srow + 8, hp);
        *(float2*)(sb + so0) = s00;
        *(float2*)(sb + so1) = s01;
        *(float2*)(sb + so0 + 8) = s10;
        *(float2*)(sb + so1 + 8) = s11;
      } else {
        *(float2*)(sb + srow * 18 + shd) = s00;
        *(float2*)(sb + (srow + 8) * 18 + shd) = s01;
        *(float2*)(sb + srow * 18 + 8 + shd) = s10;
        *(float2*)(sb + (srow + 8) * 18 + 8 + shd) = s11;
      }
    }
    __syncthreads();

    // ---- softmax over the 32 cols of this tile, 2 heads per warp
    {
      const int soff = XOR_SCORE ? score_off(soft_lane, h0 >> 1) : soft_lane * 18 + h0;
      const float2 acc = sum_qk4(Sb + soff, XOR_SCORE ? SCORE_QSTRIDE : TILE * 18);
      const bool ok = (tile0 + tl) * TILE + soft_lane < len && Idx[tl * TILE + soft_lane] >= 0;
      float p0, p1, a0 = 1.f, a1 = 1.f;
      if constexpr (kAnchor) {
        // Anchor the exponent offset once, on the first tile of this split.
        // After that the offset is loop invariant, so the softmax scale and
        // offset collapse into a single fma feeding ex2 directly.
        if (tl == 0) {
          const float s0 = ok ? acc.x * QKSCALE : -INFINITY;
          const float s1 = ok ? acc.y * QKSCALE : -INFINITY;
          const float m0 = warp_max_f32(s0), m1 = warp_max_f32(s1);
          mrun0 = (m0 == -INFINITY) ? 0.f : m0;
          mrun1 = (m1 == -INFINITY) ? 0.f : m1;
          p0 = exp2f(s0 - mrun0);
          p1 = exp2f(s1 - mrun1);
        } else {
          p0 = ok ? exp2f(__fmaf_rn(acc.x, QKSCALE, -mrun0)) : 0.f;
          p1 = ok ? exp2f(__fmaf_rn(acc.y, QKSCALE, -mrun1)) : 0.f;
        }
      } else {
        const float s0 = ok ? acc.x * QKSCALE : -INFINITY;
        const float s1 = ok ? acc.y * QKSCALE : -INFINITY;
        float g0, g1;
        float m0 = s0, m1 = s1;
        if constexpr (TILE == 32) {
          m0 = warp_max_f32(m0);
          m1 = warp_max_f32(m1);
        } else {
          const bool lo = lane < 16;
          const float lo0 = warp_max_f32(lo ? m0 : -INFINITY);
          const float hi0 = warp_max_f32(lo ? -INFINITY : m0);
          m0 = lo ? lo0 : hi0;
          const float lo1 = warp_max_f32(lo ? m1 : -INFINITY);
          const float hi1 = warp_max_f32(lo ? -INFINITY : m1);
          m1 = lo ? lo1 : hi1;
        }
        const float mn0 = fmaxf(mrun0, m0), mn1 = fmaxf(mrun1, m1);
        g0 = (mn0 == -INFINITY) ? 0.f : mn0;
        g1 = (mn1 == -INFINITY) ? 0.f : mn1;
        a0 = exp2f(mrun0 - g0);
        a1 = exp2f(mrun1 - g1);
        mrun0 = mn0;
        mrun1 = mn1;
        p0 = exp2f(s0 - g0);
        p1 = exp2f(s1 - g1);
      }
      __nv_bfloat162 pk = __floats2bfloat162_rn(p0, p1);
      const float2 used_p = __bfloat1622float2(pk);
      p0 = used_p.x;
      p1 = used_p.y;
      // Online normalization bounds every probability by one; each fixed-point
      // tile reduction fits uint32 before conversion to the FP32 running sum.
      constexpr float kProbScale = kAnchor ? 2097152.f : 67108864.f;
      constexpr float kInvProbScale = 1.0f / kProbScale;
      const uint32_t fp0 = __float2uint_rn(p0 * kProbScale);
      const uint32_t fp1 = __float2uint_rn(p1 * kProbScale);
      uint32_t psum0, psum1;
      if constexpr (TILE == 32) {
        psum0 = warp_add_u32<32>(fp0);
        psum1 = warp_add_u32<32>(fp1);
      } else {
        const bool lo = lane < 16;
        const uint32_t lo0 = warp_add_u32<32>(lo ? fp0 : 0);
        const uint32_t hi0 = warp_add_u32<32>(lo ? 0 : fp0);
        psum0 = lo ? lo0 : hi0;
        const uint32_t lo1 = warp_add_u32<32>(lo ? fp1 : 0);
        const uint32_t hi1 = warp_add_u32<32>(lo ? 0 : fp1);
        psum1 = lo ? lo1 : hi1;
      }
      if constexpr (kAnchor) {
        lfix0 += psum0;
        lfix1 += psum1;
      } else {
        const float t0 = float(psum0) * kInvProbScale;
        const float t1 = float(psum1) * kInvProbScale;
        lrun0 = lrun0 * a0 + t0;
        lrun1 = lrun1 * a1 + t1;
      }
      if (tl + 1 == ntile && soft_lane == 0) {
        Norm[h0] = lrun0 > 0.f ? 1.f / lrun0 : 0.f;
        Norm[h0 + 1] = lrun1 > 0.f ? 1.f / lrun1 : 0.f;
        float2* ml = (float2*)(smem + SM_ML_FOR(TILE));
        ml[h0] = make_float2(mrun0, lrun0);
        ml[h0 + 1] = make_float2(mrun1, lrun1);
      }
      *(uint32_t*)(Pb + soft_lane * 48 + h0 * 2) = *(uint32_t*)&pk;
      if constexpr (!kAnchor) {
        if (soft_lane == 0) {
          Alp[h0] = a0;
          Alp[h0 + 1] = a1;
        }
      }
    }
    __syncthreads();

    // ---- PV: O^T[64 d][16 heads] for this warp's d slice
    {
      float av[2][2];
      if constexpr (!kAnchor) {
        const float2 av0 = *reinterpret_cast<const float2*>(Alp + shd);
        const float2 av1 = *reinterpret_cast<const float2*>(Alp + 8 + shd);
        av[0][0] = av0.x;
        av[0][1] = av0.y;
        av[1][0] = av1.x;
        av[1][1] = av1.y;
      } else {
        av[0][0] = av[0][1] = av[1][0] = av[1][1] = 1.f;
      }
      if constexpr (false && TILE == 32 && CSZ == 4 && !GROUPED) {
        uint32_t b[4], a0[4];
        ldm4t(sa(Pb + (lane & 15) * 48 + (lane >> 4) * 16), b[0], b[1], b[2], b[3]);
        int vrow = vrow_b;
        int j0 = (warp * (512 / NWARP)) >> 3;
        ldm4t(sa(kvb + vrow * 1024 + swz(vrow, j0 + vjo) * 16), a0[0], a0[1], a0[2], a0[3]);
#pragma unroll
        for (int mi = 0; mi < MIO; ++mi)
#pragma unroll
          for (int n = 0; n < 2; ++n)
#pragma unroll
            for (int r = 0; r < 4; ++r) o[mi][n][r] *= av[n][r & 1];

        int j1 = (warp * (512 / NWARP) + 16) >> 3;
        ldm4t(sa(kvb + vrow * 1024 + swz(vrow, j1 + vjo) * 16), qf[0][0], qf[0][1], qf[0][2],
              qf[0][3]);
        mma_bf16(o[0][0], a0, b[0], b[1]);
        mma_bf16(o[0][1], a0, b[2], b[3]);
        mma_bf16(o[1][0], qf[0], b[0], b[1]);
        mma_bf16(o[1][1], qf[0], b[2], b[3]);
        j0 = (warp * (512 / NWARP) + 32) >> 3;
        j1 = (warp * (512 / NWARP) + 48) >> 3;
        ldm4t(sa(kvb + vrow * 1024 + swz(vrow, j0 + vjo) * 16), a0[0], a0[1], a0[2], a0[3]);
        ldm4t(sa(kvb + vrow * 1024 + swz(vrow, j1 + vjo) * 16), qf[0][0], qf[0][1], qf[0][2],
              qf[0][3]);
        mma_bf16(o[2][0], a0, b[0], b[1]);
        mma_bf16(o[2][1], a0, b[2], b[3]);
        mma_bf16(o[3][0], qf[0], b[0], b[1]);
        mma_bf16(o[3][1], qf[0], b[2], b[3]);

        ldm4t(sa(Pb + (16 + (lane & 15)) * 48 + (lane >> 4) * 16), b[0], b[1], b[2], b[3]);
        vrow = 16 + vrow_b;
#pragma unroll
        for (int mi = 0; mi < MIO; mi += 2) {
          j0 = (warp * (512 / NWARP) + mi * 16) >> 3;
          j1 = (warp * (512 / NWARP) + (mi + 1) * 16) >> 3;
          ldm4t(sa(kvb + vrow * 1024 + swz(vrow, j0 + vjo) * 16), a0[0], a0[1], a0[2], a0[3]);
          ldm4t(sa(kvb + vrow * 1024 + swz(vrow, j1 + vjo) * 16), qf[0][0], qf[0][1], qf[0][2],
                qf[0][3]);
          mma_bf16(o[mi][0], a0, b[0], b[1]);
          mma_bf16(o[mi][1], a0, b[2], b[3]);
          mma_bf16(o[mi + 1][0], qf[0], b[0], b[1]);
          mma_bf16(o[mi + 1][1], qf[0], b[2], b[3]);
        }
      } else {
        if constexpr (!kAnchor) {
#pragma unroll
          for (int mi = 0; mi < MIO; ++mi)
#pragma unroll
            for (int n = 0; n < 2; ++n) {
              mul_f32x2(o[mi][n][0], o[mi][n][1], av[n][0], av[n][1]);
              mul_f32x2(o[mi][n][2], o[mi][n][3], av[n][0], av[n][1]);
            }
        }

#pragma unroll
        for (int ki = 0; ki < TILE / 16; ++ki) {
          uint32_t b[4];
          ldm4t(sa(Pb + (ki * 16 + (lane & 15)) * 48 + (lane >> 4) * 16), b[0], b[1], b[2], b[3]);
          int vrow = ki * 16 + vrow_b;
          if constexpr (GROUPED) {
            uint32_t a[MIO][4];
#pragma unroll
            for (int mi = 0; mi < MIO; ++mi) {
              int j0 = (warp * (512 / NWARP) + mi * 16) >> 3;
              ldm4t(sa(kvb + vrow * 1024 + swz(vrow, j0 + vjo) * 16), a[mi][0], a[mi][1], a[mi][2],
                    a[mi][3]);
            }
#pragma unroll
            for (int mi = 0; mi < MIO; ++mi) {
              mma_bf16(o[mi][0], a[mi], b[0], b[1]);
              mma_bf16(o[mi][1], a[mi], b[2], b[3]);
            }
          } else {
            if constexpr (TILE == 32) {
              const uint32_t vb = kUseG4 ? sa(kvb)
                                         : pv_base + (uint32_t)((tl & 1) * (TILE * 1024)) +
                                               (uint32_t)(ki * 16 * 1024);
#pragma unroll
              if constexpr (kPvPrefetch) {
                uint32_t av[MIO][4];
#pragma unroll
                for (int mi = 0; mi < MIO; ++mi)
                  ldm4t(vb + goff[mi], av[mi][0], av[mi][1], av[mi][2], av[mi][3]);
#pragma unroll
                for (int mi = 0; mi < MIO; ++mi) {
                  mma_bf16(o[mi][0], av[mi], b[0], b[1]);
                  mma_bf16(o[mi][1], av[mi], b[2], b[3]);
                }
              } else
                for (int mi = 0; mi < MIO; mi += 2) {
                  uint32_t a0[4];
                  if constexpr (kUseG4) {
#ifdef ENABLE_G4
                    const int j0 = (warp * (512 / NWARP) + mi * 16) >> 3;
                    const int j1 = (warp * (512 / NWARP) + (mi + 1) * 16) >> 3;
                    ldm4t(vb + (uint32_t)g4off(vrow, j0 + vjo), a0[0], a0[1], a0[2], a0[3]);
                    ldm4t(vb + (uint32_t)g4off(vrow, j1 + vjo), qf[0][0], qf[0][1], qf[0][2],
                          qf[0][3]);
#endif
                  } else {
                    ldm4t(vb + goff[mi], a0[0], a0[1], a0[2], a0[3]);
                    ldm4t(vb + goff[mi + 1], qf[0][0], qf[0][1], qf[0][2], qf[0][3]);
                  }
                  mma_bf16(o[mi][0], a0, b[0], b[1]);
                  mma_bf16(o[mi][1], a0, b[2], b[3]);
                  mma_bf16(o[mi + 1][0], qf[0], b[0], b[1]);
                  mma_bf16(o[mi + 1][1], qf[0], b[2], b[3]);
                }
            } else {
#pragma unroll
              for (int mi = 0; mi < MIO; ++mi) {
                int j0 = (warp * (512 / NWARP) + mi * 16) >> 3;
                uint32_t a[4];
                ldm4t(sa(kvb + vrow * 1024 + swz(vrow, j0 + vjo) * 16), a[0], a[1], a[2], a[3]);
                mma_bf16(o[mi][0], a, b[0], b[1]);
                mma_bf16(o[mi][1], a, b[2], b[3]);
              }
            }
          }
        }
      }
    }
    if constexpr (!GROUPED && !kC4OverlapQReload) {
      if (tl + 1 < ntile) {
        const int qj0 = (kk * 128) >> 3;
        ldm4(sa(Qs + qrow * 1024 + swz(qrow, qj0 + qjo) * 16), qf[0][0], qf[0][1], qf[0][2],
             qf[0][3]);
      }
    }
    if (tl + 2 < ntile) __syncthreads();
  }

  if constexpr (kAnchor) {
    lrun0 = float(lfix0) * (1.0f / 2097152.f);
    lrun1 = float(lfix1) * (1.0f / 2097152.f);
  }

  // ---- cluster epilogue: reduce split partials through distributed smem
  {
    cg::cluster_group clu = cg::this_cluster();
    const unsigned rank = clu.block_rank();
    constexpr bool kBalanced7 = BAL7 && TILE == 32 && CSZ == 7 && !GROUPED;
    const int b7_group0 = 9 * static_cast<int>(rank) + min(static_cast<int>(rank), 1);
    const int b7_ngroup = rank == 0 ? 10 : 9;
    const int b7_first_h = b7_group0 >> 2;
    const int b7_last_h = (b7_group0 + b7_ngroup - 1) >> 2;
    // Q fragments live in registers after setup, so recycle their disjoint
    // 16 KiB staging region for DSM partials.  This avoids aliasing the final
    // PV reads from the KV ping-pong buffer and removes the last CTA barrier.
    __nv_bfloat16* Osh = (__nv_bfloat16*)(smem + SM_Q);
    float2* MLsh = (float2*)(smem + SM_ML_FOR(TILE));
    constexpr int kDsmXor = 8;
    float2 norms[2];
#pragma unroll
    for (int n = 0; n < 2; ++n) {
      norms[n] = *(const float2*)(Norm + 8 * n + 2 * (lane & 3));
    }

#pragma unroll
    for (int mi = 0; mi < MIO; ++mi) {
      const int dbase = warp * (512 / NWARP) + mi * 16;
#pragma unroll
      for (int n = 0; n < 2; ++n) {
        const int hh = 8 * n + (lane & 7);
        const int d8 = dbase + (((lane >> 3) & 1) * 8);
        const int dx = kDsmXor * (hh & 7);
        const float inv0 = norms[n].x, inv1 = norms[n].y;
        mul_f32x2(o[mi][n][0], o[mi][n][1], inv0, inv1);
        mul_f32x2(o[mi][n][2], o[mi][n][3], inv0, inv1);
        const __nv_bfloat162 r0 = __floats2bfloat162_rn(o[mi][n][0], o[mi][n][1]);
        const __nv_bfloat162 r1 = __floats2bfloat162_rn(o[mi][n][2], o[mi][n][3]);
        stm2t(sa(Osh + hh * DIM + (d8 ^ dx)), *reinterpret_cast<const uint32_t*>(&r0),
              *reinterpret_cast<const uint32_t*>(&r1));
      }
    }
    clu.sync();
    float* Wsh = (float*)(smem + SM_S_FOR(TILE));
    uint32_t* W2sh = (uint32_t*)(smem + SM_S_FOR(TILE));
    constexpr bool kPackedMerge =
        !GROUPED &&
        ((TILE == 32 && (((CSZ == 4 || CSZ == 5 || CSZ == 7) && !OWNER) || (CSZ == 10 && OWNER))) ||
         (TILE == 16 && CSZ == 5));
    const bool owns_merge_head =
        kBalanced7 ? (tid >= b7_first_h && tid <= b7_last_h) : rank == (unsigned)((tid >> 1) % CSZ);
    // Four width-4 groups form all rank weights in parallel.  Each lane
    // retains its local exponential and divides by the subgroup denominator,
    // avoiding a reciprocal broadcast on the c4 merge critical path.
    if constexpr (TILE == 32 && CSZ == 4 && !GROUPED && !OWNER) {
      if (tid < 16) {
        const int p = tid & 3;
        const int g = tid >> 2;
        const int h = 2 * static_cast<int>(rank) + (g & 1) + 8 * (g >> 1);
        const float sk = sink_log2(sinks[h]);
        const float2 ml = ((float2*)clu.map_shared_rank(MLsh, p))[h];
        float M = p == 0 ? fmaxf(sk, ml.x) : ml.x;
        M = fmaxf(M, __shfl_xor_sync(0x0000ffffu, M, 1, 4));
        M = fmaxf(M, __shfl_xor_sync(0x0000ffffu, M, 2, 4));
        const float e = exp2f(ml.x - M);
        float den = ml.y * e + (p == 0 ? exp2f(sk - M) : 0.f);
        den += __shfl_xor_sync(0x0000ffffu, den, 1, 4);
        den += __shfl_xor_sync(0x0000ffffu, den, 2, 4);
        const float w = ml.y * e / den;
        const uint32_t wb = __float_as_uint(w);
        W2sh[h * CSZ + p] = *reinterpret_cast<const uint32_t*>(&wb);
      }
    } else if (tid < HEADS &&
               ((TILE == 32 && !GROUPED) ? owns_merge_head : (!OWNER || owns_merge_head))) {
      const int h = tid;
      const float sk = sink_log2(sinks[h]);
      float M = sk;
      float2 mlv[CSZ];
#pragma unroll
      for (int p = 0; p < CSZ; ++p) {
        if constexpr (TILE == 32 && CSZ == 5 && !GROUPED && !OWNER) {
          const uint2 bits = dsm_ld2_c5(MLsh + h, p);
          mlv[p] = *reinterpret_cast<const float2*>(&bits);
        } else {
          mlv[p] = ((float2*)clu.map_shared_rank(MLsh, p))[h];
        }
      }
#pragma unroll
      for (int p = 0; p < CSZ; ++p) M = fmaxf(M, mlv[p].x);
      float den = exp2f(sk - M);
#pragma unroll
      for (int p = 0; p < CSZ; ++p) {
        mlv[p].x = exp2f(mlv[p].x - M);
        den += mlv[p].y * mlv[p].x;
      }
      const float inv = 1.0f / den;
#pragma unroll
      for (int p = 0; p < CSZ; ++p) {
        const float w = mlv[p].x * mlv[p].y * inv;
        if constexpr (kPackedMerge) {
          const uint32_t wb = __float_as_uint(w);
          W2sh[p * HEADS + h] = *reinterpret_cast<const uint32_t*>(&wb);
        } else {
          Wsh[p * HEADS + h] = w;
        }
      }
    }
    // Start the first remote DSM read before the local merge-weight barrier;
    // its round trip can overlap the head-wise max/exp/denominator work.
    const int pref_c0 = kBalanced7
                            ? (b7_group0 + warp) * 32 + lane
                            : (TILE == 16 ? ((warp * CSZ + static_cast<int>(rank)) * 32 + lane)
                                          : static_cast<int>(rank) * NTHREAD + tid);
    const int pref_nchunk = HEADS * DIM / 4;
    const int b7_nextra = b7_ngroup - NWARP;
    const bool b7_has_extra = warp >= NWARP - b7_nextra;
    const int b7_extra_idx = warp - (NWARP - b7_nextra);
    const int pref_c1 =
        kBalanced7 ? (b7_group0 + NWARP + b7_extra_idx) * 32 + lane : pref_c0 + CSZ * NTHREAD;
    constexpr bool kCanPrefetchSecond = !GROUPED && (TILE == 16 || CSZ * NTHREAD < HEADS * DIM / 4);
    constexpr bool kC4Wide = TILE == 32 && CSZ == 4 && !GROUPED && !OWNER;
    uint4 wide_v[CSZ];
    int wide_h = 0, wide_d8 = 0;
    if constexpr (kC4Wide) {
      const int wg = tid >> 6;
      wide_h = 2 * static_cast<int>(rank) + (wg & 1) + 8 * (wg >> 1);
      wide_d8 = (tid & 63) * 8;
      const int wide_sd8 = wide_d8 ^ (kDsmXor * (wide_h & 7));
#pragma unroll
      for (int p = 0; p < CSZ; ++p)
        wide_v[p] = *(const uint4*)clu.map_shared_rank(Osh + wide_h * DIM + wide_sd8, p);
    }
    uint2 pref_v1[CSZ];
    int pref_h1 = 0, pref_d41 = 0;
    bool has_pref1 = false;
    if constexpr (kCanPrefetchSecond && !kC4Wide) {
      has_pref1 = kBalanced7 ? b7_has_extra : (pref_c1 < pref_nchunk);
      if (has_pref1) {
        pref_h1 = pref_c1 >> 7;
        pref_d41 = (pref_c1 & 127) * 4;
        const int pref_sd41 = pref_d41 ^ (kDsmXor * (pref_h1 & 7));
#pragma unroll
        for (int p = 0; p < CSZ; ++p) {
          if constexpr (TILE == 32 && CSZ == 5 && !GROUPED && !OWNER)
            pref_v1[p] = dsm_ld2_c5(Osh + pref_h1 * DIM + pref_sd41, p);
          else
            pref_v1[p] = *(const uint2*)clu.map_shared_rank(Osh + pref_h1 * DIM + pref_sd41, p);
        }
      }
    }
    uint2 pref_v[CSZ];
    int pref_h = 0, pref_d4 = 0;
    const bool has_pref = !GROUPED && (kBalanced7 ? (warp < b7_ngroup) : (pref_c0 < pref_nchunk));
    if (has_pref && !kC4Wide) {
      pref_h = pref_c0 >> 7;
      pref_d4 = (pref_c0 & 127) * 4;
      const int pref_sd4 = pref_d4 ^ (kDsmXor * (pref_h & 7));
#pragma unroll
      for (int p = 0; p < CSZ; ++p) {
        if constexpr (TILE == 32 && CSZ == 5 && !GROUPED && !OWNER)
          pref_v[p] = dsm_ld2_c5(Osh + pref_h * DIM + pref_sd4, p);
        else
          pref_v[p] = *(const uint2*)clu.map_shared_rank(Osh + pref_h * DIM + pref_sd4, p);
      }
    }
    // Synchronize local weights; consume every DSM value before terminal arrival.
    __syncthreads();

    if constexpr (kC4Wide) {
      const uint4 wv = *(const uint4*)(W2sh + wide_h * CSZ);
      const uint32_t wq[CSZ] = {wv.x, wv.y, wv.z, wv.w};
      MergePair a0 = 0, a1 = 0, a2 = 0, a3 = 0;
#pragma unroll
      for (int p = 0; p < CSZ; ++p) {
        a0 = merge_f32x2(wide_v[p].x, wq[p], a0);
        a1 = merge_f32x2(wide_v[p].y, wq[p], a1);
        a2 = merge_f32x2(wide_v[p].z, wq[p], a2);
        a3 = merge_f32x2(wide_v[p].w, wq[p], a3);
      }
      *(uint4*)(out + ((size_t)t * HEADS + wide_h) * DIM + wide_d8) = make_uint4(a0, a1, a2, a3);
    } else if constexpr (TILE == 32 && CSZ == 4 && !GROUPED) {
      // All cluster-4 lanes own two valid output chunks. Interleave four
      // independent BF16x2 chains to cover the packed-FMA dependency latency.
      MergePair a00 = 0, a01 = 0, a10 = 0, a11 = 0;
      const uint4 wv0 = *(const uint4*)(W2sh + pref_h * CSZ);
      const uint4 wv1 = *(const uint4*)(W2sh + pref_h1 * CSZ);
      const uint32_t w0[CSZ] = {wv0.x, wv0.y, wv0.z, wv0.w};
      const uint32_t w1[CSZ] = {wv1.x, wv1.y, wv1.z, wv1.w};
#pragma unroll
      for (int p = 0; p < CSZ; ++p) {
        a00 = merge_f32x2(pref_v[p].x, w0[p], a00);
        a10 = merge_f32x2(pref_v1[p].x, w1[p], a10);
        a01 = merge_f32x2(pref_v[p].y, w0[p], a01);
        a11 = merge_f32x2(pref_v1[p].y, w1[p], a11);
      }
      *(uint2*)(out + ((size_t)t * HEADS + pref_h) * DIM + pref_d4) = make_uint2(a00, a01);
      *(uint2*)(out + ((size_t)t * HEADS + pref_h1) * DIM + pref_d41) = make_uint2(a10, a11);
    } else if constexpr (TILE == 32 && CSZ == 5 && !GROUPED) {
      // Interleave the two independent c5 merge stripes to hide packed-FMA
      // dependency latency while preserving each stripe's reduction order.
      uint2 c5_out0 = make_uint2(0, 0);
      uint2 c5_out1 = make_uint2(0, 0);
      if (has_pref1) {
        MergePair a00 = 0, a01 = 0, a10 = 0, a11 = 0;
#pragma unroll
        for (int p = 0; p < CSZ; ++p) {
          const uint32_t w0 = W2sh[p * HEADS + pref_h];
          const uint32_t w1 = W2sh[p * HEADS + pref_h1];
          a00 = merge_f32x2(pref_v[p].x, w0, a00);
          a10 = merge_f32x2(pref_v1[p].x, w1, a10);
          a01 = merge_f32x2(pref_v[p].y, w0, a01);
          a11 = merge_f32x2(pref_v1[p].y, w1, a11);
        }
        c5_out0 = make_uint2(a00, a01);
        c5_out1 = make_uint2(a10, a11);
      } else if (has_pref) {
        MergePair a0 = 0, a1 = 0;
#pragma unroll
        for (int p = 0; p < CSZ; ++p) {
          const uint32_t w = W2sh[p * HEADS + pref_h];
          a0 = merge_f32x2(pref_v[p].x, w, a0);
          a1 = merge_f32x2(pref_v[p].y, w, a1);
        }
        c5_out0 = make_uint2(a0, a1);
      }
      if (has_pref1) {
        *(uint2*)(out + ((size_t)t * HEADS + pref_h) * DIM + pref_d4) = c5_out0;
        *(uint2*)(out + ((size_t)t * HEADS + pref_h1) * DIM + pref_d41) = c5_out1;
      } else if (has_pref) {
        *(uint2*)(out + ((size_t)t * HEADS + pref_h) * DIM + pref_d4) = c5_out0;
      }
    } else if constexpr (TILE == 16 && CSZ == 5 && !GROUPED) {
      // B13 also has two universally valid prefetched stripes.  Interleave
      // their four BF16x2 chains to hide the dependency latency without
      // increasing DSM traffic or changing each chain's reduction order.
      MergePair a00 = 0, a01 = 0, a10 = 0, a11 = 0;
#pragma unroll
      for (int p = 0; p < CSZ; ++p) {
        const uint32_t w0 = W2sh[p * HEADS + pref_h];
        const uint32_t w1 = W2sh[p * HEADS + pref_h1];
        a00 = merge_f32x2(pref_v[p].x, w0, a00);
        a10 = merge_f32x2(pref_v1[p].x, w1, a10);
        a01 = merge_f32x2(pref_v[p].y, w0, a01);
        a11 = merge_f32x2(pref_v1[p].y, w1, a11);
      }
      *(uint2*)(out + ((size_t)t * HEADS + pref_h) * DIM + pref_d4) = make_uint2(a00, a01);
      *(uint2*)(out + ((size_t)t * HEADS + pref_h1) * DIM + pref_d41) = make_uint2(a10, a11);
    } else {
      if constexpr (kCanPrefetchSecond) {
        if (has_pref1) {
          if constexpr (kPackedMerge) {
            MergePair a0 = 0, a1 = 0;
#pragma unroll
            for (int p = 0; p < CSZ; ++p) {
              const uint32_t wr = W2sh[p * HEADS + pref_h1];
              a0 = merge_f32x2(pref_v1[p].x, wr, a0);
              a1 = merge_f32x2(pref_v1[p].y, wr, a1);
            }
            const uint2 pk = make_uint2(a0, a1);
            *(uint2*)(out + ((size_t)t * HEADS + pref_h1) * DIM + pref_d41) = pk;
          } else {
            float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
            for (int p = 0; p < CSZ; ++p) {
              const float w = Wsh[p * HEADS + pref_h1];
              const __nv_bfloat162 b0 = *reinterpret_cast<const __nv_bfloat162*>(&pref_v1[p].x);
              const __nv_bfloat162 b1 = *reinterpret_cast<const __nv_bfloat162*>(&pref_v1[p].y);
              const float2 f0 = __bfloat1622float2(b0);
              const float2 f1 = __bfloat1622float2(b1);
              a.x = fmaf(f0.x, w, a.x);
              a.y = fmaf(f0.y, w, a.y);
              a.z = fmaf(f1.x, w, a.z);
              a.w = fmaf(f1.y, w, a.w);
            }
            const __nv_bfloat162 o0 = __floats2bfloat162_rn(a.x, a.y);
            const __nv_bfloat162 o1 = __floats2bfloat162_rn(a.z, a.w);
            const uint2 pk = make_uint2(*(const uint32_t*)&o0, *(const uint32_t*)&o1);
            *(uint2*)(out + ((size_t)t * HEADS + pref_h1) * DIM + pref_d41) = pk;
          }
        }
      }
      if (has_pref) {
        if constexpr (kPackedMerge) {
          MergePair a0 = 0, a1 = 0;
#pragma unroll
          for (int p = 0; p < CSZ; ++p) {
            const uint32_t wr = W2sh[p * HEADS + pref_h];
            a0 = merge_f32x2(pref_v[p].x, wr, a0);
            a1 = merge_f32x2(pref_v[p].y, wr, a1);
          }
          const uint2 pk = make_uint2(a0, a1);
          *(uint2*)(out + ((size_t)t * HEADS + pref_h) * DIM + pref_d4) = pk;
        } else {
          float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
          for (int p = 0; p < CSZ; ++p) {
            const float w = Wsh[p * HEADS + pref_h];
            const __nv_bfloat162 b0 = *reinterpret_cast<const __nv_bfloat162*>(&pref_v[p].x);
            const __nv_bfloat162 b1 = *reinterpret_cast<const __nv_bfloat162*>(&pref_v[p].y);
            const float2 f0 = __bfloat1622float2(b0);
            const float2 f1 = __bfloat1622float2(b1);
            a.x = fmaf(f0.x, w, a.x);
            a.y = fmaf(f0.y, w, a.y);
            a.z = fmaf(f1.x, w, a.z);
            a.w = fmaf(f1.y, w, a.w);
          }
          const __nv_bfloat162 o0 = __floats2bfloat162_rn(a.x, a.y);
          const __nv_bfloat162 o1 = __floats2bfloat162_rn(a.z, a.w);
          const uint2 pk = make_uint2(*(const uint32_t*)&o0, *(const uint32_t*)&o1);
          *(uint2*)(out + ((size_t)t * HEADS + pref_h) * DIM + pref_d4) = pk;
        }
      }
    }
    if constexpr (GROUPED) {
      const int nchunk = HEADS * DIM / 8;
      for (int c = rank * NTHREAD + tid; c < nchunk; c += CSZ * NTHREAD) {
        const int h = c >> 6;
        const int d8 = (c & 63) * 8;
        const int sd8 = d8 ^ (kDsmXor * (h & 7));
        uint4 v[CSZ];
#pragma unroll
        for (int p = 0; p < CSZ; ++p)
          v[p] = *(const uint4*)clu.map_shared_rank(Osh + h * DIM + sd8, p);
        float4 a0 = make_float4(0.f, 0.f, 0.f, 0.f);
        float4 a1 = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int p = 0; p < CSZ; ++p) {
          const float w = Wsh[p * HEADS + h];
          const __nv_bfloat162 b0 = *reinterpret_cast<const __nv_bfloat162*>(&v[p].x);
          const __nv_bfloat162 b1 = *reinterpret_cast<const __nv_bfloat162*>(&v[p].y);
          const __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(&v[p].z);
          const __nv_bfloat162 b3 = *reinterpret_cast<const __nv_bfloat162*>(&v[p].w);
          const float2 f0 = __bfloat1622float2(b0);
          const float2 f1 = __bfloat1622float2(b1);
          const float2 f2 = __bfloat1622float2(b2);
          const float2 f3 = __bfloat1622float2(b3);
          a0.x = fmaf(f0.x, w, a0.x);
          a0.y = fmaf(f0.y, w, a0.y);
          a0.z = fmaf(f1.x, w, a0.z);
          a0.w = fmaf(f1.y, w, a0.w);
          a1.x = fmaf(f2.x, w, a1.x);
          a1.y = fmaf(f2.y, w, a1.y);
          a1.z = fmaf(f3.x, w, a1.z);
          a1.w = fmaf(f3.y, w, a1.w);
        }
        __nv_bfloat162 o0 = __floats2bfloat162_rn(a0.x, a0.y);
        __nv_bfloat162 o1 = __floats2bfloat162_rn(a0.z, a0.w);
        __nv_bfloat162 o2 = __floats2bfloat162_rn(a1.x, a1.y);
        __nv_bfloat162 o3 = __floats2bfloat162_rn(a1.z, a1.w);
        uint4 pk = make_uint4(*(uint32_t*)&o0, *(uint32_t*)&o1, *(uint32_t*)&o2, *(uint32_t*)&o3);
        *(uint4*)(out + ((size_t)t * HEADS + h) * DIM + d8) = pk;
      }
    } else {
      const int nchunk = HEADS * DIM / 4;
      const int c0 = TILE == 16 ? ((warp * CSZ + static_cast<int>(rank)) * 32 + lane)
                                : static_cast<int>(rank) * NTHREAD + tid;
      const int cstart = kBalanced7 ? nchunk : c0 + 2 * CSZ * NTHREAD;
      for (int c = cstart; c < nchunk; c += CSZ * NTHREAD) {
        const int h = c >> 7;
        const int d4 = (c & 127) * 4;
        const int sd4 = d4 ^ (kDsmXor * (h & 7));
        uint2 v[CSZ];
#pragma unroll
        for (int p = 0; p < CSZ; ++p)
          v[p] = *(const uint2*)clu.map_shared_rank(Osh + h * DIM + sd4, p);
        if constexpr (kPackedMerge) {
          MergePair a0 = 0, a1 = 0;
#pragma unroll
          for (int p = 0; p < CSZ; ++p) {
            const uint32_t w = W2sh[p * HEADS + h];
            a0 = merge_f32x2(v[p].x, w, a0);
            a1 = merge_f32x2(v[p].y, w, a1);
          }
          *(uint2*)(out + ((size_t)t * HEADS + h) * DIM + d4) = make_uint2(a0, a1);
        } else {
          float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
          for (int p = 0; p < CSZ; ++p) {
            const float w = Wsh[p * HEADS + h];
            const __nv_bfloat162 b0 = *reinterpret_cast<const __nv_bfloat162*>(&v[p].x);
            const __nv_bfloat162 b1 = *reinterpret_cast<const __nv_bfloat162*>(&v[p].y);
            const float2 f0 = __bfloat1622float2(b0);
            const float2 f1 = __bfloat1622float2(b1);
            a.x = fmaf(f0.x, w, a.x);
            a.y = fmaf(f0.y, w, a.y);
            a.z = fmaf(f1.x, w, a.z);
            a.w = fmaf(f1.y, w, a.w);
          }
          __nv_bfloat162 o0 = __floats2bfloat162_rn(a.x, a.y);
          __nv_bfloat162 o1 = __floats2bfloat162_rn(a.z, a.w);
          uint2 pk = make_uint2(*(uint32_t*)&o0, *(uint32_t*)&o1);
          *(uint2*)(out + ((size_t)t * HEADS + h) * DIM + d4) = pk;
        }
      }
    }
    asm volatile(
        "barrier.cluster.arrive.relaxed.aligned;\n\t"
        "barrier.cluster.wait.aligned;\n" ::
            : "memory");
  }
}

// Fixed two-tile path for the B=1/2 cluster-10 regimes.  Each rank owns
// exactly 64 keys, so both 32-row halves share one softmax update.
#define F2_Q 0
#define F2_KV 16384
#define F2_S (F2_KV + 2 * 32 * 1024)
#define F2_P (F2_S + 2 * 4 * SCORE_QSTRIDE * 4)
#define F2_ML (F2_P + 2 * 32 * 48)
#define F2_IDX (F2_ML + 128)
#define F2_NORM (F2_IDX + 64 * 4)
#define F2_BYTES (F2_NORM + 64)

// B=1 specialization: two independent 8-head partitions provide twice as
// many cluster CTAs while halving the Q, score, probability, and DSM state.
#define H8_HEADS 8
#define H8_Q 0
#define H8_KV 8192
#define H8_SCORE_QSTRIDE (32 * 10)
#define H8_S (H8_KV + 2 * 32 * 1024)
#define H8_P (H8_S + 2 * 4 * H8_SCORE_QSTRIDE * 4)
#define H8_PSTRIDE 16
#define H8_ML (H8_P + 2 * 32 * H8_PSTRIDE)
#define H8_IDX (H8_ML + H8_HEADS * sizeof(float2))
#define H8_NORM (H8_IDX + 64 * 4)
#define H8_BYTES (H8_NORM + H8_HEADS * 4)

__device__ __forceinline__ int score_off_h8(int row, int hp) { return row * 10 + 2 * hp; }

template <bool STAGE_WAIT = false, bool EARLY_ARRIVE = true, bool CONTIG_W = false,
          bool Q_ADDR_FOLD = false, bool PACKED_EX2 = false,
          bool DELAY_HALF1 = (STAGE_WAIT && !Q_ADDR_FOLD), bool SPLIT_DEN = false>
__global__ __launch_bounds__(256, 2) void mla_fused2_c10(
    const uint8_t* __restrict__ qp, const uint8_t* __restrict__ swap_,
    const uint8_t* __restrict__ ckvp, const int* __restrict__ sidx, const int* __restrict__ slen,
    const float* __restrict__ sinks, __nv_bfloat16* __restrict__ out) {
  extern __shared__ __align__(16) char smem[];
  char* Qs = smem + F2_Q;
  char* KVs = smem + F2_KV;
  float* Sb = (float*)(smem + F2_S);
  char* Pb = smem + F2_P;
  float2* MLsh = (float2*)(smem + F2_ML);
  int* Idx = (int*)(smem + F2_IDX);
  float* Norm = (float*)(smem + F2_NORM);

  const int t = blockIdx.y;
  const int sp = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int gcol0 = sp * 64;
  const int len = slen[t];
  const int* myidx = sidx + (size_t)t * WIDTH;
  {
    const uint8_t* src = qp + (size_t)t * (HEADS * DIM * 2);
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int ci = tid + r * 256;
      const int row = ci >> 6;
      const int ch = ci & 63;
      const uint32_t dst = sa(Qs + row * 1024 + swz(row, ch) * 16);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst),
                   "l"(src + (size_t)ci * 16));
    }
    asm volatile("cp.async.commit_group;\n");
  }
  if (tid < 64) {
    const int gcol = gcol0 + tid;
    // inactive tail columns carry no offset-validity contract: mask BEFORE
    // staging so they never reach a gather (softmax length mask unchanged)
    const int v = myidx[gcol];
    Idx[tid] = (gcol < len && v >= 0) ? v : -1;
  }
  __syncthreads();

  auto issue_half = [&](int half) {
    const int base = gcol0 + half * 32;
    const uint8_t* pool = base < 128 ? swap_ : ckvp;
    char* dst = KVs + half * 32 * 1024;
    int iv[4];
#pragma unroll
    for (int r = 0; r < 4; ++r)
      iv[r] =
          Idx[half * 32 + (STAGE_WAIT ? ((warp >> 2) * 16 + (warp & 3) + r * 4) : (warp + r * 8))];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int row = STAGE_WAIT ? ((warp >> 2) * 16 + (warp & 3) + r * 4) : (warp + r * 8);
      const int off = iv[r];
      const int cpsz = off >= 0 ? 16 : 0;  // zero-fill invalid rows
      const uint8_t* src = pool + (size_t)(off >= 0 ? off : 0) * 1024 + lane * 16;
      const uint32_t d = sa(dst + row * 1024 + swz(row, lane) * 16);
      asm volatile(
          "cp.async.cg.shared.global [%0], [%1], 16, %2;\n\t"
          "cp.async.cg.shared.global [%0+512], [%1+512], 16, %2;\n" ::"r"(d),
          "l"(src), "r"(cpsz));
    }
    asm volatile("cp.async.commit_group;\n");
  };
  issue_half(0);
  if constexpr (!DELAY_HALF1) issue_half(1);
  if constexpr (DELAY_HALF1)
    asm volatile("cp.async.wait_group 1;\n");
  else
    asm volatile("cp.async.wait_group 2;\n");
  __syncthreads();

  const int kk = STAGE_WAIT ? (warp & 3) : (warp >> 1);
  const int col0 = STAGE_WAIT ? ((warp >> 2) * 16) : ((warp & 1) * 16);
  const int qrow = (lane & 7) | ((lane & 16) >> 1);
  const int qjo = (lane >> 3) & 1;
  uint32_t qf[8][4];
  if constexpr (Q_ADDR_FOLD) {
    const uint32_t q_base = sa(Qs) + (uint32_t)(qrow * 1024 + kk * 256);
    const int qx = qrow & 7;
    uint32_t qoff[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) qoff[j] = (uint32_t)(((2 * j + qjo) ^ qx) * 16);
#pragma unroll
    for (int i = 0; i < 8; ++i)
      ldm4(q_base + qoff[i & 3] + (uint32_t)((i >> 2) * 128), qf[i][0], qf[i][1], qf[i][2],
           qf[i][3]);
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int j0 = (kk * 128 + i * 16) >> 3;
      ldm4(sa(Qs + qrow * 1024 + swz(qrow, j0 + qjo) * 16), qf[i][0], qf[i][1], qf[i][2], qf[i][3]);
    }
  }
  if constexpr (DELAY_HALF1) issue_half(1);
  if constexpr (STAGE_WAIT)
    asm volatile("cp.async.wait_group 1;\n");
  else
    asm volatile("cp.async.wait_group 0;\n");
  if constexpr (STAGE_WAIT) {
    if (warp < 4)
      asm volatile("barrier.cta.sync 1, 128;\n" ::: "memory");
    else
      asm volatile("barrier.cta.sync 2, 128;\n" ::: "memory");
  } else {
    __syncthreads();
  }

  const int arow = col0 + (lane & 15);
  const int ajo = lane >> 4;
  const int srow = col0 + (lane >> 2);
  const uint32_t qk_base0 = sa(KVs) + (uint32_t)(arow * 1024 + kk * 256);
  const int qk_xor = arow & 7;
  uint32_t qk_off[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) qk_off[j] = (uint32_t)((((2 * j + ajo) & 7) ^ qk_xor) * 16);
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    if constexpr (STAGE_WAIT) {
      if (half == 1) {
        asm volatile("cp.async.wait_group 0;\n");
        if (warp < 4)
          asm volatile("barrier.cta.sync 1, 128;\n" ::: "memory");
        else
          asm volatile("barrier.cta.sync 2, 128;\n" ::: "memory");
      }
    }
    char* kvb = KVs + half * 32 * 1024;
    float s[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    const uint32_t qb = qk_base0 + (uint32_t)(half * 32 * 1024);
    uint32_t a[4];
    ldm4(qb + qk_off[0], a[0], a[1], a[2], a[3]);
    mma_bf16(s[0], a, qf[0][0], qf[0][1]);
    mma_bf16(s[1], a, qf[0][2], qf[0][3]);

    ldm4(qb + qk_off[1], qf[0][0], qf[0][1], qf[0][2], qf[0][3]);
#pragma unroll
    for (int i = 1; i < 8; ++i) {
      const uint32_t next = qb + qk_off[(i + 1) & 3] + (uint32_t)(((i + 1) >> 2) * 128);
      if (i & 1) {
        mma_bf16(s[0], qf[0], qf[i][0], qf[i][1]);
        if (i + 1 < 8) ldm4(next, a[0], a[1], a[2], a[3]);
        mma_bf16(s[1], qf[0], qf[i][2], qf[i][3]);
      } else {
        mma_bf16(s[0], a, qf[i][0], qf[i][1]);
        if (i + 1 < 8) ldm4(next, qf[0][0], qf[0][1], qf[0][2], qf[0][3]);
        mma_bf16(s[1], a, qf[i][2], qf[i][3]);
      }
    }
    if (half == 0) {
      if constexpr (Q_ADDR_FOLD) {
        const uint32_t q_base = sa(Qs) + (uint32_t)(qrow * 1024 + kk * 256);
        const int qx = qrow & 7;
        const uint32_t qoff0 = (uint32_t)((qjo ^ qx) * 16);
        ldm4(q_base + qoff0, qf[0][0], qf[0][1], qf[0][2], qf[0][3]);
      } else {
        const int j0 = (kk * 128) >> 3;
        ldm4(sa(Qs + qrow * 1024 + swz(qrow, j0 + qjo) * 16), qf[0][0], qf[0][1], qf[0][2],
             qf[0][3]);
      }
    }
    float* sb = Sb + half * 4 * SCORE_QSTRIDE + kk * SCORE_QSTRIDE;
    const float2 s00 = make_float2(s[0][0], s[0][1]);
    const float2 s01 = make_float2(s[0][2], s[0][3]);
    const float2 s10 = make_float2(s[1][0], s[1][1]);
    const float2 s11 = make_float2(s[1][2], s[1][3]);
    const int hp = lane & 3;
    const int sx = (srow ^ (srow >> 2)) & 7;
    const int so0 = score_off(srow, hp);
    const int so1 = score_off(srow + 8, hp);
    *(float2*)(sb + so0) = s00;
    *(float2*)(sb + so1) = s01;
    *(float2*)(sb + so0 + 8) = s10;
    *(float2*)(sb + so1 + 8) = s11;
  }
  __syncthreads();

  const int h0 = 2 * warp;
  float mrun0, mrun1, lrun0, lrun1;
  {
    float sv[2][2];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      float* sb = Sb + half * 4 * SCORE_QSTRIDE;
      const int off = score_off(lane, h0 >> 1);
      const float2 f = sum_qk4(sb + off, SCORE_QSTRIDE);
      const bool ok = gcol0 + half * 32 + lane < len && Idx[half * 32 + lane] >= 0;
      sv[half][0] = ok ? f.x * QKSCALE : -INFINITY;
      sv[half][1] = ok ? f.y * QKSCALE : -INFINITY;
    }
    mrun0 = warp_max_f32(fmaxf(sv[0][0], sv[1][0]));
    mrun1 = warp_max_f32(fmaxf(sv[0][1], sv[1][1]));
    const float g0 = mrun0 == -INFINITY ? 0.f : mrun0;
    const float g1 = mrun1 == -INFINITY ? 0.f : mrun1;
    float pv[2][2];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      uint32_t pk;
      if constexpr (PACKED_EX2) {
        pk = ex2_bf16x2(sv[half][0] - g0, sv[half][1] - g1);
        pv[half][0] = __uint_as_float(pk << 16);
        pv[half][1] = __uint_as_float(pk & 0xffff0000u);
      } else {
        pv[half][0] = exp2f(sv[half][0] - g0);
        pv[half][1] = exp2f(sv[half][1] - g1);
        const __nv_bfloat162 bpk = __floats2bfloat162_rn(pv[half][0], pv[half][1]);
        pk = *reinterpret_cast<const uint32_t*>(&bpk);
        const float2 used_p = __bfloat1622float2(bpk);
        pv[half][0] = used_p.x;
        pv[half][1] = used_p.y;
      }
      *(uint32_t*)(Pb + half * 32 * 48 + lane * 48 + h0 * 2) = pk;
    }
    constexpr float kProbScale = 33554432.f;
    constexpr float kInvProbScale = 1.f / kProbScale;
    const uint32_t fp0 = __float2uint_rn((pv[0][0] + pv[1][0]) * kProbScale);
    const uint32_t fp1 = __float2uint_rn((pv[0][1] + pv[1][1]) * kProbScale);
    lrun0 = float(warp_add_u32<32>(fp0)) * kInvProbScale;
    lrun1 = float(warp_add_u32<32>(fp1)) * kInvProbScale;
  }
  if (lane == 0) {
    Norm[h0] = lrun0 > 0.f ? 1.f / lrun0 : 0.f;
    Norm[h0 + 1] = lrun1 > 0.f ? 1.f / lrun1 : 0.f;
    MLsh[h0] = make_float2(mrun0, lrun0);
    MLsh[h0 + 1] = make_float2(mrun1, lrun1);
  }
  __syncthreads();

  float o[4][2][4];
#pragma unroll
  for (int mi = 0; mi < 4; ++mi)
#pragma unroll
    for (int n = 0; n < 2; ++n)
#pragma unroll
      for (int r = 0; r < 4; ++r) o[mi][n][r] = 0.f;
  const int vrow_b = (lane & 7) + ((lane & 16) >> 1);
  const int vjo = (lane >> 3) & 1;
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    char* kvb = KVs + half * 32 * 1024;
    char* pb = Pb + half * 32 * 48;
#pragma unroll
    for (int ki = 0; ki < 2; ++ki) {
      uint32_t b[4];
      ldm4t(sa(pb + (ki * 16 + (lane & 15)) * 48 + (lane >> 4) * 16), b[0], b[1], b[2], b[3]);
      const int vrow = ki * 16 + vrow_b;
#pragma unroll
      for (int mi = 0; mi < 4; mi += 2) {
        uint32_t a0[4];
        const int j0 = (warp * 64 + mi * 16) >> 3;
        const int j1 = (warp * 64 + (mi + 1) * 16) >> 3;
        ldm4t(sa(kvb + vrow * 1024 + swz(vrow, j0 + vjo) * 16), a0[0], a0[1], a0[2], a0[3]);
        ldm4t(sa(kvb + vrow * 1024 + swz(vrow, j1 + vjo) * 16), qf[0][0], qf[0][1], qf[0][2],
              qf[0][3]);
        mma_bf16(o[mi][0], a0, b[0], b[1]);
        mma_bf16(o[mi][1], a0, b[2], b[3]);
        mma_bf16(o[mi + 1][0], qf[0], b[0], b[1]);
        mma_bf16(o[mi + 1][1], qf[0], b[2], b[3]);
      }
    }
  }

  cg::cluster_group clu = cg::this_cluster();
  const unsigned rank = clu.block_rank();
  __nv_bfloat16* Osh = (__nv_bfloat16*)Qs;
  constexpr int kDsmXor = 8;
  float2 norms[2];
#pragma unroll
  for (int n = 0; n < 2; ++n) {
    norms[n] = *(const float2*)(Norm + 8 * n + 2 * (lane & 3));
  }

#pragma unroll
  for (int mi = 0; mi < 4; ++mi) {
    const int dbase = warp * 64 + mi * 16;
#pragma unroll
    for (int n = 0; n < 2; ++n) {
      const int hh = 8 * n + (lane & 7);
      const int d8 = dbase + (((lane >> 3) & 1) * 8);
      const int dx = kDsmXor * (hh & 7);
      const float inv0 = norms[n].x, inv1 = norms[n].y;
      mul_f32x2(o[mi][n][0], o[mi][n][1], inv0, inv1);
      mul_f32x2(o[mi][n][2], o[mi][n][3], inv0, inv1);
      const __nv_bfloat162 r0 = __floats2bfloat162_rn(o[mi][n][0], o[mi][n][1]);
      const __nv_bfloat162 r1 = __floats2bfloat162_rn(o[mi][n][2], o[mi][n][3]);
      stm2t(sa(Osh + hh * DIM + (d8 ^ dx)), *reinterpret_cast<const uint32_t*>(&r0),
            *reinterpret_cast<const uint32_t*>(&r1));
    }
  }
  clu.sync();
  uint32_t* W2sh = (uint32_t*)(smem + F2_S);
  // Partition the 64 output warp-groups contiguously across the ten ranks.
  // Each rank only normalizes heads touched by its six or seven groups.
  const int merge_g0 = 6 * (int)rank + min((int)rank, 4);
  const int merge_ng = rank < 4 ? 7 : 6;
  const int merge_h0 = merge_g0 >> 2;
  const int merge_h1 = (merge_g0 + merge_ng - 1) >> 2;
  if (tid < HEADS && tid >= merge_h0 && tid <= merge_h1) {
    const int h = tid;
    const float sk = sink_log2(sinks[h]);
    float M = sk;
    float2 mlv[10];
#pragma unroll
    for (int p = 0; p < 10; ++p) {
      mlv[p] = ((float2*)clu.map_shared_rank(MLsh, p))[h];
      M = fmaxf(M, mlv[p].x);
    }
    float den;
    if constexpr (SPLIT_DEN) {
      float den0 = exp2f(sk - M), den1 = 0.f;
#pragma unroll
      for (int p = 0; p < 10; p += 2) {
        mlv[p].x = exp2f(mlv[p].x - M);
        mlv[p + 1].x = exp2f(mlv[p + 1].x - M);
        den0 += mlv[p].y * mlv[p].x;
        den1 += mlv[p + 1].y * mlv[p + 1].x;
      }
      den = den0 + den1;
    } else {
      den = exp2f(sk - M);
#pragma unroll
      for (int p = 0; p < 10; ++p) {
        mlv[p].x = exp2f(mlv[p].x - M);
        den += mlv[p].y * mlv[p].x;
      }
    }
    const float inv = 1.f / den;
    if constexpr (CONTIG_W) {
#pragma unroll
      for (int wp = 0; wp < 5; ++wp) {
        const float w0 = mlv[2 * wp].x * mlv[2 * wp].y * inv;
        const float w1 = mlv[2 * wp + 1].x * mlv[2 * wp + 1].y * inv;
        const uint32_t wb0 = __float_as_uint(w0);
        const uint32_t wb1 = __float_as_uint(w1);
        *(uint2*)(W2sh + h * 10 + 2 * wp) = make_uint2(*reinterpret_cast<const uint32_t*>(&wb0),
                                                       *reinterpret_cast<const uint32_t*>(&wb1));
      }
    } else {
#pragma unroll
      for (int p = 0; p < 10; ++p) {
        const float w = mlv[p].x * mlv[p].y * inv;
        const uint32_t wb = __float_as_uint(w);
        W2sh[p * HEADS + h] = *reinterpret_cast<const uint32_t*>(&wb);
      }
    }
  }
  const int c = (merge_g0 + warp) * 32 + lane;
  const bool valid_c = warp < merge_ng;
  int out_h = 0, out_d4 = 0;
  uint2 v[10];
  if (valid_c) {
    out_h = c >> 7;
    out_d4 = (c & 127) * 4;
    const int sd4 = out_d4 ^ (kDsmXor * (out_h & 7));
#pragma unroll
    for (int p = 0; p < 10; ++p)
      v[p] = *(const uint2*)clu.map_shared_rank(Osh + out_h * DIM + sd4, p);
  }
  __syncthreads();
  // The terminal cluster barrier follows all uses of the remote values.

  if (valid_c) {
    MergePair a0 = 0;
    MergePair a1 = 0;
    if constexpr (CONTIG_W) {
#pragma unroll
      for (int wp = 0; wp < 5; ++wp) {
        const uint2 w = *(const uint2*)(W2sh + out_h * 10 + 2 * wp);
        a0 = merge_f32x2(v[2 * wp].x, w.x, a0);
        a1 = merge_f32x2(v[2 * wp].y, w.x, a1);
        a0 = merge_f32x2(v[2 * wp + 1].x, w.y, a0);
        a1 = merge_f32x2(v[2 * wp + 1].y, w.y, a1);
      }
    } else {
#pragma unroll
      for (int p = 0; p < 10; ++p) {
        const uint32_t w = W2sh[p * HEADS + out_h];
        a0 = merge_f32x2(v[p].x, w, a0);
        a1 = merge_f32x2(v[p].y, w, a1);
      }
    }
    const uint2 pk = make_uint2(a0, a1);
    *(uint2*)(out + ((size_t)t * HEADS + out_h) * DIM + out_d4) = pk;
  }
  asm volatile(
      "barrier.cluster.arrive.relaxed.aligned;\n\t"
      "barrier.cluster.wait.aligned;\n" ::
          : "memory");
}

template <bool STAGE_WAIT = false>
__global__ __launch_bounds__(256, 1) void mla_fused2_h8_c10(
    const uint8_t* __restrict__ qp, const uint8_t* __restrict__ swap_,
    const uint8_t* __restrict__ ckvp, const int* __restrict__ sidx, const int* __restrict__ slen,
    const float* __restrict__ sinks, __nv_bfloat16* __restrict__ out) {
  extern __shared__ __align__(16) char smem[];
  char* Qs = smem + H8_Q;
  char* KVs = smem + H8_KV;
  float* Sb = (float*)(smem + H8_S);
  char* Pb = smem + H8_P;
  float2* MLsh = (float2*)(smem + H8_ML);
  int* Idx = (int*)(smem + H8_IDX);
  float* Norm = (float*)(smem + H8_NORM);

  const int t = blockIdx.y;
  const int sp = blockIdx.x;
  const int head_base = blockIdx.z * H8_HEADS;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int gcol0 = sp * 64;
  const int len = slen[t];
  const int* myidx = sidx + (size_t)t * WIDTH;

  {
    const uint8_t* src = qp + ((size_t)t * HEADS + head_base) * DIM * 2;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int ci = tid + r * 256;
      const int row = ci >> 6;
      const int ch = ci & 63;
      const uint32_t dst = sa(Qs + row * 1024 + swz(row, ch) * 16);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                   :
                   : "r"(dst), "l"(src + (size_t)ci * 16));
    }
    asm volatile("cp.async.commit_group;\n");
  }
  if (tid < 64) {
    const int gcol = gcol0 + tid;
    // inactive tail columns carry no offset-validity contract: mask BEFORE
    // staging so they never reach a gather (softmax length mask unchanged)
    const int v = myidx[gcol];
    Idx[tid] = (gcol < len && v >= 0) ? v : -1;
  }
  __syncthreads();

  auto issue_half = [&](int half) {
    const int base = gcol0 + half * 32;
    const uint8_t* pool = base < 128 ? swap_ : ckvp;
    char* dst = KVs + half * 32 * 1024;
    int iv[4];
#pragma unroll
    for (int r = 0; r < 4; ++r)
      iv[r] =
          Idx[half * 32 + (STAGE_WAIT ? ((warp >> 2) * 16 + (warp & 3) + r * 4) : (warp + r * 8))];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int row = STAGE_WAIT ? ((warp >> 2) * 16 + (warp & 3) + r * 4) : (warp + r * 8);
      const int off = iv[r];
      const int cpsz = off >= 0 ? 16 : 0;  // zero-fill invalid rows
      const uint8_t* src = pool + (size_t)(off >= 0 ? off : 0) * 1024 + lane * 16;
      const uint32_t d = sa(dst + row * 1024 + swz(row, lane) * 16);
      asm volatile(
          "cp.async.cg.shared.global [%0], [%1], 16, %2;\n\t"
          "cp.async.cg.shared.global [%0+512], [%1+512], 16, %2;\n"
          :
          : "r"(d), "l"(src), "r"(cpsz));
    }
    asm volatile("cp.async.commit_group;\n");
  };
  issue_half(0);
  issue_half(1);
  asm volatile("cp.async.wait_group 2;\n");
  __syncthreads();

  const int kk = STAGE_WAIT ? (warp & 3) : (warp >> 1);
  const int col0 = STAGE_WAIT ? ((warp >> 2) * 16) : ((warp & 1) * 16);
  const int qrow = (lane & 7) | ((lane & 16) >> 1);
  const int qjo = (lane >> 3) & 1;
  uint32_t qf[8][2];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int j0 = (kk * 128 + i * 16) >> 3;
    ldm2(sa(Qs + qrow * 1024 + swz(qrow, j0 + qjo) * 16), qf[i][0], qf[i][1]);
  }
  if constexpr (STAGE_WAIT)
    asm volatile("cp.async.wait_group 1;\n");
  else
    asm volatile("cp.async.wait_group 0;\n");
  if constexpr (STAGE_WAIT) {
    if (warp < 4)
      asm volatile("barrier.cta.sync 1, 128;\n" ::: "memory");
    else
      asm volatile("barrier.cta.sync 2, 128;\n" ::: "memory");
  } else {
    __syncthreads();
  }

  const int arow = col0 + (lane & 15);
  const int ajo = lane >> 4;
  const int srow = col0 + (lane >> 2);
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    if constexpr (STAGE_WAIT) {
      if (half == 1) {
        asm volatile("cp.async.wait_group 0;\n");
        if (warp < 4)
          asm volatile("barrier.cta.sync 1, 128;\n" ::: "memory");
        else
          asm volatile("barrier.cta.sync 2, 128;\n" ::: "memory");
      }
    }
    char* kvb = KVs + half * 32 * 1024;
    float s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int j0 = (kk * 128 + i * 16) >> 3;
      uint32_t a[4];
      ldm4(sa(kvb + arow * 1024 + swz(arow, j0 + ajo) * 16), a[0], a[1], a[2], a[3]);
      mma_bf16(s, a, qf[i][0], qf[i][1]);
    }
    float* sb = Sb + half * 4 * H8_SCORE_QSTRIDE + kk * H8_SCORE_QSTRIDE;
    const float2 s00 = make_float2(s[0], s[1]);
    const float2 s01 = make_float2(s[2], s[3]);
    const int hp = lane & 3;
    const int sx = (srow ^ (srow >> 2)) & 3;
    const int so0 = score_off_h8(srow, hp);
    const int so1 = score_off_h8(srow + 8, hp);
    *(float2*)(sb + so0) = s00;
    *(float2*)(sb + so1) = s01;
  }
  __syncthreads();

  const int h0 = 2 * warp;
  float mrun0 = -INFINITY;
  float mrun1 = -INFINITY;
  float lrun0 = 0.f;
  float lrun1 = 0.f;
  if (warp < H8_HEADS / 2) {
    float sv[2][2];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      float* sb = Sb + half * 4 * H8_SCORE_QSTRIDE;
      const int off = score_off_h8(lane, h0 >> 1);
      const float2 f = sum_qk4(sb + off, H8_SCORE_QSTRIDE);
      const bool ok = gcol0 + half * 32 + lane < len && Idx[half * 32 + lane] >= 0;
      sv[half][0] = ok ? f.x * QKSCALE : -INFINITY;
      sv[half][1] = ok ? f.y * QKSCALE : -INFINITY;
    }
    mrun0 = warp_max_f32(fmaxf(sv[0][0], sv[1][0]));
    mrun1 = warp_max_f32(fmaxf(sv[0][1], sv[1][1]));
    const float g0 = mrun0 == -INFINITY ? 0.f : mrun0;
    const float g1 = mrun1 == -INFINITY ? 0.f : mrun1;
    float pv[2][2];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      pv[half][0] = exp2f(sv[half][0] - g0);
      pv[half][1] = exp2f(sv[half][1] - g1);
      const __nv_bfloat162 pk = __floats2bfloat162_rn(pv[half][0], pv[half][1]);
      const float2 used_p = __bfloat1622float2(pk);
      pv[half][0] = used_p.x;
      pv[half][1] = used_p.y;
      *(uint32_t*)(Pb + half * 32 * H8_PSTRIDE + lane * H8_PSTRIDE + h0 * 2) =
          *(const uint32_t*)&pk;
    }
    constexpr float kProbScale = 33554432.f;
    constexpr float kInvProbScale = 1.f / kProbScale;
    const uint32_t fp0 = __float2uint_rn((pv[0][0] + pv[1][0]) * kProbScale);
    const uint32_t fp1 = __float2uint_rn((pv[0][1] + pv[1][1]) * kProbScale);
    lrun0 = float(warp_add_u32<32>(fp0)) * kInvProbScale;
    lrun1 = float(warp_add_u32<32>(fp1)) * kInvProbScale;
  }
  if (warp < H8_HEADS / 2 && lane == 0) {
    Norm[h0] = lrun0 > 0.f ? 1.f / lrun0 : 0.f;
    Norm[h0 + 1] = lrun1 > 0.f ? 1.f / lrun1 : 0.f;
    MLsh[h0] = make_float2(mrun0, lrun0);
    MLsh[h0 + 1] = make_float2(mrun1, lrun1);
  }
  __syncthreads();

  float o[4][4];
#pragma unroll
  for (int mi = 0; mi < 4; ++mi)
#pragma unroll
    for (int r = 0; r < 4; ++r) o[mi][r] = 0.f;
  const int vrow_b = (lane & 7) + ((lane & 16) >> 1);
  const int vjo = (lane >> 3) & 1;
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    char* kvb = KVs + half * 32 * 1024;
    char* pb = Pb + half * 32 * H8_PSTRIDE;
#pragma unroll
    for (int ki = 0; ki < 2; ++ki) {
      uint32_t b[2];
      ldm2t(sa(pb + (ki * 16 + (lane & 15)) * H8_PSTRIDE), b[0], b[1]);
#pragma unroll
      for (int mi = 0; mi < 4; mi += 2) {
        uint32_t a0[4], a1[4];
        const int j0 = (warp * 64 + mi * 16) >> 3;
        const int j1 = (warp * 64 + (mi + 1) * 16) >> 3;
        const int vrow = ki * 16 + vrow_b;
        ldm4t(sa(kvb + vrow * 1024 + swz(vrow, j0 + vjo) * 16), a0[0], a0[1], a0[2], a0[3]);
        ldm4t(sa(kvb + vrow * 1024 + swz(vrow, j1 + vjo) * 16), a1[0], a1[1], a1[2], a1[3]);
        mma_bf16(o[mi], a0, b[0], b[1]);
        mma_bf16(o[mi + 1], a1, b[0], b[1]);
      }
    }
  }

  cg::cluster_group clu = cg::this_cluster();
  const unsigned rank = clu.block_rank();
  __nv_bfloat16* Osh = (__nv_bfloat16*)Qs;
  constexpr int kDsmXor = 8;
  const float2 norm = *(const float2*)(Norm + 2 * (lane & 3));
  const float inv0 = norm.x, inv1 = norm.y;

#pragma unroll
  for (int mi = 0; mi < 4; ++mi) {
    const int dbase = warp * 64 + mi * 16;
    const int hh = lane & 7;
    const int d8 = dbase + (((lane >> 3) & 1) * 8);
    const int dx = kDsmXor * hh;
    mul_f32x2(o[mi][0], o[mi][1], inv0, inv1);
    mul_f32x2(o[mi][2], o[mi][3], inv0, inv1);
    const __nv_bfloat162 r0 = __floats2bfloat162_rn(o[mi][0], o[mi][1]);
    const __nv_bfloat162 r1 = __floats2bfloat162_rn(o[mi][2], o[mi][3]);
    stm2t(sa(Osh + hh * DIM + (d8 ^ dx)), *reinterpret_cast<const uint32_t*>(&r0),
          *reinterpret_cast<const uint32_t*>(&r1));
  }
  clu.sync();

  uint32_t* W2sh = (uint32_t*)(smem + H8_S);
  constexpr int kH8WStride = 12;
  const unsigned weight_mask = rank < 2   ? 0xa5u
                               : rank < 4 ? 0x29u
                               : rank < 6 ? 0x4au
                               : rank < 8 ? 0x52u
                                          : 0x94u;
  // Warp 7 owns no output chunks.  Let it normalize while warps 0-3 issue
  // their remote DSM reads, then join at the existing CTA barrier.
  if (warp == 7 && lane < H8_HEADS && ((weight_mask >> lane) & 1u)) {
    const int h = lane;
    const float sk = sink_log2(sinks[head_base + h]);
    float M = sk;
    float2 mlv[10];
#pragma unroll
    for (int p = 0; p < 10; ++p) {
      mlv[p] = ((float2*)clu.map_shared_rank(MLsh, p))[h];
      M = fmaxf(M, mlv[p].x);
    }
    float den = exp2f(sk - M);
#pragma unroll
    for (int p = 0; p < 10; ++p) {
      mlv[p].x = exp2f(mlv[p].x - M);
      den += mlv[p].y * mlv[p].x;
    }
    const float inv = 1.f / den;
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      const int p = 4 * q;
      const float w0 = mlv[p].x * mlv[p].y * inv;
      const float w1 = mlv[p + 1].x * mlv[p + 1].y * inv;
      const float w2 = mlv[p + 2].x * mlv[p + 2].y * inv;
      const float w3 = mlv[p + 3].x * mlv[p + 3].y * inv;
      const uint32_t b0 = __float_as_uint(w0);
      const uint32_t b1 = __float_as_uint(w1);
      const uint32_t b2 = __float_as_uint(w2);
      const uint32_t b3 = __float_as_uint(w3);
      asm volatile(
          "st.shared::cta.v4.b32 [%0], {%1,%2,%3,%4};\n" ::"r"(sa(W2sh + h * kH8WStride + p)),
          "r"(*reinterpret_cast<const uint32_t*>(&b0)),
          "r"(*reinterpret_cast<const uint32_t*>(&b1)),
          "r"(*reinterpret_cast<const uint32_t*>(&b2)), "r"(*reinterpret_cast<const uint32_t*>(&b3))
          : "memory");
    }
    const float w8 = mlv[8].x * mlv[8].y * inv;
    const float w9 = mlv[9].x * mlv[9].y * inv;
    const uint32_t b8 = __float_as_uint(w8);
    const uint32_t b9 = __float_as_uint(w9);
    *(uint2*)(W2sh + h * kH8WStride + 8) = make_uint2(*reinterpret_cast<const uint32_t*>(&b8),
                                                      *reinterpret_cast<const uint32_t*>(&b9));
  }

  const int c = (warp * 10 + (int)rank) * 32 + lane;
  const bool valid_c = c < H8_HEADS * DIM / 4;
  int out_h = 0;
  int out_d4 = 0;
  uint2 v[10];
  if (valid_c) {
    out_h = c >> 7;
    out_d4 = (c & 127) * 4;
    const int sd4 = out_d4 ^ (kDsmXor * out_h);
#pragma unroll
    for (int p = 0; p < 10; ++p)
      v[p] = *(const uint2*)clu.map_shared_rank(Osh + out_h * DIM + sd4, p);
  }
  __syncthreads();
  // The terminal cluster barrier follows all uses of the remote values.
  if (valid_c) {
    MergePair a0 = 0;
    MergePair a1 = 0;
    uint32_t w0, w1, w2, w3;
    asm volatile("ld.shared::cta.v4.b32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(w0), "=r"(w1), "=r"(w2), "=r"(w3)
                 : "r"(sa(W2sh + out_h * kH8WStride))
                 : "memory");
    a0 = merge_f32x2(v[0].x, w0, a0);
    a1 = merge_f32x2(v[0].y, w0, a1);
    a0 = merge_f32x2(v[1].x, w1, a0);
    a1 = merge_f32x2(v[1].y, w1, a1);
    a0 = merge_f32x2(v[2].x, w2, a0);
    a1 = merge_f32x2(v[2].y, w2, a1);
    a0 = merge_f32x2(v[3].x, w3, a0);
    a1 = merge_f32x2(v[3].y, w3, a1);
    asm volatile("ld.shared::cta.v4.b32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(w0), "=r"(w1), "=r"(w2), "=r"(w3)
                 : "r"(sa(W2sh + out_h * kH8WStride + 4))
                 : "memory");
    a0 = merge_f32x2(v[4].x, w0, a0);
    a1 = merge_f32x2(v[4].y, w0, a1);
    a0 = merge_f32x2(v[5].x, w1, a0);
    a1 = merge_f32x2(v[5].y, w1, a1);
    a0 = merge_f32x2(v[6].x, w2, a0);
    a1 = merge_f32x2(v[6].y, w2, a1);
    a0 = merge_f32x2(v[7].x, w3, a0);
    a1 = merge_f32x2(v[7].y, w3, a1);
    const uint2 wt = *(const uint2*)(W2sh + out_h * kH8WStride + 8);
    a0 = merge_f32x2(v[8].x, wt.x, a0);
    a1 = merge_f32x2(v[8].y, wt.x, a1);
    a0 = merge_f32x2(v[9].x, wt.y, a0);
    a1 = merge_f32x2(v[9].y, wt.y, a1);
    const uint2 pk = make_uint2(a0, a1);
    *(uint2*)(out + ((size_t)t * HEADS + head_base + out_h) * DIM + out_d4) = pk;
  }
  asm volatile(
      "barrier.cluster.arrive.relaxed.aligned;\n\t"
      "barrier.cluster.wait.aligned;\n" ::
          : "memory");
}

// Blackwell tensor-memory specialization for the same 64-key/rank regime.
// QK is one M64xN16xK512 UMMA and PV is four M128xN16xK64 UMMAs.
#define TC_Q 0
#define TC_KV (TC_Q + 16 * 1024)
#define TC_S (TC_KV + 64 * 1024)
#define TC_P (TC_S + 64 * HEADS * 2)
#define TC_PLEG (TC_P + HEADS * 64 * 2)
#define TC_ML (TC_PLEG + 64 * 48)
#define TC_IDX (TC_ML + HEADS * 8)
#define TC_BYTES (TC_IDX + 64 * 4 + 1023)

__global__ __launch_bounds__(256, 2) void mla_tcgen_c10(
    const uint8_t* __restrict__ qp, const uint8_t* __restrict__ swap_,
    const uint8_t* __restrict__ ckvp, const int* __restrict__ sidx, const int* __restrict__ slen,
    const float* __restrict__ sinks, __nv_bfloat16* __restrict__ out) {
  extern __shared__ __align__(1024) char smem[];
  __shared__ uint64_t tcbar[2];
  __shared__ uint32_t tslot[1];
  char* tcbase = smem + ((-int(sa(smem))) & 1023);
  char* Qs = tcbase + TC_Q;
  char* KVs = tcbase + TC_KV;
  __nv_bfloat16* Sb = (__nv_bfloat16*)(tcbase + TC_S);
  __nv_bfloat16* Pb = (__nv_bfloat16*)(tcbase + TC_P);
  char* Pleg = tcbase + TC_PLEG;
  float2* MLsh = (float2*)(tcbase + TC_ML);
  int* Idx = (int*)(tcbase + TC_IDX);

  const int t = blockIdx.y;
  const int sp = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int gcol0 = sp * 64;
  const int len = slen[t];
  const int* myidx = sidx + (size_t)t * WIDTH;

  // Q is staged as eight independent [16,64] 128B-swizzled slabs.
  const uint8_t* qsrc = qp + (size_t)t * (HEADS * DIM * 2);
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const int ci = tid + r * 256;
    const int row = ci >> 6;
    const int ch = ci & 63;
    char* dst = Qs + (ch >> 3) * (HEADS * 128) + row * 128 + (((ch & 7) ^ (row & 7)) * 16);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sa(dst)),
                 "l"(qsrc + (size_t)ci * 16));
  }
  asm volatile("cp.async.commit_group;\n");
  if (tid < 64) {
    const int gcol = gcol0 + tid;
    const int v = myidx[gcol];
    Idx[tid] = (gcol < len && v >= 0) ? v : -1;
  }
  __syncthreads();

  // KV is staged as eight [64,64] slabs.  Keep the two copy groups small
  // enough for cp.async while TMEM allocation and barrier setup overlap them.
  const uint8_t* pool = gcol0 < 128 ? swap_ : ckvp;
#pragma unroll
  for (int pass = 0; pass < 2; ++pass) {
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      const int rr = r + pass * 8;
      const int grow = (tid >> 6) + rr * 4;
      const int ch = tid & 63;
      const int off = Idx[grow];
      const int cpsz = off >= 0 ? 16 : 0;  // zero-fill invalid rows
      const uint8_t* src = pool + (size_t)(off >= 0 ? off : 0) * 1024 + ch * 16;
      char* dst = KVs + (ch >> 3) * (64 * 128) + grow * 128 + (((ch & 7) ^ (grow & 7)) * 16);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa(dst)), "l"(src),
                   "r"(cpsz));
    }
    asm volatile("cp.async.commit_group;\n");
  }
  if (tid == 0) {
    ws_mbar_init(tcbar);
    ws_mbar_init(tcbar + 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  if (warp == 1) {
    tc_alloc64(tslot);
    tc_relinquish();
  }
  asm volatile("cp.async.wait_group 0;\n");
  __syncthreads();
  // cp.async completion plus the CTA barrier publishes the gathered operands
  // to generic shared memory.  UMMA reads them through the async proxy.
  tc_smem_async_fence();
  const uint32_t taddr = tslot[0];

  constexpr bool kTcQk = true;
  if constexpr (kTcQk) {
    constexpr uint32_t QK_DESC = 0x04040490u;
    tc_fence_after_sync();
    if (tid == 0) {
#pragma unroll
      for (int db = 0; db < 8; ++db) {
#pragma unroll
        for (int ki = 0; ki < 4; ++ki) {
          const uint64_t ad = tc_desc128(KVs + db * 8192 + ki * 32, 16, 1024);
          const uint64_t bd = tc_desc128(Qs + db * 2048 + ki * 32, 16, 1024);
          tc_mma(taddr, ad, bd, QK_DESC, db != 0 || ki != 0);
        }
      }
      tc_commit(tcbar);
    }
    ws_mbar_wait(tcbar, 0);
    tc_fence_after_sync();

    if (warp < 4) {
      uint32_t r[8];
      tc_ld_score(taddr + ((warp * 32) << 16), r);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int key = warp * 16 + 8 * ((j >> 1) & 1) + (lane >> 2);
        const int head = 8 * (j >> 2) + 2 * (lane & 3) + (j & 1);
        Sb[head * 64 + key] = __float2bfloat16(__uint_as_float(r[j]));
      }
      tc_fence_before_sync();
    }
    __syncthreads();
  } else {
    for (int c = tid; c < 64 * HEADS; c += 256) {
      const int key = c >> 4;
      const int h = c & 15;
      float acc = 0.f;
#pragma unroll
      for (int d = 0; d < DIM; ++d) {
        const int ch = d >> 3;
        const char* kp =
            KVs + (ch >> 3) * (64 * 128) + key * 128 + (((ch & 7) ^ (key & 7)) * 16) + (d & 7) * 2;
        const char* qq =
            Qs + (ch >> 3) * (HEADS * 128) + h * 128 + (((ch & 7) ^ (h & 7)) * 16) + (d & 7) * 2;
        acc = fmaf(__bfloat162float(*(const __nv_bfloat16*)kp),
                   __bfloat162float(*(const __nv_bfloat16*)qq), acc);
      }
      Sb[h * 64 + key] = __float2bfloat16(acc);
    }
    __syncthreads();
  }

  const int h0 = 2 * warp;
  float mrun0, mrun1, lrun0, lrun1;
  {
    float sv[2][2];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int key = lane + half * 32;
      const bool ok = Idx[key] >= 0;
      sv[half][0] = ok ? __bfloat162float(Sb[h0 * 64 + key]) * QKSCALE : -INFINITY;
      sv[half][1] = ok ? __bfloat162float(Sb[(h0 + 1) * 64 + key]) * QKSCALE : -INFINITY;
    }
    mrun0 = warp_max_f32(fmaxf(sv[0][0], sv[1][0]));
    mrun1 = warp_max_f32(fmaxf(sv[0][1], sv[1][1]));
    const float g0 = mrun0 == -INFINITY ? 0.f : mrun0;
    const float g1 = mrun1 == -INFINITY ? 0.f : mrun1;
    float pv[2][2];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int key = lane + half * 32;
      pv[half][0] = exp2f(sv[half][0] - g0);
      pv[half][1] = exp2f(sv[half][1] - g1);
      Pb[tc_swz_hk(h0, key)] = __float2bfloat16(pv[half][0]);
      Pb[tc_swz_hk(h0 + 1, key)] = __float2bfloat16(pv[half][1]);
      const __nv_bfloat162 pk = __floats2bfloat162_rn(pv[half][0], pv[half][1]);
      *(uint32_t*)(Pleg + key * 48 + h0 * 2) = *reinterpret_cast<const uint32_t*>(&pk);
    }
    constexpr float kProbScale = 33554432.f;
    constexpr float kInvProbScale = 1.f / kProbScale;
    const uint32_t fp0 =
        __float2uint_rn(pv[0][0] * kProbScale) + __float2uint_rn(pv[1][0] * kProbScale);
    const uint32_t fp1 =
        __float2uint_rn(pv[0][1] * kProbScale) + __float2uint_rn(pv[1][1] * kProbScale);
    lrun0 = float(warp_add_u32<32>(fp0)) * kInvProbScale;
    lrun1 = float(warp_add_u32<32>(fp1)) * kInvProbScale;
  }
  if (lane == 0) {
    MLsh[h0] = make_float2(mrun0, lrun0);
    MLsh[h0 + 1] = make_float2(mrun1, lrun1);
  }
  // Publish generic-proxy probability stores before the CTA rendezvous that
  // lets warp 0 issue the PV UMMA through the async shared-memory proxy.
  tc_smem_async_fence();
  __syncthreads();

  __nv_bfloat16* Osh = (__nv_bfloat16*)Qs;
  constexpr int kDsmXor = 8;
  constexpr bool kTcPv = true;
  if constexpr (kTcPv) {
    constexpr uint32_t PV_DESC = 0x08048490u;
    tc_fence_after_sync();
    if (tid == 0) {
#pragma unroll
      for (int dc = 0; dc < 4; ++dc) {
#pragma unroll
        for (int ki = 0; ki < 4; ++ki) {
          const uint64_t ad = tc_desc128(KVs + dc * 16384 + ki * 2048, 8192, 1024);
          const uint64_t bd = tc_desc128((char*)Pb + ki * 32, 16, 1024);
          tc_mma(taddr + dc * 16, ad, bd, PV_DESC, ki != 0);
        }
      }
      tc_commit(tcbar + 1);
    }
    ws_mbar_wait(tcbar + 1, 0);
    tc_fence_after_sync();
    const int row_group = warp & 3;
#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
      const int dc = (warp >> 2) + 2 * pass;
      const int d = dc * 128 + row_group * 32 + lane;
#pragma unroll
      for (int nh = 0; nh < 2; ++nh) {
        uint32_t r[8];
        tc_ld_pv(taddr + dc * 16 + ((row_group * 32) << 16) + nh * 8, r);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int h = nh * 8 + j;
          Osh[h * DIM + (d ^ (kDsmXor * (h & 7)))] = __float2bfloat16(__uint_as_float(r[j]));
        }
      }
    }
    tc_fence_before_sync();
    __syncthreads();
  } else {
    constexpr bool kScalarPv = true;
    if constexpr (kScalarPv) {
      for (int c = tid; c < HEADS * DIM; c += 256) {
        const int h = c >> 9;
        const int d = c & 511;
        float acc = 0.f;
#pragma unroll
        for (int key = 0; key < 64; ++key) {
          const int ch = d >> 3;
          const char* vp = KVs + (ch >> 3) * (64 * 128) + key * 128 +
                           (((ch & 7) ^ (key & 7)) * 16) + (d & 7) * 2;
          const __nv_bfloat16 vv = *(const __nv_bfloat16*)vp;
          acc = fmaf(__bfloat162float(Pb[tc_swz_hk(h, key)]), __bfloat162float(vv), acc);
        }
        Osh[h * DIM + (d ^ (kDsmXor * (h & 7)))] = __float2bfloat16(acc);
      }
      __syncthreads();
    } else {
      float o[4][2][4];
#pragma unroll
      for (int mi = 0; mi < 4; ++mi)
#pragma unroll
        for (int n = 0; n < 2; ++n)
#pragma unroll
          for (int r = 0; r < 4; ++r) o[mi][n][r] = 0.f;
      const int vrow_b = (lane & 7) + ((lane & 16) >> 1);
      const int vjo = (lane >> 3) & 1;
#pragma unroll
      for (int ki = 0; ki < 4; ++ki) {
        uint32_t b[4];
        ldm4t(sa(Pleg + (ki * 16 + (lane & 15)) * 48 + (lane >> 4) * 16), b[0], b[1], b[2], b[3]);
        const int vrow = ki * 16 + vrow_b;
#pragma unroll
        for (int mi = 0; mi < 4; mi += 2) {
          uint32_t a0[4], a1[4];
          const int ch0 = (warp * 64 + mi * 16) >> 3;
          const int ch1 = (warp * 64 + (mi + 1) * 16) >> 3;
          const int c0 = ch0 + vjo;
          const int c1 = ch1 + vjo;
          char* p0 = KVs + (c0 >> 3) * (64 * 128) + vrow * 128 + (((c0 & 7) ^ (vrow & 7)) * 16);
          char* p1 = KVs + (c1 >> 3) * (64 * 128) + vrow * 128 + (((c1 & 7) ^ (vrow & 7)) * 16);
          ldm4t(sa(p0), a0[0], a0[1], a0[2], a0[3]);
          ldm4t(sa(p1), a1[0], a1[1], a1[2], a1[3]);
          mma_bf16(o[mi][0], a0, b[0], b[1]);
          mma_bf16(o[mi][1], a0, b[2], b[3]);
          mma_bf16(o[mi + 1][0], a1, b[0], b[1]);
          mma_bf16(o[mi + 1][1], a1, b[2], b[3]);
        }
      }
#pragma unroll
      for (int mi = 0; mi < 4; ++mi) {
        const int dbase = warp * 64 + mi * 16;
#pragma unroll
        for (int n = 0; n < 2; ++n) {
          const int hh = 8 * n + (lane & 7);
          const int d8 = dbase + (((lane >> 3) & 1) * 8);
          const int dx = kDsmXor * (hh & 7);
          const __nv_bfloat162 r0 = __floats2bfloat162_rn(o[mi][n][0], o[mi][n][1]);
          const __nv_bfloat162 r1 = __floats2bfloat162_rn(o[mi][n][2], o[mi][n][3]);
          stm2t(sa(Osh + hh * DIM + (d8 ^ dx)), *reinterpret_cast<const uint32_t*>(&r0),
                *reinterpret_cast<const uint32_t*>(&r1));
        }
      }
      __syncthreads();
    }
  }
  if (warp == 1) tc_dealloc64(taddr);
  __syncthreads();

  cg::cluster_group clu = cg::this_cluster();
  const unsigned rank = clu.block_rank();
  clu.sync();
  uint32_t* W2sh = (uint32_t*)(smem + TC_S);
  const int merge_g0 = 6 * (int)rank + min((int)rank, 4);
  const int merge_ng = rank < 4 ? 7 : 6;
  const int merge_h0 = merge_g0 >> 2;
  const int merge_h1 = (merge_g0 + merge_ng - 1) >> 2;
  if (tid < HEADS && tid >= merge_h0 && tid <= merge_h1) {
    const int h = tid;
    const float sk = sinks[h] * LOG2E;
    float M = sk;
    float2 mlv[10];
#pragma unroll
    for (int p = 0; p < 10; ++p) {
      mlv[p] = ((float2*)clu.map_shared_rank(MLsh, p))[h];
      M = fmaxf(M, mlv[p].x);
    }
    float den = exp2f(sk - M);
#pragma unroll
    for (int p = 0; p < 10; ++p) {
      mlv[p].x = exp2f(mlv[p].x - M);
      den += mlv[p].y * mlv[p].x;
    }
    const float inv = 1.f / den;
#pragma unroll
    for (int p = 0; p < 10; ++p) {
      const __nv_bfloat162 wb = __floats2bfloat162_rn(mlv[p].x * inv, mlv[p].x * inv);
      W2sh[p * HEADS + h] = *reinterpret_cast<const uint32_t*>(&wb);
    }
  }
  const int c = (merge_g0 + warp) * 32 + lane;
  const bool valid_c = warp < merge_ng;
  int out_h = 0, out_d4 = 0;
  uint2 v[10];
  if (valid_c) {
    out_h = c >> 7;
    out_d4 = (c & 127) * 4;
    const int sd4 = out_d4 ^ (kDsmXor * (out_h & 7));
#pragma unroll
    for (int p = 0; p < 10; ++p)
      v[p] = *(const uint2*)clu.map_shared_rank(Osh + out_h * DIM + sd4, p);
  }
  __syncthreads();
  if (valid_c) {
    uint32_t a0 = 0, a1 = 0;
#pragma unroll
    for (int p = 0; p < 10; ++p) {
      const uint32_t w = W2sh[p * HEADS + out_h];
      a0 = fma_bf16x2(v[p].x, w, a0);
      a1 = fma_bf16x2(v[p].y, w, a1);
    }
    *(uint2*)(out + ((size_t)t * HEADS + out_h) * DIM + out_d4) = make_uint2(a0, a1);
  }
  asm volatile(
      "barrier.cluster.arrive.relaxed.aligned;\n\t"
      "barrier.cluster.wait.aligned;\n" ::
          : "memory");
}

// ---------------------------------------------------------------------------
// T=78 (B=13) specialization.  The TILE16 kernel pays one softmax, one 64-float
// output rescale and three barriers per 16 keys; this one uses 32-key tiles with
// only four warps, halving all three per key while keeping 3 CTAs/SM.  Q is
// staged in KV slot 1 and the epilogue DSM buffer reuses KV slot 0, so dropping
// the dedicated 16 KiB Q region is what buys the third CTA.
#define B13_SCORE_QSTRIDE (32 * HEADS)
__device__ __forceinline__ int score_off_b13(int row, int hp) {
  return row * HEADS + 2 * (hp ^ ((row ^ (row >> 2)) & 7));
}
#define B13_KV0 0
#define B13_KV1 32768
#define B13_S 65536
#define B13_P (B13_S + 8192)
#define B13_ALP (B13_P + 1536)
#define B13_ML (B13_ALP + 64)
#define B13_IDX (B13_ML + 128)
#define B13_NORM (B13_IDX + 512)
#define B13_BYTES (B13_NORM + 64)

template <int CSZ>
__global__ __launch_bounds__(128, 3) void mla_b13(
    const uint8_t* __restrict__ qp, const uint8_t* __restrict__ swap_,
    const uint8_t* __restrict__ ckvp, const int* __restrict__ sidx, const int* __restrict__ slen,
    const float* __restrict__ sinks, __nv_bfloat16* __restrict__ out) {
  extern __shared__ __align__(16) char smem[];
  char* KVs = smem;
  float* Sb = (float*)(smem + B13_S);
  char* Pb = smem + B13_P;
  float* Alp = (float*)(smem + B13_ALP);
  int* Idx = (int*)(smem + B13_IDX);
  float* Norm = (float*)(smem + B13_NORM);

  constexpr int TILE = 32;
  constexpr int NTHREAD = 128;
  constexpr int NWARP = 4;
  constexpr int MIO = 8;
  constexpr int TOTAL_TILES = WIDTH / TILE;
  constexpr int TPS = (TOTAL_TILES + CSZ - 1) / CSZ;
  constexpr int SLOT = 32 * 1024;

  const int t = blockIdx.y;
  const int sp = blockIdx.x;
  const int tile0 = sp * TPS;
  int ntile = TPS;
  if constexpr (TOTAL_TILES % CSZ != 0) {
    ntile = TOTAL_TILES - tile0;
    if (ntile > TPS) ntile = TPS;
  }
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int len = slen[t];
  const int* myidx = sidx + (size_t)t * WIDTH;

  auto issue_tile = [&](int tl) {
    const int base = (tile0 + tl) * TILE;
    char* dst = KVs + (tl & 1) * SLOT;
    const uint8_t* pool = (base < 128) ? swap_ : ckvp;
    int iv[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) iv[r] = Idx[tl * TILE + warp + r * NWARP];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      const int row = warp + r * NWARP;
      const int off = iv[r];
      const int cpsz = off >= 0 ? 16 : 0;  // zero-fill invalid rows
      const uint8_t* src = pool + (size_t)(off >= 0 ? off : 0) * 1024 + lane * 16;
      const uint32_t d = sa(dst + row * 1024 + swz(row, lane) * 16);
      asm volatile(
          "cp.async.cg.shared.global [%0], [%1], 16, %2;\n\t"
          "cp.async.cg.shared.global [%0+512], [%1+512], 16, %2;\n" ::"r"(d),
          "l"(src), "r"(cpsz));
    }
  };

  {  // Q -> KV slot 1 (group 0)
    const uint8_t* src = qp + (size_t)t * (HEADS * DIM * 2);
    char* Qs = KVs + SLOT;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      const int ci = tid + r * NTHREAD;
      const int row = ci >> 6, ch = ci & 63;
      const uint32_t d = sa(Qs + row * 1024 + swz(row, ch) * 16);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d),
                   "l"(src + (size_t)ci * 16));
    }
    asm volatile("cp.async.commit_group;\n");
  }
  {
    const int ncols = ntile * TILE;
    for (int i = tid; i < ncols; i += NTHREAD) {
      const int gcol = tile0 * TILE + i;
      // mask inactive tail columns and holes before staging (see sibling
      // kernels): they carry no offset-validity contract
      const int v = myidx[gcol];
      Idx[i] = (gcol < len && v >= 0) ? v : -1;
    }
  }
  __syncthreads();
  issue_tile(0);
  asm volatile(
      "cp.async.commit_group;\n"
      "cp.async.wait_group 1;\n" ::
          : "memory");
  __syncthreads();

  const int kk = warp;
  const int qrow = (lane & 7) | ((lane & 16) >> 1);
  const int qjo = (lane >> 3) & 1;
  uint32_t qf[8][4];
  {
    const char* Qs = KVs + SLOT;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int j0 = (kk * 128 + i * 16) >> 3;
      ldm4(sa(Qs + qrow * 1024 + swz(qrow, j0 + qjo) * 16), qf[i][0], qf[i][1], qf[i][2], qf[i][3]);
    }
  }
  __syncthreads();  // Q consumed; slot 1 free for tile 1
  if (ntile > 1) {
    issue_tile(1);
    asm volatile("cp.async.commit_group;\n");
  }

  const int arow = lane & 15;
  const int ajo = lane >> 4;
  const int vrow_b = (lane & 7) + ((lane & 16) >> 1);
  const int vjo = (lane >> 3) & 1;
  const int srow = lane >> 2;
  const int shd = 2 * (lane & 3);
  const int hpair0 = 2 * warp;  // head pairs 2w and 2w+1

  float mrun[2][2], lrun[2][2];
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    mrun[u][0] = -INFINITY;
    mrun[u][1] = -INFINITY;
    lrun[u][0] = 0.f;
    lrun[u][1] = 0.f;
  }
  float o[MIO][2][4];
#pragma unroll
  for (int i = 0; i < MIO; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) o[i][j][k] = 0.f;

  for (int tl = 0; tl < ntile; ++tl) {
    // Release the ping-pong slot before overwriting it.  Issuing the next
    // gather after this rendezvous removes the former end-of-body barrier:
    // one gather/compute iteration now needs three CTA barriers, not four.
    if (tl == 0)
      asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    else
      asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
    if (tl != 0 && tl + 1 < ntile) {
      issue_tile(tl + 1);
      asm volatile("cp.async.commit_group;\n");
    }
    // Keep the ping-pong slot as a 32-bit shared address.  QK and PV reuse it
    // throughout the tile, avoiding repeated generic-to-shared conversions.
    const uint32_t kvs = sa(KVs + (tl & 1) * SLOT);

    // ---- QK: two 16-key m-tiles, four independent accumulator chains
    float s[2][2][4];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int k = 0; k < 4; ++k) s[m][j][k] = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int j0 = (kk * 128 + i * 16) >> 3;
      uint32_t a0[4], a1[4];
      const uint32_t qka = kvs + (uint32_t)(arow * 1024 + swz(arow, j0 + ajo) * 16);
      ldm4(qka, a0[0], a0[1], a0[2], a0[3]);
      ldm4(qka + 16 * 1024, a1[0], a1[1], a1[2], a1[3]);
      mma_bf16(s[0][0], a0, qf[i][0], qf[i][1]);
      mma_bf16(s[0][1], a0, qf[i][2], qf[i][3]);
      mma_bf16(s[1][0], a1, qf[i][0], qf[i][1]);
      mma_bf16(s[1][1], a1, qf[i][2], qf[i][3]);
    }
    {
      float* sb = Sb + kk * B13_SCORE_QSTRIDE;
      const int hp = lane & 3;
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        const int sr = m * 16 + srow;
        const int sx = (sr ^ (sr >> 2)) & 7;
        const int so0 = sr * HEADS + 2 * (hp ^ sx);
        const int so1 = (sr + 8) * HEADS + 2 * (hp ^ sx ^ 2);
        const float2 s00 = make_float2(s[m][0][0], s[m][0][1]);
        const float2 s01 = make_float2(s[m][0][2], s[m][0][3]);
        const float2 s10 = make_float2(s[m][1][0], s[m][1][1]);
        const float2 s11 = make_float2(s[m][1][2], s[m][1][3]);
        *(float2*)(sb + so0) = s00;
        *(float2*)(sb + so1) = s01;
        *(float2*)(sb + (so0 ^ 8)) = s10;
        *(float2*)(sb + (so1 ^ 8)) = s11;
      }
    }
    __syncthreads();

    // ---- softmax: each warp owns four heads, handled as two head pairs
    const bool ok = (tile0 + tl) * TILE + lane < len && Idx[tl * TILE + lane] >= 0;
    uint32_t pkv[2];
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int hpr = hpair0 + u;
      const int soff = score_off_b13(lane, hpr);
      const float2 acc = sum_qk4(Sb + soff, B13_SCORE_QSTRIDE);
      const float x0 = ok ? acc.x * QKSCALE : -INFINITY;
      const float x1 = ok ? acc.y * QKSCALE : -INFINITY;
      const float mn0 = fmaxf(mrun[u][0], warp_max_f32(x0));
      const float mn1 = fmaxf(mrun[u][1], warp_max_f32(x1));
      const float g0 = mn0 == -INFINITY ? 0.f : mn0;
      const float g1 = mn1 == -INFINITY ? 0.f : mn1;
      const float a0 = exp2f(mrun[u][0] - g0);
      const float a1 = exp2f(mrun[u][1] - g1);
      mrun[u][0] = mn0;
      mrun[u][1] = mn1;
      if (lane == 0) {
        Alp[2 * hpr] = a0;
        Alp[2 * hpr + 1] = a1;
      }
      const uint32_t pk = ex2_bf16x2(x0 - g0, x1 - g1);
      pkv[u] = pk;
      const float p0 = __uint_as_float(pk << 16);
      const float p1 = __uint_as_float(pk & 0xffff0000u);
      constexpr float kProbScale = 2097152.f;
      constexpr float kInvProbScale = 1.0f / kProbScale;
      const uint32_t ps0 = warp_add_u32<32>(__float2uint_rn(p0 * kProbScale));
      const uint32_t ps1 = warp_add_u32<32>(__float2uint_rn(p1 * kProbScale));
      const float z0 = float(ps0) * kInvProbScale;
      const float z1 = float(ps1) * kInvProbScale;
      lrun[u][0] = lrun[u][0] * a0 + z0;
      lrun[u][1] = lrun[u][1] * a1 + z1;
      if (tl + 1 == ntile && lane == 0) {
        Norm[2 * hpr] = lrun[u][0] > 0.f ? 1.f / lrun[u][0] : 0.f;
        Norm[2 * hpr + 1] = lrun[u][1] > 0.f ? 1.f / lrun[u][1] : 0.f;
        float2* ml = (float2*)(smem + B13_ML);
        ml[2 * hpr] = make_float2(mrun[u][0], lrun[u][0]);
        ml[2 * hpr + 1] = make_float2(mrun[u][1], lrun[u][1]);
      }
    }
    asm volatile("st.shared.v2.u32 [%0], {%1, %2};\n" ::"r"(sa(Pb + lane * 48 + hpair0 * 4)),
                 "r"(pkv[0]), "r"(pkv[1])
                 : "memory");
    __syncthreads();

    // ---- PV, rescaled when this tile increases the running maximum.
    {
#pragma unroll
      for (int mi = 0; mi < MIO; ++mi)
#pragma unroll
        for (int n = 0; n < 2; ++n) {
          const float2 av = *(const float2*)(Alp + 8 * n + shd);
          mul_f32x2(o[mi][n][0], o[mi][n][1], av.x, av.y);
          mul_f32x2(o[mi][n][2], o[mi][n][3], av.x, av.y);
        }
#pragma unroll
      for (int ki = 0; ki < 2; ++ki) {
        uint32_t b[4];
        ldm4t(sa(Pb + (ki * 16 + (lane & 15)) * 48 + (lane >> 4) * 16), b[0], b[1], b[2], b[3]);
        const int vrow = ki * 16 + vrow_b;
#pragma unroll
        for (int mi = 0; mi < MIO; mi += 2) {
          uint32_t a0[4], a1[4];
          const int j0 = (warp * 128 + mi * 16) >> 3;
          const int j1 = (warp * 128 + (mi + 1) * 16) >> 3;
          ldm4t(kvs + (uint32_t)(vrow * 1024 + swz(vrow, j0 + vjo) * 16), a0[0], a0[1], a0[2],
                a0[3]);
          ldm4t(kvs + (uint32_t)(vrow * 1024 + swz(vrow, j1 + vjo) * 16), a1[0], a1[1], a1[2],
                a1[3]);
          mma_bf16(o[mi][0], a0, b[0], b[1]);
          mma_bf16(o[mi][1], a0, b[2], b[3]);
          mma_bf16(o[mi + 1][0], a1, b[0], b[1]);
          mma_bf16(o[mi + 1][1], a1, b[2], b[3]);
        }
      }
    }
  }

  {  // cluster epilogue
    cg::cluster_group clu = cg::this_cluster();
    const unsigned rank = clu.block_rank();
    __nv_bfloat16* Osh = (__nv_bfloat16*)smem;
    float2* Lsh = (float2*)(smem + B13_ML);
    constexpr int kDsmXor = 8;
    float2 norms[2];
#pragma unroll
    for (int n = 0; n < 2; ++n) {
      norms[n] = *(const float2*)(Norm + 8 * n + 2 * (lane & 3));
    }

    __syncthreads();
#pragma unroll
    for (int mi = 0; mi < MIO; ++mi) {
      const int dbase = warp * 128 + mi * 16;
#pragma unroll
      for (int n = 0; n < 2; ++n) {
        const int hh = 8 * n + (lane & 7);
        const int d8 = dbase + (((lane >> 3) & 1) * 8);
        const int dx = kDsmXor * (hh & 7);
        const float inv0 = norms[n].x, inv1 = norms[n].y;
        mul_f32x2(o[mi][n][0], o[mi][n][1], inv0, inv1);
        mul_f32x2(o[mi][n][2], o[mi][n][3], inv0, inv1);
        const __nv_bfloat162 r0 = __floats2bfloat162_rn(o[mi][n][0], o[mi][n][1]);
        const __nv_bfloat162 r1 = __floats2bfloat162_rn(o[mi][n][2], o[mi][n][3]);
        stm2t(sa(Osh + hh * DIM + (d8 ^ dx)), *reinterpret_cast<const uint32_t*>(&r0),
              *reinterpret_cast<const uint32_t*>(&r1));
      }
    }
    clu.sync();
    uint32_t* W2sh = (uint32_t*)(smem + B13_S);
    if (tid < HEADS && rank == (unsigned)(tid % CSZ)) {
      const int h = tid;
      const float sk = sink_log2(sinks[h]);
      float M = sk;
      float2 ml[CSZ];
#pragma unroll
      for (int p = 0; p < CSZ; ++p) {
        ml[p] = ((float2*)clu.map_shared_rank(Lsh, p))[h];
        M = fmaxf(M, ml[p].x);
      }
      float den = exp2f(sk - M);
#pragma unroll
      for (int p = 0; p < CSZ; ++p) {
        ml[p].x = exp2f(ml[p].x - M);
        den += ml[p].y * ml[p].x;
      }
      const float inv = 1.f / den;
#pragma unroll
      for (int p = 0; p < CSZ; ++p) {
        const float w = ml[p].x * ml[p].y * inv;
        const uint32_t wb = __float_as_uint(w);
        W2sh[h * 6 + p] = *reinterpret_cast<const uint32_t*>(&wb);
      }
    }
    const int nchunk = HEADS * DIM / 4;
    const int c0 = static_cast<int>(rank) * NTHREAD + tid;
    uint2 pv0[CSZ], pv1[CSZ];
    int h_0 = 0, d_0 = 0, h_1 = 0, d_1 = 0;
    const int c1 = c0 + CSZ * NTHREAD;
    {
      h_0 = c0 >> 7;
      d_0 = (c0 & 127) * 4;
      const int sd = d_0 ^ (kDsmXor * (h_0 & 7));
#pragma unroll
      for (int p = 0; p < CSZ; ++p)
        pv0[p] = *(const uint2*)clu.map_shared_rank(Osh + h_0 * DIM + sd, p);
      h_1 = c1 >> 7;
      d_1 = (c1 & 127) * 4;
      const int sd1 = d_1 ^ (kDsmXor * (h_1 & 7));
#pragma unroll
      for (int p = 0; p < CSZ; ++p)
        pv1[p] = *(const uint2*)clu.map_shared_rank(Osh + h_1 * DIM + sd1, p);
    }
    __syncthreads();
    {
      MergePair a00 = 0, a01 = 0, a10 = 0, a11 = 0;
      const uint2 w00 = *(const uint2*)(W2sh + h_0 * 6);
      const uint2 w01 = *(const uint2*)(W2sh + h_0 * 6 + 2);
      const uint32_t w04 = W2sh[h_0 * 6 + 4];
      const uint2 w10 = *(const uint2*)(W2sh + h_1 * 6);
      const uint2 w11 = *(const uint2*)(W2sh + h_1 * 6 + 2);
      const uint32_t w14 = W2sh[h_1 * 6 + 4];
      const uint32_t w0[CSZ] = {w00.x, w00.y, w01.x, w01.y, w04};
      const uint32_t w1[CSZ] = {w10.x, w10.y, w11.x, w11.y, w14};
#pragma unroll
      for (int p = 0; p < CSZ; ++p) {
        a00 = merge_f32x2(pv0[p].x, w0[p], a00);
        a10 = merge_f32x2(pv1[p].x, w1[p], a10);
        a01 = merge_f32x2(pv0[p].y, w0[p], a01);
        a11 = merge_f32x2(pv1[p].y, w1[p], a11);
      }
      *(uint2*)(out + ((size_t)t * HEADS + h_0) * DIM + d_0) = make_uint2(a00, a01);
      *(uint2*)(out + ((size_t)t * HEADS + h_1) * DIM + d_1) = make_uint2(a10, a11);
    }
    for (int c = c0 + 2 * CSZ * NTHREAD; c < nchunk; c += CSZ * NTHREAD) {
      const int h = c >> 7;
      const int d4 = (c & 127) * 4;
      const int sd4 = d4 ^ (kDsmXor * (h & 7));
      uint2 v[CSZ];
#pragma unroll
      for (int p = 0; p < CSZ; ++p)
        v[p] = *(const uint2*)clu.map_shared_rank(Osh + h * DIM + sd4, p);
      MergePair a0 = 0, a1 = 0;
      const uint2 w0 = *(const uint2*)(W2sh + h * 6);
      const uint2 w1 = *(const uint2*)(W2sh + h * 6 + 2);
      const uint32_t w[CSZ] = {w0.x, w0.y, w1.x, w1.y, W2sh[h * 6 + 4]};
#pragma unroll
      for (int p = 0; p < CSZ; ++p) {
        a0 = merge_f32x2(v[p].x, w[p], a0);
        a1 = merge_f32x2(v[p].y, w[p], a1);
      }
      *(uint2*)(out + ((size_t)t * HEADS + h) * DIM + d4) = make_uint2(a0, a1);
    }
    asm volatile(
        "barrier.cluster.arrive.relaxed.aligned;\n\t"
        "barrier.cluster.wait.aligned;\n" ::
            : "memory");
  }
}

// Per-device one-time setup state (multi-GPU: attributes, occupancy and SM
// counts differ per device; the init block is idempotent per device).
struct DevCfg {
  int sm = 0;
  bool init = false;
  int maxclu = 8;
  int maxcl[11];
};
constexpr int kMaxDevices = 64;
static DevCfg g_devcfg[kMaxDevices];
// TVM FFI releases the GIL, so two host threads can race the per-device
// init block; call_once makes a cold concurrent call wait for a complete,
// published configuration instead of reading one mid-rewrite.
static std::once_flag g_devcfg_once[kMaxDevices];

#define SET_ATTR(N)                                                                  \
  cudaFuncSetAttribute(mla_main<32, N>, cudaFuncAttributeMaxDynamicSharedMemorySize, \
                       SM_BYTES_FOR(32));                                            \
  cudaFuncSetAttribute(mla_main<32, N>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);

#define SET_OWNER_ATTR(N)                                                                         \
  cudaFuncSetAttribute(mla_main<32, N, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, \
                       SM_BYTES_FOR(32));                                                         \
  cudaFuncSetAttribute(mla_main<32, N, false, true>,                                              \
                       cudaFuncAttributeNonPortableClusterSizeAllowed, 1);

void Run(ffi::TensorView q, ffi::TensorView swa, ffi::TensorView ckv,
         ffi::TensorView sparse_indices, ffi::TensorView sparse_topk_lens, ffi::TensorView seq_lens,
         ffi::TensorView cum_seq_lens_q, int64_t max_q_len, ffi::TensorView sinks,
         ffi::TensorView out) {
  if (q.shape()[0] == 0) TVM_FFI_THROW(ValueError) << "q must contain at least one token";
  DLDevice dev = q.device();
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));
#ifdef ENABLE_G4
  CUtensorMap swa_tm = {}, ckv_tm = {};
  const cuuint64_t swa_rows = (cuuint64_t)swa.shape()[0] * (cuuint64_t)swa.shape()[1];
  const cuuint64_t ckv_rows = (cuuint64_t)ckv.shape()[0] * (cuuint64_t)ckv.shape()[1];
  const cuuint64_t swa_dims[2] = {512, swa_rows};
  const cuuint64_t ckv_dims[2] = {512, ckv_rows};
  const cuuint64_t strides[1] = {1024};
  const cuuint32_t box[2] = {64, 1};
  const cuuint32_t elem_strides[2] = {1, 1};
  const CUresult swa_map_ok = cuTensorMapEncodeTiled(
      &swa_tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, swa.data_ptr(), swa_dims, strides, box,
      elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  const CUresult ckv_map_ok = cuTensorMapEncodeTiled(
      &ckv_tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ckv.data_ptr(), ckv_dims, strides, box,
      elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (swa_map_ok != CUDA_SUCCESS || ckv_map_ok != CUDA_SUCCESS) {
    TVM_FFI_THROW(RuntimeError) << "sparse_mla_decode_dsv4: cuTensorMapEncodeTiled failed (swa="
                                << (int)swa_map_ok << ", ckv=" << (int)ckv_map_ok
                                << "); refusing to return stale output";
  }
#endif
  ffi::CUDADeviceGuard device_guard(dev.device_id);
  DevCfg& dc = g_devcfg[dev.device_id & (kMaxDevices - 1)];
  std::call_once(g_devcfg_once[dev.device_id & (kMaxDevices - 1)], [&] {
    SET_ATTR(1)
    SET_ATTR(2)
    SET_ATTR(3)
    SET_ATTR(4)
    SET_ATTR(5)
    SET_ATTR(7)
    SET_ATTR(10)
    SET_OWNER_ATTR(10)
    cudaFuncSetAttribute(mla_main<32, 7, false, false, false>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SM_BYTES_FOR(32));
    cudaFuncSetAttribute(mla_main<32, 7, false, false, false>,
                         cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaFuncSetAttribute(mla_main<32, 7, false, false, false, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SM_BYTES_FOR(32));
    cudaFuncSetAttribute(mla_main<32, 7, false, false, false, true>,
                         cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaFuncSetAttribute(mla_ws_c4, cudaFuncAttributeMaxDynamicSharedMemorySize, WS_BYTES);
    cudaFuncSetAttribute(mla_ws_c4, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaFuncSetAttribute(mla_fused2_c10<false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         F2_BYTES);
    cudaFuncSetAttribute(mla_fused2_c10<false>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaFuncSetAttribute(mla_fused2_c10<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         F2_BYTES);
    cudaFuncSetAttribute(mla_fused2_c10<true>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaFuncSetAttribute(mla_fused2_c10<true, true, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, F2_BYTES);
    cudaFuncSetAttribute(mla_fused2_c10<true, true, true>,
                         cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaFuncSetAttribute(mla_fused2_c10<true, false, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, F2_BYTES);
    cudaFuncSetAttribute(mla_fused2_c10<true, false, true>,
                         cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaFuncSetAttribute(mla_fused2_c10<true, false, true, true, false, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, F2_BYTES);
    cudaFuncSetAttribute(mla_fused2_c10<true, false, true, true, false, true>,
                         cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaFuncSetAttribute(mla_fused2_c10<true, true, true, true, false, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, F2_BYTES);
    cudaFuncSetAttribute(mla_fused2_c10<true, true, true, true, false, true>,
                         cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaFuncSetAttribute(mla_fused2_c10<true, true, true, false, false, true, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, F2_BYTES);
    cudaFuncSetAttribute(mla_fused2_c10<true, true, true, false, false, true, true>,
                         cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaFuncSetAttribute(mla_fused2_c10<true, true, true, true, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, F2_BYTES);
    cudaFuncSetAttribute(mla_fused2_c10<true, true, true, true, true>,
                         cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaFuncSetAttribute(mla_fused2_c10<false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         F2_BYTES);
    cudaFuncSetAttribute(mla_fused2_c10<false, false>,
                         cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaFuncSetAttribute(mla_fused2_h8_c10<false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         H8_BYTES);
    cudaFuncSetAttribute(mla_fused2_h8_c10<false>, cudaFuncAttributeNonPortableClusterSizeAllowed,
                         1);
    cudaFuncSetAttribute(mla_fused2_h8_c10<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         H8_BYTES);
    cudaFuncSetAttribute(mla_fused2_h8_c10<true>, cudaFuncAttributeNonPortableClusterSizeAllowed,
                         1);
    cudaFuncSetAttribute(mla_tcgen_c10, cudaFuncAttributeMaxDynamicSharedMemorySize, TC_BYTES);
    cudaFuncSetAttribute(mla_tcgen_c10, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaFuncSetAttribute(mla_main<32, 3, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         SM_BYTES_FOR(32));
    cudaFuncSetAttribute(mla_main<32, 3, true>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaFuncSetAttribute(mla_main<16, 5>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         SM_BYTES_FOR(16));
    cudaFuncSetAttribute(mla_main<16, 5>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaFuncSetAttribute(mla_b13<5>, cudaFuncAttributeMaxDynamicSharedMemorySize, B13_BYTES);
    cudaFuncSetAttribute(mla_b13<5>, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    cudaFuncSetAttribute(mla_b13<5>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaDeviceGetAttribute(&dc.sm, cudaDevAttrMultiProcessorCount, dev.device_id);
    if (dc.sm <= 0) dc.sm = 152;
    int mc = 0;
    cudaLaunchConfig_t probe = {};
    probe.gridDim = dim3(1, 1, 1);
    probe.blockDim = dim3(256, 1, 1);
    probe.dynamicSmemBytes = SM_BYTES_FOR(32);
    if (cudaOccupancyMaxPotentialClusterSize(&mc, (const void*)mla_main<32, 4>, &probe) !=
            cudaSuccess ||
        mc <= 0)
      mc = 8;
    dc.maxclu = mc > 10 ? 10 : mc;
    for (int i = 0; i < 11; ++i) dc.maxcl[i] = 0;
    cudaLaunchAttribute pat[1];
    pat[0].id = cudaLaunchAttributeClusterDimension;
    pat[0].val.clusterDim.y = 1;
    pat[0].val.clusterDim.z = 1;
    probe.attrs = pat;
    probe.numAttrs = 1;
#define PROBE(N)                                                                     \
  {                                                                                  \
    int nc = 0;                                                                      \
    pat[0].val.clusterDim.x = N;                                                     \
    probe.gridDim = dim3(N, 1, 1);                                                   \
    if (cudaOccupancyMaxActiveClusters(&nc, (const void*)mla_main<32, N>, &probe) == \
            cudaSuccess &&                                                           \
        nc > 0)                                                                      \
      dc.maxcl[N] = nc;                                                              \
    else                                                                             \
      dc.maxcl[N] = (2 * dc.sm) / N;                                                 \
  }
    PROBE(1)
    PROBE(2)
    PROBE(3)
    PROBE(4)
    PROBE(5)
    PROBE(7)
    PROBE(10)
#undef PROBE
    dc.init = true;
  });
  const int64_t T64 = q.shape()[0];
  // CUDA grid Y is limited to 65535.  Large in-scope packed-token counts use
  // the unchanged single-rank generic kernel in same-stream pointer chunks.
  if (T64 > 65535) {
    const uint8_t* qp0 = (const uint8_t*)q.data_ptr();
    const uint8_t* sp0 = (const uint8_t*)swa.data_ptr();
    const uint8_t* cp0 = (const uint8_t*)ckv.data_ptr();
    const int* ip0 = (const int*)sparse_indices.data_ptr();
    const int* lp0 = (const int*)sparse_topk_lens.data_ptr();
    const float* skp0 = (const float*)sinks.data_ptr();
    __nv_bfloat16* op0 = (__nv_bfloat16*)out.data_ptr();

    cudaLaunchConfig_t lc = {};
    lc.blockDim = dim3(256, 1, 1);
    lc.dynamicSmemBytes = SM_BYTES_FOR(32);
    lc.stream = stream;
    cudaLaunchAttribute la[2];
    la[0].id = cudaLaunchAttributeClusterDimension;
    la[0].val.clusterDim.x = 1;
    la[0].val.clusterDim.y = 1;
    la[0].val.clusterDim.z = 1;
    la[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
    la[1].val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicyLoadBalancing;
    lc.attrs = la;
    lc.numAttrs = 2;
    for (int64_t t0 = 0; t0 < T64; t0 += 65535) {
      const int64_t rem = T64 - t0;
      const unsigned n = (unsigned)(rem > 65535 ? 65535 : rem);
      lc.gridDim = dim3(1, n, 1);
      cudaError_t e = cudaLaunchKernelEx(&lc, mla_main<32, 1>, qp0 + (size_t)t0 * HEADS * DIM * 2,
                                         sp0, cp0, ip0 + (size_t)t0 * WIDTH, lp0 + t0, skp0,
                                         op0 + (size_t)t0 * HEADS * DIM, 20 G4_ARG);
      if (e != cudaSuccess) {
        cudaGetLastError();
        TVM_FFI_THROW(RuntimeError) << "sparse_mla_decode_dsv4: chunked launch failed at token "
                                       "offset "
                                    << t0 << " of " << T64 << ": " << cudaGetErrorString(e);
      }
    }
    return;
  }
  const int T = (int)T64;
  const bool use_tile16 = T == 78;
  const bool use_fused2 = T == 12 || T == 18 || T == 24;
  static const int kCand[7] = {10, 7, 5, 4, 3, 2, 1};
  int nsplit = use_tile16 ? 5 : 1;
  if (!use_tile16) {
    for (int i = 0; i < 7; ++i) {
      const int c = kCand[i];
      if (c > dc.maxclu) continue;
      if ((long long)c * T > 2LL * dc.sm) continue;
      if (T > dc.maxcl[c]) continue;
      nsplit = c;
      break;
    }
  }
  const int total_tiles = use_tile16 ? 40 : 20;
  int tps = (total_tiles + nsplit - 1) / nsplit;
  nsplit = (total_tiles + tps - 1) / tps;
  if (T == 60) {
    nsplit = 4;
    tps = 5;
  } else if (T == 30) {
    nsplit = 7;
    tps = 3;
  }
  const bool use_ws = false;
  const bool use_tcgen = false;
  const bool use_h8 = T == 6 && nsplit == 10;

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(nsplit, T, use_h8 ? 2 : 1);
  cfg.blockDim = dim3(use_tile16 ? 128 : 256, 1, 1);
  const bool use_b13 = use_tile16;
  cfg.dynamicSmemBytes = use_h8       ? H8_BYTES
                         : use_ws     ? WS_BYTES
                         : use_tcgen  ? TC_BYTES
                         : (use_fused2 && nsplit == 10) ? F2_BYTES
                         : use_b13    ? B13_BYTES
                                      : SM_BYTES_FOR(32);
  if (use_b13) cfg.blockDim = dim3(128, 1, 1);
  cfg.stream = stream;
  cudaLaunchAttribute attr[2];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = nsplit;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  attr[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
  attr[1].val.clusterSchedulingPolicyPreference =
      (T == 12 || T == 30 || T == 36) ? cudaClusterSchedulingPolicySpread
                                      : (T == 6 ? cudaClusterSchedulingPolicyDefault
                                                : cudaClusterSchedulingPolicyLoadBalancing);
  cfg.attrs = attr;
  cfg.numAttrs = 2;
  const uint8_t* qp = (const uint8_t*)q.data_ptr();
  const uint8_t* sp = (const uint8_t*)swa.data_ptr();
  const uint8_t* cp = (const uint8_t*)ckv.data_ptr();
  const int* ip = (const int*)sparse_indices.data_ptr();
  const int* lp = (const int*)sparse_topk_lens.data_ptr();
  const float* skp = (const float*)sinks.data_ptr();
  __nv_bfloat16* outp = (__nv_bfloat16*)out.data_ptr();
  cudaError_t err = cudaErrorInvalidValue;
#define LAUNCH(N)                                                                               \
  case N:                                                                                       \
    err = cudaLaunchKernelEx(&cfg, mla_main<32, N>, qp, sp, cp, ip, lp, skp, outp, tps G4_ARG); \
    break;
  const bool use_owner = T == 18 || T == 24;
  const bool use_bal7 = (T == 30 || T == 36) && nsplit == 7;
  if (use_h8) {
    err = cudaLaunchKernelEx(&cfg, mla_fused2_h8_c10<true>, qp, sp, cp, ip, lp, skp, outp);
  } else if (use_ws) {
    err = cudaLaunchKernelEx(&cfg, mla_ws_c4, qp, sp, cp, ip, lp, skp, outp);
  } else if (use_tcgen) {
    err = cudaLaunchKernelEx(&cfg, mla_tcgen_c10, qp, sp, cp, ip, lp, skp, outp);
  } else if (use_fused2 && nsplit == 10) {
    if (T == 12)
      err = cudaLaunchKernelEx(&cfg, mla_fused2_c10<true, true, true, true, true>, qp, sp, cp, ip,
                               lp, skp, outp);
    else if (T == 24)
      err = cudaLaunchKernelEx(&cfg, mla_fused2_c10<true, false, true, true, false, true>, qp, sp,
                               cp, ip, lp, skp, outp);
    else if (T == 18)
      err = cudaLaunchKernelEx(&cfg, mla_fused2_c10<true, true, true, false, false, true, true>, qp,
                               sp, cp, ip, lp, skp, outp);
    else
      err =
          cudaLaunchKernelEx(&cfg, mla_fused2_c10<true, true, true>, qp, sp, cp, ip, lp, skp, outp);
  } else if (use_b13) {
    err = cudaLaunchKernelEx(&cfg, mla_b13<5>, qp, sp, cp, ip, lp, skp, outp);
  } else {
    if (use_owner && nsplit == 10) {
      err = cudaLaunchKernelEx(&cfg, mla_main<32, 10, false, true>, qp, sp, cp, ip, lp, skp, outp,
                               tps G4_ARG);
    } else if (nsplit == 7) {
      if (use_bal7) {
        err = cudaLaunchKernelEx(&cfg, mla_main<32, 7, false, false, false, true>, qp, sp, cp, ip,
                                 lp, skp, outp, tps G4_ARG);
      } else {
        err = cudaLaunchKernelEx(&cfg, mla_main<32, 7, false, false, false>, qp, sp, cp, ip, lp,
                                 skp, outp, tps G4_ARG);
      }
    } else {
      switch (nsplit) {
        LAUNCH(1) LAUNCH(2) LAUNCH(3) LAUNCH(4) LAUNCH(5) LAUNCH(7) LAUNCH(10) default : break;
      }
    }
  }
#undef LAUNCH
  if (err != cudaSuccess) {
    cudaGetLastError();
    cfg.blockDim = dim3(256, 1, 1);
    cfg.dynamicSmemBytes = SM_BYTES_FOR(32);
    cfg.gridDim = dim3(1, T, 1);
    attr[0].val.clusterDim.x = 1;
    const cudaError_t fallback =
        cudaLaunchKernelEx(&cfg, mla_main<32, 1>, qp, sp, cp, ip, lp, skp, outp, 20 G4_ARG);
    if (fallback != cudaSuccess) {
      cudaGetLastError();
      TVM_FFI_THROW(RuntimeError) << cudaGetErrorString(fallback);
    }
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_mla_decode_dsv4_run, Run);
