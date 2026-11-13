// Complementary wider-N tiling path for the high-M shapes where one n8 tile
// per CTA leaves too little instruction-level parallelism.  This is a
// standalone, hand-written raw-MMA kernel; the public TVM-FFI entry remains in
// kernel.cu.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace wide {

__device__ __forceinline__ void mma_e4m3(
    float& d0, float& d1, float& d2, float& d3, uint32_t a0, uint32_t a1,
    uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// The alternate K permutation makes all eight fragment words contiguous, so
// one 256-bit transaction replaces the two 128-bit loads used by the original
// wider-N kernel.
struct Frag8 {
  uint32_t v[8];
};

__device__ __forceinline__ Frag8 ld_stream256(const void* p) {
  Frag8 r;
  asm("ld.global.nc.L1::evict_first.L2::256B.v8.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3]),
        "=r"(r.v[4]), "=r"(r.v[5]), "=r"(r.v[6]), "=r"(r.v[7])
      : "l"(p));
  return r;
}

__device__ __forceinline__ Frag8 ld_keep256(const void* p) {
  Frag8 r;
  asm("ld.global.nc.L1::evict_last.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3]),
        "=r"(r.v[4]), "=r"(r.v[5]), "=r"(r.v[6]), "=r"(r.v[7])
      : "l"(p));
  return r;
}

__device__ __forceinline__ Frag8 ld_stream128(const void* p) {
  Frag8 r;
  asm("ld.global.nc.L1::no_allocate.L2::256B.v4.b32 {%0,%1,%2,%3}, [%4];"
      : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3])
      : "l"(p));
  asm("ld.global.nc.L1::no_allocate.L2::256B.v4.b32 {%0,%1,%2,%3}, [%4];"
      : "=r"(r.v[4]), "=r"(r.v[5]), "=r"(r.v[6]), "=r"(r.v[7])
      : "l"(static_cast<const uint8_t*>(p) + 64));
  return r;
}

__device__ __forceinline__ Frag8 ld_pair128(const void* p) {
  Frag8 r;
  const uint4 x0 = *reinterpret_cast<const uint4*>(p);
  const uint4 x1 = *reinterpret_cast<const uint4*>(
      static_cast<const uint8_t*>(p) + 64);
  r.v[0] = x0.x;
  r.v[1] = x0.y;
  r.v[2] = x0.z;
  r.v[3] = x0.w;
  r.v[4] = x1.x;
  r.v[5] = x1.y;
  r.v[6] = x1.z;
  r.v[7] = x1.w;
  return r;
}

__device__ __forceinline__ Frag8 ld_keep128(const void* p) {
  Frag8 r;
  asm("ld.global.nc.L1::evict_last.v4.b32 {%0,%1,%2,%3}, [%4];"
      : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3])
      : "l"(p));
  asm("ld.global.nc.L1::evict_last.v4.b32 {%0,%1,%2,%3}, [%4];"
      : "=r"(r.v[4]), "=r"(r.v[5]), "=r"(r.v[6]), "=r"(r.v[7])
      : "l"(static_cast<const uint8_t*>(p) + 64));
  return r;
}

// NT: n8 tiles per warp; MQ: m16 tiles; NWN: warp groups spread along N.
template <int NW, int NT, int MQ, int NWN, bool V256, bool HALF,
          bool SINGLE = false>
__global__ void __launch_bounds__(NW * 32, ((NW == 16 || SINGLE) ? 1 : 2))
gemm_fp8_blk(
    const uint8_t* __restrict__ Ap, const uint8_t* __restrict__ Bp,
    const float* __restrict__ Asp, const float* __restrict__ Bsp,
    __nv_bfloat16* __restrict__ Y, int M, int N, int K, int KB) {
  constexpr int NTHREADS = NW * 32;
  constexpr int NWK = NW / NWN;
  constexpr int WNW = NT * 8;
  constexpr int MP = MQ * (HALF ? 8 : 16);
  constexpr int RB = HALF ? 8 : 16;
  constexpr int MPE = MQ * RB;
  constexpr int RW = (NT >= 2) ? 16 : 8;
  constexpr int TPR = RW / 8;
  constexpr int LD = RW + 4;
  constexpr int ROUNDS = NT / TPR;

  __shared__ float sR[NW * MPE * LD];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wid = tid >> 5;
  const int ng = wid % NWN;
  const int wk = wid / NWN;
  const int g = lane >> 2;
  const int tg = lane & 3;
  const int nc = blockIdx.x * (NWN * WNW);
  const int n0 = nc + ng * WNW;
  const int m0 = blockIdx.y * MP;

  const uint8_t* bptr[NT];
#pragma unroll
  for (int t = 0; t < NT; ++t) {
    int r = n0 + t * 8 + g;
    r = r < N ? r : N - 1;
    bptr[t] = Bp + (size_t)r * K + tg * (V256 ? 32 : 16);
  }
  const uint8_t* aptr[MQ][2];
  const float* asp[MQ][2];
#pragma unroll
  for (int q = 0; q < MQ; ++q) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      int r = m0 + q * (HALF ? 8 : 16) + g + j * 8;
      r = r < M ? r : M - 1;
      aptr[q][j] = Ap + (size_t)r * K + tg * (V256 ? 32 : 16);
      asp[q][j] = Asp + (size_t)r * KB;
    }
  }
  const float* bsp = Bsp + (size_t)(nc >> 7) * KB;

  float acc[MQ][NT][4];
#pragma unroll
  for (int q = 0; q < MQ; ++q)
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[q][t][r] = 0.f;

  struct Blk {
    Frag8 wb[NT];
    Frag8 wa[MQ][2];
    float s0[MQ], s1[MQ];
  };

  auto load_blk = [&](int kb, Blk& x) {
    const int kbase = kb * 128;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      if constexpr (V256)
        x.wb[t] = ld_stream256(bptr[t] + kbase);
      else if constexpr (NT <= 2)
        x.wb[t] = ld_stream128(bptr[t] + kbase);
      else
        x.wb[t] = ld_pair128(bptr[t] + kbase);
    }
#pragma unroll
    for (int q = 0; q < MQ; ++q)
#pragma unroll
      for (int j = 0; j < (HALF ? 1 : 2); ++j) {
        if constexpr (V256)
          x.wa[q][j] = ld_keep256(aptr[q][j] + kbase);
        else
          x.wa[q][j] = ld_pair128(aptr[q][j] + kbase);
      }
    const float bs = bsp[kb];
#pragma unroll
    for (int q = 0; q < MQ; ++q) {
      x.s0[q] = asp[q][0][kb] * bs;
      x.s1[q] = HALF ? x.s0[q] : asp[q][1][kb] * bs;
    }
  };

  auto compute_blk = [&](const Blk& x) {
    float d[MQ][NT][4];
#pragma unroll
    for (int q = 0; q < MQ; ++q)
#pragma unroll
      for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int r = 0; r < 4; ++r) d[q][t][r] = 0.f;
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
#pragma unroll
      for (int q = 0; q < MQ; ++q) {
        const Frag8& u0 = x.wa[q][0];
        const Frag8& u1 = x.wa[q][HALF ? 0 : 1];
        const uint32_t a0 = u0.v[2 * ks];
        const uint32_t a1 = u1.v[2 * ks];
        const uint32_t a2 = u0.v[2 * ks + 1];
        const uint32_t a3 = u1.v[2 * ks + 1];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
          const Frag8& v = x.wb[t];
          mma_e4m3(d[q][t][0], d[q][t][1], d[q][t][2], d[q][t][3],
                   a0, a1, a2, a3, v.v[2 * ks], v.v[2 * ks + 1]);
        }
      }
    }
#pragma unroll
    for (int q = 0; q < MQ; ++q)
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        acc[q][t][0] += d[q][t][0] * x.s0[q];
        acc[q][t][1] += d[q][t][1] * x.s0[q];
        acc[q][t][2] += d[q][t][2] * x.s1[q];
        acc[q][t][3] += d[q][t][3] * x.s1[q];
      }
  };

  if (NT <= 2 && MQ == 1 && !SINGLE) {
    Blk x0, x1;
    for (int kb = wk; kb < KB; kb += 2 * NWK) {
      const int kn = kb + NWK;
      const bool two = kn < KB;
      load_blk(kb, x0);
      if (two) load_blk(kn, x1);
      compute_blk(x0);
      if (two) compute_blk(x1);
    }
  } else {
    Blk x0;
    for (int kb = wk; kb < KB; kb += NWK) {
      load_blk(kb, x0);
      compute_blk(x0);
    }
  }

  if (NWK == 1) {
#pragma unroll
    for (int q = 0; q < MQ; ++q)
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        const int gn = n0 + t * 8 + tg * 2;
        if (gn >= N) continue;
        const int ma = m0 + q * (HALF ? 8 : 16) + g;
        const int mb = ma + 8;
        if (ma < M) {
          Y[(size_t)ma * N + gn] = __float2bfloat16(acc[q][t][0]);
          if (gn + 1 < N)
            Y[(size_t)ma * N + gn + 1] = __float2bfloat16(acc[q][t][1]);
        }
        if (mb < M) {
          Y[(size_t)mb * N + gn] = __float2bfloat16(acc[q][t][2]);
          if (gn + 1 < N)
            Y[(size_t)mb * N + gn + 1] = __float2bfloat16(acc[q][t][3]);
        }
      }
    return;
  }

  float* sw = sR + wid * MPE * LD;
#pragma unroll
  for (int r = 0; r < ROUNDS; ++r) {
    // Round zero has no prior readers of sR.  Later rounds must wait until
    // every thread finishes reading the previous contents before reuse.
    if (r > 0) __syncthreads();
#pragma unroll
    for (int q = 0; q < MQ; ++q)
#pragma unroll
      for (int j = 0; j < TPR; ++j) {
        const int t = r * TPR + j;
        const int c = j * 8 + tg * 2;
        sw[(q * RB + g) * LD + c] = acc[q][t][0];
        sw[(q * RB + g) * LD + c + 1] = acc[q][t][1];
        if constexpr (!HALF) {
          sw[(q * RB + g + 8) * LD + c] = acc[q][t][2];
          sw[(q * RB + g + 8) * LD + c + 1] = acc[q][t][3];
        }
      }
    __syncthreads();
    for (int e = tid; e < NWN * MPE * RW; e += NTHREADS) {
      const int eg = e / (MPE * RW);
      const int rest = e - eg * (MPE * RW);
      const int mm = rest / RW;
      const int cc = rest % RW;
      float s = 0.f;
#pragma unroll
      for (int i = 0; i < NWK; ++i)
        s += sR[(eg + i * NWN) * MPE * LD + mm * LD + cc];
      const int gm = m0 + mm;
      const int gn = nc + eg * WNW + r * RW + cc;
      if (gm < M && gn < N)
        Y[(size_t)gm * N + gn] = __float2bfloat16(s);
    }
  }
}

template <int NW, int NT, int MQ, int NWN, bool V256, bool HALF,
          bool SINGLE = false>
inline void launch(const uint8_t* a, const uint8_t* b, const float* as,
                   const float* bs, __nv_bfloat16* y, int M, int N, int K,
                   int KB, cudaStream_t stream) {
  constexpr int CN = NWN * NT * 8;
  constexpr int CM = MQ * (HALF ? 8 : 16);
  dim3 grid((N + CN - 1) / CN, (M + CM - 1) / CM);
  gemm_fp8_blk<NW, NT, MQ, NWN, V256, HALF, SINGLE>
      <<<grid, NW * 32, 0, stream>>>(
      a, b, as, bs, y, M, N, K, KB);
}

template <bool V256, bool HALF = false>
void dispatch(const uint8_t* a, const uint8_t* b, const float* as,
              const float* bs, __nv_bfloat16* y, int M, int N, int K, int KB,
              cudaStream_t stream) {
  int cn;
  if (N >= 64 * 188)
    cn = 64;
  else if (N >= 32 * 188)
    cn = 32;
  else if (N > 512)
    cn = 16;
  else
    cn = 8;

  if (M <= 16) {
    if (cn == 64)
      launch<8, 4, 1, 2, V256, HALF>(a, b, as, bs, y, M, N, K, KB, stream);
    else if (cn == 32)
      launch<8, 2, 1, 2, V256, HALF>(a, b, as, bs, y, M, N, K, KB, stream);
    else if (cn == 16) {
      if (KB <= 4) {
        if constexpr (HALF) {
          if (M == 16) {
            launch<8, 2, 1, 2, V256, HALF, true>(
                a, b, as, bs, y, M, N, K, KB, stream);
            return;
          }
        }
        launch<4, 2, 1, 1, V256, HALF>(a, b, as, bs, y, M, N, K, KB, stream);
      }
      else if (M == 16)
        launch<16, 2, 1, 1, V256, HALF>(a, b, as, bs, y, M, N, K, KB, stream);
      else if constexpr (HALF) {
        if (M == 8 && KB == 32)
          launch<16, 2, 1, 1, V256, HALF>(
              a, b, as, bs, y, M, N, K, KB, stream);
        else
          launch<8, 2, 1, 1, V256, HALF>(
              a, b, as, bs, y, M, N, K, KB, stream);
      } else {
        launch<8, 2, 1, 1, V256, HALF>(a, b, as, bs, y, M, N, K, KB, stream);
      }
    } else {
      launch<8, 1, 1, 1, V256, HALF>(a, b, as, bs, y, M, N, K, KB, stream);
    }
  } else {
    if (cn == 64)
      launch<8, 4, 2, 2, V256, HALF>(a, b, as, bs, y, M, N, K, KB, stream);
    else if (cn == 32)
      launch<8, 2, 2, 2, V256, HALF>(a, b, as, bs, y, M, N, K, KB, stream);
    else if (cn == 16) {
      if (KB <= 4) {
        if (M == 32) {
          launch<8, 2, 1, 2, V256, HALF, true>(
              a, b, as, bs, y, M, N, K, KB, stream);
          return;
        }
        launch<4, 2, 1, 1, V256, HALF>(a, b, as, bs, y, M, N, K, KB, stream);
      }
      else
        launch<16, 2, 2, 1, V256, HALF>(a, b, as, bs, y, M, N, K, KB, stream);
    } else {
      launch<8, 1, 2, 1, V256, HALF>(a, b, as, bs, y, M, N, K, KB, stream);
    }
  }
}

}  // namespace wide

void LaunchWide(const uint8_t* a, const uint8_t* b, const float* as,
                const float* bs, __nv_bfloat16* y, int M, int N, int K,
                int KB, cudaStream_t stream) {
  if (M <= 16 && N == 2048 &&
      ((M <= 8 && K == 4096) || (K == 512 && M >= 2)))
    wide::dispatch<true, true>(a, b, as, bs, y, M, N, K, KB, stream);
  else if (M == 16)
    wide::dispatch<true, false>(a, b, as, bs, y, M, N, K, KB, stream);
  else if (M == 32)
    wide::dispatch<true, false>(a, b, as, bs, y, M, N, K, KB, stream);
  else
    wide::dispatch<false, false>(a, b, as, bs, y, M, N, K, KB, stream);
}





