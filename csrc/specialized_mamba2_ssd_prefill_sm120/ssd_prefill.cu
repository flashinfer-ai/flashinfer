// Mamba2 SSD chunk-scan prefill (SSM stage), sm_120.
// One block owns one head (and optionally half of HD); persistent over chunks,
// SSM state kept in registers. bf16 mma.sync m16n8k16 with fp32 accumulation.
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/device_guard.h>
#include <tvm/ffi/tvm_ffi.h>

#include <atomic>
#include <cstring>
#include <mutex>

extern "C" __device__ float __nv_expf(float);
extern "C" __device__ float __nv_log1pf(float);

namespace ffi = tvm::ffi;
typedef __nv_bfloat16 bf16;
#define DEVI __device__ __forceinline__

DEVI uint32_t peer_sptr(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

template <int UNITS>
DEVI uint32_t addrU(uint32_t base, int row, int unit) {
  // A 32-column X tile has a 64-byte row stride.  Using row&3 repeats
  // ldmatrix bank groups at rows r and r+4; row bits 1-2 instead spread
  // each 8-row address group over all 32 banks.  Keep the proven layout
  // for the wider B/C/state tiles.
  int sw = (UNITS == 4) ? ((row >> 1) & 3) : (row & (UNITS - 1));
  return base + (uint32_t)(row * (UNITS * 16)) + (uint32_t)((unit ^ sw) << 4);
}

// Lane-remapped ldmatrix.x4 address: results (r0,r1) and (r2,r3) are
// directly aligned MMA B-operand pairs.  addrU's natural result order needs
// (r0,r2)/(r1,r3), which makes ptxas insert register-pair assembly MOVs.
template <int UNITS>
DEVI void addrUB_split(uint32_t base, int row0, int lane, uint32_t& a0, uint32_t& px) {
  int row = row0 + 8 * ((lane >> 4) & 1) + (lane & 7);
  int sw = (UNITS == 4) ? ((row >> 1) & 3) : (row & (UNITS - 1));
  a0 = base + (uint32_t)(row * (UNITS * 16));
  px = (uint32_t)((((lane >> 3) & 1) ^ sw) << 4);
}

DEVI void peer_ldm4(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
DEVI void peer_ldm4t(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
DEVI void peer_stm2(uint32_t a, uint32_t r0, uint32_t r1) {
  asm volatile("stmatrix.sync.aligned.m8n8.x2.shared::cta.b16 [%0], {%1,%2};\n" ::"r"(a), "r"(r0),
               "r"(r1));
}
DEVI void peer_stm4(uint32_t a, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  asm volatile(
      "stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 "
      "[%0], {%1,%2,%3,%4};\n" ::"r"(a),
      "r"(r0), "r"(r1), "r"(r2), "r"(r3));
}
DEVI void mmaf(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
DEVI void peer_cpa16(uint32_t dst, const void* src, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
}
DEVI void peer_cpa_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
DEVI void cpa_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

template <bool L2>
DEVI float dexp(float x) {
  return L2 ? exp2f(x) : __expf(fminf(0.f, x));
}

// Value punning between a two-element bf16 vector and its 32 packed bits.
// memcpy is the aliasing-safe spelling; nvcc lowers it to a plain move.
template <typename To, typename From>
DEVI To punned(From v) {
  static_assert(sizeof(To) == sizeof(From), "punned requires same-size types");
  To r;
  memcpy(&r, &v, sizeof(To));
  return r;
}

DEVI uint32_t pack2(float a, float b) { return punned<uint32_t>(__floats2bfloat162_rn(a, b)); }
DEVI uint32_t scale2(uint32_t v, float s0, float s1) {
  __nv_bfloat162 b = punned<__nv_bfloat162>(v);
  __nv_bfloat162 s = __floats2bfloat162_rn(s0, s1);
  return punned<uint32_t>(__hmul2(b, s));
}
DEVI uint32_t scale2_packed(uint32_t v, uint32_t w) {
  __nv_bfloat162 b = punned<__nv_bfloat162>(v);
  __nv_bfloat162 s = punned<__nv_bfloat162>(w);
  return punned<uint32_t>(__hmul2(b, s));
}

DEVI float softplus_fast(float z) {
  return z > 8.f ? z : (z < -5.f ? __expf(z) : __logf(1.f + __expf(z)));
}

// Rare large-magnitude fp32 replay of the chunked SSD formula for one head's
// p-columns [p0, p0+pcnt).  Defined below (after the regx kernels); shared by
// the NH128 (regx) and NH64 escalation paths.
__device__ __noinline__ void precise_head(const __nv_bfloat16* x, const __nv_bfloat16* dt,
                                          const float* A, const __nv_bfloat16* B,
                                          const __nv_bfloat16* C, const __nv_bfloat16* D,
                                          const __nv_bfloat16* bias, __nv_bfloat16* y, int T,
                                          int NH, int h, int p0, int pcnt, char* scratch);

// HD_BLK: p-values handled per block (64 -> whole head, 32 -> half head)
template <int HD_BLK, bool FULL_T = false, bool ONE_CHUNK = false, bool DIRECT_DECAY = false>
__global__ __launch_bounds__(256, 1) void ssd_kernel(
    const bf16* __restrict__ xg, const bf16* __restrict__ dtg, const float* __restrict__ Ag,
    const bf16* __restrict__ Bg, const bf16* __restrict__ Cg, const bf16* __restrict__ Dg,
    const bf16* __restrict__ dtbg, bf16* __restrict__ yg, int T, int NH, int nchunks) {
  constexpr int P_BLOCKS = HD_BLK / 16;
  constexpr int N_PART = 16 * P_BLOCKS;
  constexpr int NT_Y = HD_BLK / 8;
  constexpr int NT_S = N_PART / 8;
  constexpr int XU = HD_BLK / 8;
  constexpr int HDSPLIT = 64 / HD_BLK;
  constexpr int HEADS = (HD_BLK == 32) ? 64 : 128;
  constexpr bool L2E = (HD_BLK <= 32);  // decay math in log2 units (fewer MUFU inputs)

  extern __shared__ char smem[];
  bf16* sC = (bf16*)smem;
  bf16* sB = (bf16*)(smem + 32768);
  // x is double-buffered so the next chunk's rows stage a whole chunk ahead;
  // its live range used to cover the entire chunk, pinning the cp.async to the
  // last ~15% of the body.  The room comes from dropping the shared dt_f array:
  // its only consumer runs on the producer's thread, so it rides in a register.
  bf16* sX = (bf16*)(smem + 65536);
  bf16* sSb0 = (bf16*)(smem + 65536 + 2 * 128 * HD_BLK * 2);
  bf16* sSb1 = sSb0 + HD_BLK * 128;  // double-buffered [HD_BLK][128] state images
  float* sCum = (float*)(sSb1 + HD_BLK * 128);
  float* sSc = sCum + 256;
  float* sLoc = sSc + 256;
  constexpr bool EARLY_B = true;  // prefetch B as soon as the last state update is done
  constexpr bool WARP_COPY =
      (HD_BLK <= 32);  // warp-private y copy-out only pays on the small-HD path
  // The half-head path owns disjoint X rows per warp.  Reuse those rows for
  // the epilogue after the block-wide phase barrier: each warp copies its
  // output before restaging the same rows for the next chunk.  Keeping the
  // output in sSb0 would race the next iteration's cross-warp state publish.
  bf16* sScr = WARP_COPY ? sSb1 : (EARLY_B ? sSb0 : sB);

  const uint32_t aC = peer_sptr(sC), aB = peer_sptr(sB);
  const uint32_t aX0 = peer_sptr(sX), aX1 = peer_sptr(sX + 128 * HD_BLK);
  const uint32_t aScr = peer_sptr(sScr), aSb0 = peer_sptr(sSb0), aSb1 = peer_sptr(sSb1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int gid = lane >> 2, t4 = lane & 3;
  // Row block per warp. Warps map to SM quadrants as warp%4, and the two chunk
  // halves do different work in each phase, so both warps of a quadrant must sit
  // in the SAME half (and their tile counts must sum to 5) for every phase to be
  // balanced across quadrants, not just the chunk as a whole.
  const int qd = warp & 3, sx = warp >> 2;
  const int I = (qd >= 2 ? 4 : 0) + ((qd & 1) ? (sx ? 2 : 1) : (sx ? 3 : 0));
  const int HALF = I >> 2;
  const int WP = warp % P_BLOCKS, WN = warp / P_BLOCKS;

  const int h = blockIdx.x / HDSPLIT;
  const int pslot = blockIdx.x - h * HDSPLIT;
  const int gq = h / (HEADS >> 3);

  const float A_h = Ag[h] * (L2E ? 1.4426950408889634f : 1.0f);
  const float dtb_h = __bfloat162float(dtbg[h]);
  const float D_h = __bfloat162float(Dg[h]);

  // Large-magnitude escalation flag, mirroring the NH128 path's DT_SCAN
  // detection: a large step amplifies operand narrowing, and a large chunk
  // prefix swallows small later increments in the fp32 scan, so subtracted
  // cumulative values collapse later decays to exp(0) = 1.  Either condition
  // reroutes this block's own p-columns through precise_head at the end.
  // The flag rides in sSc's unused upper half (prolog_sc's 128 bf16 weights
  // fill only 64 of its 128 floats per buffer): the dynamic allocation is
  // already at the sm_120 per-block opt-in cap, so static __shared__ would
  // push cudaFuncSetAttribute over the limit.
  int* sPrec = reinterpret_cast<int*>(sSc + 64);
  if (tid == 0) *sPrec = 0;
  __syncthreads();

  constexpr size_t xstride = (size_t)HEADS * 64;
  const bf16* xbase = xg + (size_t)h * 64 + pslot * HD_BLK;
  bf16* ybase = yg + (size_t)h * 64 + pslot * HD_BLK;
  const bf16* bbase = Bg + (size_t)gq * 128;
  const bf16* cbase = Cg + (size_t)gq * 128;
  const bf16* dtbase = dtg + h;

  float Sacc[NT_S][4];
#pragma unroll
  for (int i = 0; i < NT_S; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) Sacc[i][j] = 0.f;

  // ---- pipelined prologue helpers ----
  unsigned short dtnext_bits = 0xff80u;
  // Padding is identified by the token-activity predicate, never inferred
  // from the dt payload: every finite bf16 dt value is a legal input.
  bool dtnext_active = false;
  float dtf_reg = 0.f;
  auto fetch_dt = [&](int cc) {
    if (tid < 128) {
      int t = cc * 128 + tid;
      bool active = FULL_T ? (cc < nchunks) : (t < T);
      dtnext_active = active;
      if (active) {
        const bf16* p = dtbase + (size_t)t * HEADS;
        asm volatile("ld.global.nc.L2::128B.b16 %0, [%1];\n" : "=h"(dtnext_bits) : "l"(p));
      } else {
        dtnext_bits = 0xff80u;
      }
    }
  };
  auto prolog_a = [&](int buf) {  // softplus + a, from prefetched dtnext
    if (tid < 128) {
      float dtnext = __uint_as_float((uint32_t)dtnext_bits << 16);
      float z = dtnext + dtb_h;
      float dtf = dtnext_active ? softplus_fast(z) : 0.f;
      // Atomic: up to 128 threads (and prolog_scan's lane 31) can escalate
      // concurrently; plain stores of the same value are still a data race.
      if (dtf > 4096.f) atomicOr(sPrec, 1);
      dtf_reg = dtf;
      sCum[buf * 128 + tid] = dtf * A_h;
    }
  };
  auto prolog_scan = [&](int buf) {
    if (warp == 2) {  // lightest warp in the phase where the scan runs (row block 4, 1 tile)
      float* p = sCum + buf * 128;
      float v[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) v[i] = p[lane * 4 + i];
      float s = (v[0] + v[1]) + (v[2] + v[3]);
      float pre = s;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        float t2 = __shfl_up_sync(0xffffffffu, pre, o);
        if (lane >= o) pre += t2;
      }
      float acc = pre - s;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        acc += v[i];
        p[lane * 4 + i] = acc;
      }
      // Lane 31's final acc is the whole chunk's cumulative log-decay.
      if (lane == 31 && fabsf(acc) > 65536.f) atomicOr(sPrec, 1);
    }
  };
  auto prolog_sc = [&](int buf) {
    const float* cm = sCum + buf * 128;
    if (tid < 128) {
      float ref = (tid < 64) ? cm[63] : cm[127];
      float sw = dtf_reg * dexp<L2E>(ref - cm[tid]);
      float sw_next = __shfl_down_sync(0xffffffffu, sw, 1);
      if (!(lane & 1))
        *reinterpret_cast<uint32_t*>(reinterpret_cast<bf16*>(sSc + buf * 128) + tid) =
            pack2(sw, sw_next);
      // Form every intra-chunk weight from its non-positive row-minus-
      // column exponent, retaining dt without a reciprocal-decay factor.
      sLoc[buf * 128 + tid] = dtf_reg;
    }
  };

  auto issue_C = [&](int cc) {  // warp-private rows -> no block barrier needed
    int cl = cc < nchunks ? cc : nchunks - 1;
    const bf16* cs = cbase + (size_t)(cl * 128) * 1024;
    int rem = T - cl * 128;
#pragma unroll
    for (int it = 0; it < 8; ++it) {
      int idx = it * 32 + lane;
      int row = 16 * I + (idx >> 4), unit = idx & 15;
      if constexpr (FULL_T) {
        peer_cpa16(addrU<16>(aC, row, unit), cs + row * 1024 + unit * 8, 16);
      } else {
        int nb = (row < rem) ? 16 : 0;
        int ts = (row < rem) ? row : 0;
        peer_cpa16(addrU<16>(aC, row, unit), cs + ts * 1024 + unit * 8, nb);
      }
    }
    peer_cpa_commit();
  };
  auto issue_B = [&](int cc) {
    int cl = cc < nchunks ? cc : nchunks - 1;
    const bf16* bs = bbase + (size_t)(cl * 128) * 1024;
    int rem = T - cl * 128;
#pragma unroll
    for (int it = 0; it < 8; ++it) {
      int idx = it * 256 + tid;
      int row = idx >> 4, unit = idx & 15;
      if constexpr (FULL_T) {
        peer_cpa16(addrU<16>(aB, row, unit), bs + row * 1024 + unit * 8, 16);
      } else {
        int nb = (row < rem) ? 16 : 0;
        int ts = (row < rem) ? row : 0;
        peer_cpa16(addrU<16>(aB, row, unit), bs + ts * 1024 + unit * 8, nb);
      }
    }
    peer_cpa_commit();
  };
  auto issue_X = [&](int cc) {
    int cl = cc < nchunks ? cc : nchunks - 1;
    const uint32_t aXd = (cc & 1) ? aX1 : aX0;
    const bf16* xs = xbase + (size_t)(cl * 128) * xstride;
    int rem = T - cl * 128;
#pragma unroll
    for (int it = 0; it < (16 * XU) / 32; ++it) {
      int idx = it * 32 + lane;
      int row = 16 * I + idx / XU;
      int unit = idx - (idx / XU) * XU;
      if constexpr (FULL_T) {
        peer_cpa16(addrU<XU>(aXd, row, unit), xs + (size_t)row * xstride + unit * 8, 16);
      } else {
        int nb = (row < rem) ? 16 : 0;
        int ts = (row < rem) ? row : 0;
        peer_cpa16(addrU<XU>(aXd, row, unit), xs + (size_t)ts * xstride + unit * 8, nb);
      }
    }
    peer_cpa_commit();
  };
  // The 16 half-head CTAs sharing a B/C group cooperatively cover one pair of
  // contiguous 128-byte cache lines per row, avoiding duplicate requests.
  const int cohort = (h & 7) * HDSPLIT + pslot;
  auto l2_prefetch = [&](int cc) {
    if (tid < 256 / (8 * HDSPLIT)) {
      const int line = cohort * (256 / (8 * HDSPLIT)) + tid;
      const int t = cc * 128 + (line >> 1);
      if (t < T) {
        const size_t off = (size_t)t * 1024 + (line & 1) * 64;
        asm volatile("prefetch.global.L2 [%0];\n" ::"l"(bbase + off));
        asm volatile("prefetch.global.L2 [%0];\n" ::"l"(cbase + off));
      }
    }
  };
  // The next chunk's prefix scan is published while the current chunk is
  // still computing.  Carry the derived decay scalars in registers so their
  // five exponentials and four shared loads are off the next chunk's critical
  // prologue.
  float cum63, cum127, carry_cI0, carry_u0, carry_u1;
  float carry_ebase, dec0, dec1;
  auto load_decay = [&](int buf) {
    const float* p = sCum + buf * 128;
    cum63 = p[63];
    cum127 = p[127];
    carry_cI0 = p[16 * I];
    const float href = (HALF == 0) ? 0.f : cum63;
    carry_u0 = dexp<L2E>(p[16 * I + gid] - carry_cI0);
    carry_u1 = dexp<L2E>(p[16 * I + gid + 8] - carry_cI0);
    carry_ebase = dexp<L2E>(carry_cI0 - href);
    dec0 = dexp<L2E>(cum63);
    dec1 = dexp<L2E>(cum127 - cum63);
  };
  issue_C(0);
  issue_X(0);
  issue_B(0);
  // Warm the two chunks that steady-state c+2 prefetching cannot reach, while
  // the initial async B/C/X copies are already in flight.
#pragma unroll
  for (int d = 1; d <= 2; ++d)
    if (d < nchunks) l2_prefetch(d);

  fetch_dt(0);
  prolog_a(0);
  __syncthreads();
  prolog_scan(0);
  fetch_dt(1);
  __syncthreads();
  prolog_sc(0);
  __syncthreads();
  if constexpr (!DIRECT_DECAY) load_decay(0);

  const int loop_chunks = ONE_CHUNK ? 1 : nchunks;
#pragma unroll 2
  for (int c = 0; c < loop_chunks; ++c) {
    const int t0 = c * 128;
    const int pb = c & 1, qb = pb ^ 1;
    const bool has_next = c + 1 < nchunks;
    const float* cm = sCum + pb * 128;
    const bf16* scv = reinterpret_cast<const bf16*>(sSc + pb * 128);
    const float* lcv = sLoc + pb * 128;

    if (has_next) prolog_a(qb);
    if (c + 2 < nchunks) {
      fetch_dt(c + 2);
      l2_prefetch(c + 2);
    }
    // The double-buffered X copy below already runs a full chunk ahead; a
    // separate row-wise L2 prefetch here duplicates requests in both half-head
    // CTAs and sits only a few instructions before the actual cp.async issue.

    // Long sequences benefit from recomputing these factors at use: removing
    // the eight loop-carried scalars relieves the NH64 register-latency cliff.
    // Shorter sequences retain the scan-shadow preload, where five extra
    // exponentials on the chunk path cost more than the register pressure.
    float local_c63, local_c127, cI0, u0, u1, ebase;
    if constexpr (DIRECT_DECAY) {
      local_c63 = cm[63];
      local_c127 = cm[127];
      cI0 = cm[16 * I];
      const float href = (HALF == 0) ? 0.f : local_c63;
      u0 = dexp<L2E>(cm[16 * I + gid] - cI0);
      u1 = dexp<L2E>(cm[16 * I + gid + 8] - cI0);
      ebase = dexp<L2E>(cI0 - href);
    } else {
      local_c63 = cum63;
      local_c127 = cum127;
      cI0 = carry_cI0;
      u0 = carry_u0;
      u1 = carry_u1;
      ebase = carry_ebase;
    }
    const float ec0 = u0 * ebase, ec1 = u1 * ebase;

    cpa_wait<2>();  // C(c) landed (this warp's own rows)
    __syncwarp();   // rows this lane reads were staged by other lanes of this warp
    uint32_t Cfrag[8][4];
#pragma unroll
    for (int k = 0; k < 8; ++k)
      peer_ldm4(Cfrag[k], addrU<16>(aC, 16 * I + (lane & 15), 2 * k + (lane >> 4)));

    float Yacc[NT_Y][4];
#pragma unroll
    for (int i = 0; i < NT_Y; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) Yacc[i][j] = 0.f;

    __syncwarp();  // all lanes done reading sC before this warp restages it
    if constexpr (!ONE_CHUNK) {
      issue_C(c + 1);  // overwrites only this warp's own C rows
      cpa_wait<1>();   // B(c), x(c) landed; C(c+1) may remain in flight
    } else {
      cpa_wait<0>();  // terminal chunk: wait for B(c) and x(c), no dummy C
    }
    const uint32_t aX = (c & 1) ? aX1 : aX0;
    if constexpr (!ONE_CHUNK)
      issue_X(c + 1);  // own rows into the other buffer: a full chunk of lead
    auto put_state = [&](uint32_t sbbase) {
#pragma unroll
      for (int nt = 0; nt < NT_S; nt += 2) {
        int n0 = WN * N_PART + nt * 8, p0 = 16 * WP;
        uint32_t o = addrU<16>(sbbase, p0 + (lane & 15), (n0 >> 3) + (lane >> 4));
        uint32_t v0 = pack2(Sacc[nt][0], Sacc[nt][1]);
        uint32_t v1 = pack2(Sacc[nt][2], Sacc[nt][3]);
        uint32_t v2 = pack2(Sacc[nt + 1][0], Sacc[nt + 1][1]);
        uint32_t v3 = pack2(Sacc[nt + 1][2], Sacc[nt + 1][3]);
        peer_stm4(o, v0, v1, v2, v3);
      }
    };
    // Accumulate carried-state output, then apply its row decay.
    auto y_off = [&](uint32_t sbbase) {
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        uint32_t sb[NT_Y / 2][4];
#pragma unroll
        for (int pp = 0; pp < NT_Y / 2; ++pp)
          peer_ldm4(sb[pp], addrU<16>(sbbase, 16 * pp + (lane & 15), 2 * k + (lane >> 4)));
#pragma unroll
        for (int pp = 0; pp < NT_Y / 2; ++pp) {
          mmaf(Yacc[2 * pp], Cfrag[k], sb[pp][0], sb[pp][2]);
          mmaf(Yacc[2 * pp + 1], Cfrag[k], sb[pp][1], sb[pp][3]);
        }
      }
#pragma unroll
      for (int nt = 0; nt < NT_Y; ++nt) {
        Yacc[nt][0] *= ec0;
        Yacc[nt][1] *= ec0;
        Yacc[nt][2] *= ec1;
        Yacc[nt][3] *= ec1;
      }
    };
    auto state_up = [&](int kb, float dec) {
      uint32_t sw[4][2];
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        sw[kk][0] = *(const uint32_t*)(scv + 16 * (kb + kk) + t4 * 2);
        sw[kk][1] = *(const uint32_t*)(scv + 16 * (kb + kk) + t4 * 2 + 8);
      }
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        int k = kb + kk;
        uint32_t xa[4];
        peer_ldm4t(xa, addrU<XU>(aX, 16 * k + (lane & 15), 2 * WP + (lane >> 4)));
        uint32_t bb[NT_S / 2][4];
#pragma unroll
        for (int nn = 0; nn < NT_S / 2; ++nn)
          peer_ldm4t(bb[nn], addrU<16>(aB, 16 * k + (lane & 15),
                                       (WN * N_PART + nn * 16) / 8 + (lane >> 4)));
        if (kk == 0) {
#pragma unroll
          for (int i = 0; i < NT_S; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) Sacc[i][j] *= dec;
        }
        uint32_t sa = sw[kk][0];
        uint32_t sb2 = sw[kk][1];
        uint32_t af[4];
        af[0] = scale2_packed(xa[0], sa);
        af[1] = scale2_packed(xa[2], sa);
        af[2] = scale2_packed(xa[1], sb2);
        af[3] = scale2_packed(xa[3], sb2);
#pragma unroll
        for (int nn = 0; nn < NT_S / 2; ++nn) {
          mmaf(Sacc[2 * nn], af, bb[nn][0], bb[nn][1]);
          mmaf(Sacc[2 * nn + 1], af, bb[nn][2], bb[nn][3]);
        }
      }
    };
    auto gy = [&]() {  // G + y_diag over this warp's triangular tiles
      auto apply_off = [&](int J, float(&ga)[2][4]) {
        uint32_t xb[NT_Y / 2][4];
#pragma unroll
        for (int pp = 0; pp < NT_Y / 2; ++pp)
          peer_ldm4t(xb[pp], addrU<XU>(aX, 16 * J + (lane & 15), 2 * pp + (lane >> 4)));
        uint32_t mf[4];
        const int jb = 16 * J;
        float2 lja = *(const float2*)(lcv + jb + t4 * 2);
        float2 ljb = *(const float2*)(lcv + jb + t4 * 2 + 8);
        float ca = cm[16 * I + gid], cb = cm[16 * I + gid + 8];
        float2 cja = *(const float2*)(cm + jb + t4 * 2);
        float2 cjb = *(const float2*)(cm + jb + t4 * 2 + 8);
        mf[0] = pack2(ga[0][0] * (lja.x * dexp<L2E>(ca - cja.x)),
                      ga[0][1] * (lja.y * dexp<L2E>(ca - cja.y)));
        mf[1] = pack2(ga[0][2] * (lja.x * dexp<L2E>(cb - cja.x)),
                      ga[0][3] * (lja.y * dexp<L2E>(cb - cja.y)));
        mf[2] = pack2(ga[1][0] * (ljb.x * dexp<L2E>(ca - cjb.x)),
                      ga[1][1] * (ljb.y * dexp<L2E>(ca - cjb.y)));
        mf[3] = pack2(ga[1][2] * (ljb.x * dexp<L2E>(cb - cjb.x)),
                      ga[1][3] * (ljb.y * dexp<L2E>(cb - cjb.y)));
#pragma unroll
        for (int pp = 0; pp < NT_Y / 2; ++pp) {
          mmaf(Yacc[2 * pp], mf, xb[pp][0], xb[pp][1]);
          mmaf(Yacc[2 * pp + 1], mf, xb[pp][2], xb[pp][3]);
        }
      };

      auto apply_diag = [&](int J, float(&ga)[2][4]) {
        uint32_t xb[NT_Y / 2][4];
#pragma unroll
        for (int pp = 0; pp < NT_Y / 2; ++pp)
          peer_ldm4t(xb[pp], addrU<XU>(aX, 16 * J + (lane & 15), 2 * pp + (lane >> 4)));
        const int jb = 16 * J;
        float2 lja = *(const float2*)(lcv + jb + t4 * 2);
        float2 ljb = *(const float2*)(lcv + jb + t4 * 2 + 8);
        int ja = t4 * 2, jbb = ja + 8;
        uint32_t mf[4];
        float ca = cm[16 * I + gid], cb = cm[16 * I + gid + 8];
        float2 cja = *(const float2*)(cm + jb + t4 * 2);
        float2 cjb = *(const float2*)(cm + jb + t4 * 2 + 8);
        mf[0] = pack2((ja <= gid) ? ga[0][0] * (lja.x * dexp<L2E>(ca - cja.x)) : 0.f,
                      (ja + 1 <= gid) ? ga[0][1] * (lja.y * dexp<L2E>(ca - cja.y)) : 0.f);
        mf[1] = pack2((ja <= gid + 8) ? ga[0][2] * (lja.x * dexp<L2E>(cb - cja.x)) : 0.f,
                      (ja + 1 <= gid + 8) ? ga[0][3] * (lja.y * dexp<L2E>(cb - cja.y)) : 0.f);
        mf[2] = pack2((jbb <= gid) ? ga[1][0] * (ljb.x * dexp<L2E>(ca - cjb.x)) : 0.f,
                      (jbb + 1 <= gid) ? ga[1][1] * (ljb.y * dexp<L2E>(ca - cjb.y)) : 0.f);
        mf[3] = pack2((jbb <= gid + 8) ? ga[1][2] * (ljb.x * dexp<L2E>(cb - cjb.x)) : 0.f,
                      (jbb + 1 <= gid + 8) ? ga[1][3] * (ljb.y * dexp<L2E>(cb - cjb.y)) : 0.f);
#pragma unroll
        for (int pp = 0; pp < NT_Y / 2; ++pp) {
          mmaf(Yacc[2 * pp], mf, xb[pp][0], xb[pp][1]);
          mmaf(Yacc[2 * pp + 1], mf, xb[pp][2], xb[pp][3]);
        }
      };

      auto gram_pair = [&](int J, float(&ga)[2][2][4]) {
        uint32_t ba[4], bb[4], na[4], nb[4];
        uint32_t bA0, bPX, bB0, bQX;
        addrUB_split<16>(aB, 16 * J, lane, bA0, bPX);
        addrUB_split<16>(aB, 16 * (J + 1), lane, bB0, bQX);
        peer_ldm4(ba, bA0 + bPX);
        peer_ldm4(bb, bB0 + bQX);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          if (k < 7) {
            peer_ldm4(na, bA0 + ((((uint32_t)k + 1) << 5) ^ bPX));
            peer_ldm4(nb, bB0 + ((((uint32_t)k + 1) << 5) ^ bQX));
          }
          mmaf(ga[0][0], Cfrag[k], ba[0], ba[1]);
          mmaf(ga[0][1], Cfrag[k], ba[2], ba[3]);
          mmaf(ga[1][0], Cfrag[k], bb[0], bb[1]);
          mmaf(ga[1][1], Cfrag[k], bb[2], bb[3]);
#pragma unroll
          if (k < 7)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              ba[q] = na[q];
              bb[q] = nb[q];
            }
        }
      };

      auto gram_one = [&](int J, float(&ga)[2][4]) {
        uint32_t bb[4], bn[4];
        uint32_t bA0, bPX;
        addrUB_split<16>(aB, 16 * J, lane, bA0, bPX);
        peer_ldm4(bb, bA0 + bPX);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          if (k < 7) peer_ldm4(bn, bA0 + ((((uint32_t)k + 1) << 5) ^ bPX));
          mmaf(ga[0], Cfrag[k], bb[0], bb[1]);
          mmaf(ga[1], Cfrag[k], bb[2], bb[3]);
#pragma unroll
          if (k < 7)
#pragma unroll
            for (int q = 0; q < 4; ++q) bb[q] = bn[q];
        }
      };

      const int J0 = HALF * 4;
      int J = J0;
#pragma unroll
      for (; J + 1 < I; J += 2) {
        float ga[2][2][4] = {};
        gram_pair(J, ga);
        apply_off(J, ga[0]);
        apply_off(J + 1, ga[1]);
      }
      if (J < I) {
        float ga[2][2][4] = {};
        gram_pair(J, ga);
        apply_off(J, ga[0]);
        apply_diag(I, ga[1]);
      } else {
        float ga[2][4] = {};
        gram_one(I, ga);
        apply_diag(I, ga);
      }
    };

    // ---- phase 1 ----
    if (c > 0) put_state(aSb0);
    __syncthreads();  // old state image visible + cp.async B/x visible
    if (has_next) prolog_scan(qb);
    if constexpr (DIRECT_DECAY)
      state_up(0, dexp<L2E>(local_c63));
    else
      state_up(0, dec0);

    // ---- phase 2 ----
    put_state(aSb1);
    // Warps 4/5 do not consume the next scan. Let them enter their half's
    // carried-state work while the six consumer warps wait for the state.
    if (warp == 4 || warp == 5)
      asm volatile("bar.arrive 1, 256;" ::: "memory");
    else
      asm volatile("bar.sync 1, 256;" ::: "memory");
    if (has_next) prolog_sc(qb);
    if (HALF == 0) {
      if (c > 0) y_off(aSb0);
      gy();
    } else {
      y_off(aSb1);
      gy();
    }
    if constexpr (!ONE_CHUNK) {
      if constexpr (DIRECT_DECAY)
        state_up(4, dexp<L2E>(local_c127 - local_c63));
      else
        state_up(4, dec1);
    }

    // ---------------- epilogue ----------------
    __syncthreads();
    if (EARLY_B && !ONE_CHUNK) issue_B(c + 1);  // B buffer free from here
#pragma unroll
    for (int nt = 0; nt < NT_Y; nt += 2) {
      int row0 = 16 * I + gid, row1 = row0 + 8;
      uint32_t out[4];
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        uint32_t xv0, xv1;
        asm volatile("ld.shared.b32 %0, [%1];\n"
                     : "=r"(xv0)
                     : "r"(addrU<XU>(aX, row0, nt + q) + t4 * 4));
        asm volatile("ld.shared.b32 %0, [%1];\n"
                     : "=r"(xv1)
                     : "r"(addrU<XU>(aX, row1, nt + q) + t4 * 4));
        float2 f0 = __bfloat1622float2(punned<__nv_bfloat162>(xv0));
        float2 f1 = __bfloat1622float2(punned<__nv_bfloat162>(xv1));
        out[2 * q] = pack2(Yacc[nt + q][0] + D_h * f0.x, Yacc[nt + q][1] + D_h * f0.y);
        out[2 * q + 1] = pack2(Yacc[nt + q][2] + D_h * f1.x, Yacc[nt + q][3] + D_h * f1.y);
      }
      uint32_t o = addrU<XU>(aScr, 16 * I + (lane & 15), nt + (lane >> 4));
      peer_stm4(o, out[0], out[1], out[2], out[3]);
    }
    if (WARP_COPY) {
      __syncwarp();  // all lanes finished reading the old X rows
      const int unit = lane & 3;
      const int row0 = 16 * I + (lane >> 2), row1 = row0 + 8;
      const int ot0 = t0 + row0, ot1 = t0 + row1;
      uint4 v0, v1;
      bool out0, out1;
      if constexpr (FULL_T) {
        out0 = out1 = true;
      } else {
        out0 = ot0 < T;
        out1 = ot1 < T;
      }
      if (out0)
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(v0.x), "=r"(v0.y), "=r"(v0.z), "=r"(v0.w)
                     : "r"(addrU<XU>(aScr, row0, unit)));
      if (out1)
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(v1.x), "=r"(v1.y), "=r"(v1.z), "=r"(v1.w)
                     : "r"(addrU<XU>(aScr, row1, unit)));
      if (out0) *(uint4*)(ybase + (size_t)ot0 * xstride + unit * 8) = v0;
      if (out1) *(uint4*)(ybase + (size_t)ot1 * xstride + unit * 8) = v1;
    } else {
      __syncthreads();
#pragma unroll
      for (int it = 0; it < XU / 2; ++it) {
        int idx = it * 256 + tid;
        int row = idx / XU, unit = idx - row * XU;
        int t = t0 + row;
        if (t < T) {
          uint4 v;
          asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];\n"
                       : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                       : "r"(addrU<XU>(aScr, row, unit)));
          *(uint4*)(ybase + (size_t)t * xstride + unit * 8) = v;
        }
      }
    }
    if (!EARLY_B) issue_B(c + 1);  // y staging lived in the B buffer
    if constexpr (!DIRECT_DECAY)
      if (!ONE_CHUNK && has_next) load_decay(qb);
  }
  // Rare large-magnitude replay, the same escalation the NH128 path takes
  // through regx's DT_SCAN flag: rewrite this block's own p-columns with the
  // fp32 chunk formula.  Both half-head CTAs trip on the same dt stream, so
  // the whole head is rewritten consistently, each block within its own
  // disjoint columns (no cross-CTA write ordering is needed).
  // Drain this thread's outstanding cp.async groups first: the final
  // iteration's dummy C/X/B copies may still be in flight, and __syncthreads
  // alone does not order cp.async completion against the replay's reuse of
  // the same shared memory.
  cpa_wait<0>();
  __syncthreads();
  if (*sPrec) {
    precise_head(xg, dtg, Ag, Bg, Cg, Dg, dtbg, yg, T, NH, h, pslot * HD_BLK, HD_BLK, smem);
  }
}

// Atomic: the Python wrapper does not serialize calls, so two host threads
// can reach the first-call attribute setup concurrently.  The setup itself is
// idempotent; the atomic only removes the unsynchronized int data race.
static std::atomic<int> g_init[2] = {0, 0};

void RunPeer(ffi::TensorView x, ffi::TensorView dt, ffi::TensorView A, ffi::TensorView Bmat,
             ffi::TensorView Cmat, ffi::TensorView D, ffi::TensorView dt_bias, ffi::TensorView y) {
  const int T = (int)x.shape()[0];
  const int NH = (int)x.shape()[1];
  const int nchunks = (T + 127) / 128;
  auto dev = x.device();
  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(dev.device_type, dev.device_id);

  const bf16* xp = (const bf16*)x.data_ptr();
  const bf16* dtp = (const bf16*)dt.data_ptr();
  const float* Ap = (const float*)A.data_ptr();
  const bf16* Bp = (const bf16*)Bmat.data_ptr();
  const bf16* Cp = (const bf16*)Cmat.data_ptr();
  const bf16* Dp = (const bf16*)D.data_ptr();
  const bf16* dbp = (const bf16*)dt_bias.data_ptr();
  bf16* yp = (bf16*)y.data_ptr();

  if (NH >= 96) {
    constexpr int SH = 65536 + 128 * 64 * 2 + 2 * 64 * 128 * 2 + 4 * 2 * 128 * 4;
    if (!g_init[0]) {
      TVM_FFI_CHECK_CUDA_ERROR(
          cudaFuncSetAttribute(ssd_kernel<64>, cudaFuncAttributeMaxDynamicSharedMemorySize, SH));
      g_init[0] = 1;
    }
    ssd_kernel<64><<<NH, 256, SH, stream>>>(xp, dtp, Ap, Bp, Cp, Dp, dbp, yp, T, NH, nchunks);
  } else {
    constexpr int SH = 65536 + 2 * 128 * 32 * 2 + 2 * 32 * 128 * 2 + 3 * 2 * 128 * 4;
    if (!g_init[1]) {
      TVM_FFI_CHECK_CUDA_ERROR(
          cudaFuncSetAttribute(ssd_kernel<32>, cudaFuncAttributeMaxDynamicSharedMemorySize, SH));
      TVM_FFI_CHECK_CUDA_ERROR(cudaFuncSetAttribute(
          ssd_kernel<32, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SH));
      TVM_FFI_CHECK_CUDA_ERROR(cudaFuncSetAttribute(
          ssd_kernel<32, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SH));
      TVM_FFI_CHECK_CUDA_ERROR(cudaFuncSetAttribute(
          ssd_kernel<32, true, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SH));
      TVM_FFI_CHECK_CUDA_ERROR(cudaFuncSetAttribute(
          ssd_kernel<32, false, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SH));
      g_init[1] = 1;
    }
    if (T <= 128)
      ssd_kernel<32, false, true>
          <<<NH * 2, 256, SH, stream>>>(xp, dtp, Ap, Bp, Cp, Dp, dbp, yp, T, NH, nchunks);
    else if (T >= 12288) {
      if ((T & 127) == 0)
        ssd_kernel<32, true, false, true>
            <<<NH * 2, 256, SH, stream>>>(xp, dtp, Ap, Bp, Cp, Dp, dbp, yp, T, NH, nchunks);
      else
        ssd_kernel<32, false, false, true>
            <<<NH * 2, 256, SH, stream>>>(xp, dtp, Ap, Bp, Cp, Dp, dbp, yp, T, NH, nchunks);
    } else if ((T & 127) == 0)
      ssd_kernel<32, true>
          <<<NH * 2, 256, SH, stream>>>(xp, dtp, Ap, Bp, Cp, Dp, dbp, yp, T, NH, nchunks);
    else
      ssd_kernel<32><<<NH * 2, 256, SH, stream>>>(xp, dtp, Ap, Bp, Cp, Dp, dbp, yp, T, NH, nchunks);
  }
  TVM_FFI_CHECK_CUDA_ERROR(cudaGetLastError());
}

#undef DEVI

namespace regx {
// Mamba2 SSD chunk-scan prefill, sm120 (RTX PRO 6000 / GB202)
// One block per (head, p-split).  Sequential over 64-token chunks; the SSM
// state lives in registers as MMA accumulators, so no per-chunk state array
// is ever written to global memory.  MMA operands are bf16 with fp32
// accumulation, matching the operation's required precision class.

#define DI __device__ __forceinline__

static constexpr int QT = 64;    // tokens per internal chunk
static constexpr int DSN = 128;  // state dim
static constexpr int HDP = 64;   // head dim
static constexpr int NGR = 8;    // groups
static constexpr int LDN = 136;  // padded stride for 128-col rows
// Ten causal tiles, each with two row-major 16x16 BF16 planes.
static constexpr int SCORE_WORDS = 10 * 2 * 16 * 16;
DI int score_tile(int mi, int kj) { return (mi * (mi + 1) / 2 + kj) * 512; }
DI int score_index(int i, int j) { return score_tile(i >> 4, j >> 4) + (i & 15) * 16 + (j & 15); }
DI void stage_pair(bf16* dst, float a, float b) {
  const __nv_bfloat162 hi = __floats2bfloat162_rn(a, b);
  const float2 f = __bfloat1622float2(hi);
  *reinterpret_cast<__nv_bfloat162*>(dst) = hi;
  *reinterpret_cast<__nv_bfloat162*>(dst + 256) =
      __floats2bfloat162_rn(__fsub_rn(a, f.x), __fsub_rn(b, f.y));
}

DI uint32_t sptr(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

template <int OFF>
DI void ldm4o(uint32_t (&d)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4+%5];"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(a), "n"(OFF));
}
template <int OFF>
DI void ldm4to(uint32_t (&d)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4+%5];"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(a), "n"(OFF));
}
DI void ldm4(uint32_t (&d)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(a));
}
DI void ldm4t(uint32_t (&d)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(a));
}
DI void ldm2(uint32_t (&d)[2], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];"
               : "=r"(d[0]), "=r"(d[1])
               : "r"(a));
}
template <int OFF>
DI void ldm2o(uint32_t (&d)[2], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2+%3];"
               : "=r"(d[0]), "=r"(d[1])
               : "r"(a), "n"(OFF));
}
DI void ldm2t(uint32_t (&d)[2], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];"
               : "=r"(d[0]), "=r"(d[1])
               : "r"(a));
}
// Maximum magnitude of eight packed bf16 operands. Map NaNs to infinity so
// a nonfinite fast-path result requests replay rather than certifying itself.
DI float peak8(int4 v) {
  uint32_t m = __vmaxu2((uint32_t)v.x & 0x7fff7fffu, (uint32_t)v.y & 0x7fff7fffu);
  uint32_t n = __vmaxu2((uint32_t)v.z & 0x7fff7fffu, (uint32_t)v.w & 0x7fff7fffu);
  m = __vmaxu2(m, n);
  return __uint_as_float(min(max(m >> 16, m & 0xffffu), 0x7f80u) << 16);
}

DI uint32_t pack2f(float a, float b) { return punned<uint32_t>(__floats2bfloat162_rn(a, b)); }
DI void mma_f32(float* d, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
DI void cpa16(uint32_t dst, const void* src, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" ::"r"(dst), "l"(src), "r"(bytes));
}
DI void cpa_commit() { asm volatile("cp.async.commit_group;"); }
DI void cpa_wait() { asm volatile("cp.async.wait_group 0;"); }

DI void tma_barrier_init(uint64_t* bar) {
  uint32_t p = sptr(bar);
  asm volatile(
      "mbarrier.init.shared::cta.b64 [%0], 1;\n"
      "fence.proxy.async.shared::cta;\n" ::"r"(p)
      : "memory");
}
DI void barrier_init(uint64_t* bar, int count) {
  uint32_t p = sptr(bar);
  asm volatile(
      "mbarrier.init.shared::cta.b64 [%0], %1;\n"
      "fence.proxy.async.shared::cta;\n" ::"r"(p),
      "r"(count)
      : "memory");
}
DI void barrier_arrive(uint64_t* bar) {
  uint32_t p = sptr(bar);
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n" ::"r"(p) : "memory");
}
DI void tma_expect(uint64_t* bar, uint32_t bytes) {
  uint32_t p = sptr(bar);
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n" ::"r"(p),
               "r"(bytes)
               : "memory");
}
DI void tma_wait(uint64_t* bar, int phase) {
  uint32_t p = sptr(bar);
  uint32_t done;
  do {
    asm volatile(
        "{ .reg .pred q;\n"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 q, [%1], %2;\n"
        "selp.u32 %0, 1, 0, q; }\n"
        : "=r"(done)
        : "r"(p), "r"(phase)
        : "memory");
  } while (!done);
}
DI void tma_load_2d(bf16* dst, const CUtensorMap* map, uint64_t* bar, int col, int row) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3}], [%4];\n" ::"r"(sptr(dst)),
      "l"((uint64_t)map), "r"(col), "r"(row), "r"(sptr(bar))
      : "memory");
}
DI void prefetch_tma(const CUtensorMap* map) {
  asm volatile("prefetch.const.tensormap [%0];\n" ::"l"((uint64_t)map));
}
DI void tma_prefetch_2d(const CUtensorMap* map, int col, int row) {
  asm volatile(
      "cp.async.bulk.prefetch.tensor.2d.L2.global "
      "[%0, {%1, %2}];\n" ::"l"((uint64_t)map),
      "r"(col), "r"(row)
      : "memory");
}
DI float softplus_f(float z) {
  return z > 20.f ? z : (z < -5.f ? __expf(z) : __logf(1.f + __expf(z)));
}

// two packed bf16 -> two packed fp16 (exact for every value in range)
DI uint32_t bf2h2(uint32_t w) {
  float2 f;
  f.x = __uint_as_float(w << 16);
  f.y = __uint_as_float(w & 0xffff0000u);
  __half2 h = __float22half2_rn(f);
  return punned<uint32_t>(h);
}
DI int4 bf2h2_v(int4 v) {
  int4 r;
  r.x = (int)bf2h2((uint32_t)v.x);
  r.y = (int)bf2h2((uint32_t)v.y);
  r.z = (int)bf2h2((uint32_t)v.z);
  r.w = (int)bf2h2((uint32_t)v.w);
  return r;
}

template <int LD, class TT>
DI uint32_t addr4(const TT* base, int r0, int k0, int lane) {
  int r = r0 + 8 * ((lane >> 3) & 1) + (lane & 7);
  int k = k0 + 8 * (lane >> 4);
  return sptr(base + r * LD + k);
}
template <int LD, class TT>
DI uint32_t addr4t(const TT* base, int r0, int k0, int lane) {
  int sr = k0 + 8 * (lane >> 4) + (lane & 7);
  int sc = r0 + 8 * ((lane >> 3) & 1);
  return sptr(base + sr * LD + sc);
}
template <int LD, class TT>
DI uint32_t addr4b(const TT* base, int r0, int k0, int lane) {
  int q = lane >> 3;
  int r = r0 + 8 * (q >> 1) + (lane & 7);
  int k = k0 + 8 * (q & 1);
  return sptr(base + r * LD + k);
}
template <int LD, class TT>
DI uint32_t addr4bt(const TT* base, int r0, int k0, int lane) {
  int q = lane >> 3;
  int sr = k0 + 8 * (q & 1) + (lane & 7);
  int sc = r0 + 8 * (q >> 1);
  return sptr(base + sr * LD + sc);
}
template <int LD, class TT>
DI uint32_t addr2(const TT* base, int r0, int k0, int lane) {
  int r = r0 + (lane & 7);
  int k = k0 + 8 * ((lane >> 3) & 1);
  return sptr(base + r * LD + k);
}
template <int LD, class TT>
DI uint32_t addr2t(const TT* base, int r0, int k0, int lane) {
  int sr = k0 + 8 * ((lane >> 3) & 1) + (lane & 7);
  return sptr(base + sr * LD + r0);
}

template <int NSN, int LDX>
DI void update_state_tile(float (&accSt)[NSN][4], const __nv_bfloat16* sX,
                          const __nv_bfloat16* bCur, const __nv_bfloat16* sWt, const float* sEcum,
                          int pm, int nn, int lane) {
  float etot = sEcum[QT - 1];
#pragma unroll
  for (int n = 0; n < NSN; ++n)
#pragma unroll
    for (int r = 0; r < 4; ++r) accSt[n][r] *= etot;
  const uint32_t aXb = addr4t<LDX>(sX, 16 * pm, 0, lane);
  const uint32_t bNb = addr4bt<LDN>(bCur, 8 * NSN * nn, 0, lane);
#define PHB_TILE(kt)                                           \
  {                                                            \
    uint32_t aX[4];                                            \
    ldm4to<32 * (kt) * LDX>(aX, aXb);                          \
    uint32_t wp[2];                                            \
    ldm2(wp, sptr(sWt + 16 * (kt) + 8 * ((lane >> 3) & 1)));   \
    uint32_t wa = wp[0], wb = wp[1];                           \
    aX[0] = scale2_packed(aX[0], wa);                          \
    aX[1] = scale2_packed(aX[1], wa);                          \
    aX[2] = scale2_packed(aX[2], wb);                          \
    aX[3] = scale2_packed(aX[3], wb);                          \
    _Pragma("unroll") for (int u = 0; u < NSN / 2; ++u) {      \
      uint32_t bN[4];                                          \
      if (u == 0)                                              \
        ldm4to<32 * (kt) * LDN>(bN, bNb);                      \
      else                                                     \
        ldm4to<32 * (kt) * LDN + 32>(bN, bNb);                 \
      uint32_t q0[2] = {bN[0], bN[1]}, q1[2] = {bN[2], bN[3]}; \
      mma_f32(accSt[2 * u + 0], aX, q0);                       \
      mma_f32(accSt[2 * u + 1], aX, q1);                       \
    }                                                          \
  }
  PHB_TILE(0) PHB_TILE(1) PHB_TILE(2) PHB_TILE(3)
#undef PHB_TILE
}

template <int NSN, int LDX>
DI void update_state_tile2(float (&accSt)[2][NSN][4], const __nv_bfloat16* sX,
                           const __nv_bfloat16* bCur, const __nv_bfloat16* sWt, const float* sEcum,
                           int pm0, int nn, int lane) {
  float etot = sEcum[QT - 1];
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int n = 0; n < NSN; ++n)
#pragma unroll
      for (int r = 0; r < 4; ++r) accSt[m][n][r] *= etot;
  const uint32_t aXb0 = addr4t<LDX>(sX, 16 * pm0, 0, lane);
  const uint32_t aXb1 = addr4t<LDX>(sX, 16 * pm0 + 16, 0, lane);
  const uint32_t bNb = addr4bt<LDN>(bCur, 8 * NSN * nn, 0, lane);
#define PHB2_TILE(kt)                                          \
  {                                                            \
    uint32_t aX0[4], aX1[4];                                   \
    ldm4to<32 * (kt) * LDX>(aX0, aXb0);                        \
    ldm4to<32 * (kt) * LDX>(aX1, aXb1);                        \
    uint32_t wp[2];                                            \
    ldm2(wp, sptr(sWt + 16 * (kt) + 8 * ((lane >> 3) & 1)));   \
    uint32_t wa = wp[0], wb = wp[1];                           \
    aX0[0] = scale2_packed(aX0[0], wa);                        \
    aX0[1] = scale2_packed(aX0[1], wa);                        \
    aX0[2] = scale2_packed(aX0[2], wb);                        \
    aX0[3] = scale2_packed(aX0[3], wb);                        \
    aX1[0] = scale2_packed(aX1[0], wa);                        \
    aX1[1] = scale2_packed(aX1[1], wa);                        \
    aX1[2] = scale2_packed(aX1[2], wb);                        \
    aX1[3] = scale2_packed(aX1[3], wb);                        \
    _Pragma("unroll") for (int u = 0; u < NSN / 2; ++u) {      \
      uint32_t bN[4];                                          \
      if (u == 0)                                              \
        ldm4to<32 * (kt) * LDN>(bN, bNb);                      \
      else                                                     \
        ldm4to<32 * (kt) * LDN + 32>(bN, bNb);                 \
      uint32_t q0[2] = {bN[0], bN[1]}, q1[2] = {bN[2], bN[3]}; \
      mma_f32(accSt[0][2 * u + 0], aX0, q0);                   \
      mma_f32(accSt[0][2 * u + 1], aX0, q1);                   \
      mma_f32(accSt[1][2 * u + 0], aX1, q0);                   \
      mma_f32(accSt[1][2 * u + 1], aX1, q1);                   \
    }                                                          \
  }
  PHB2_TILE(0) PHB2_TILE(1) PHB2_TILE(2) PHB2_TILE(3)
#undef PHB2_TILE
}

// ---------------------------------------------------------------------------
// Group-level Gram producer.
//
// B and C are shared by the NH/NG = 16 heads of a group, so the raw score
// block G = C . B^T is identical for all 16 head CTAs of that group: they used
// to recompute it 16 times (21% of every head CTA's MMA work plus the whole
// B-operand ldmatrix stream).  One extra CTA per group now computes it once on
// an otherwise idle SM (128 head CTAs on a 188-SM part) and publishes the ten
// causal 16x16 tiles.
//
// A tile is stored in the exact register order an mma.m16n8k16 A-operand
// wants, so a consumer lane picks its eight values up with one 16-byte load
// and feeds them straight into the decay scaling: no shared-memory staging
// and no relayout on either side.
static constexpr int GTILES = 10;     // causal 16x16 tiles of a 64x64 block
static constexpr int GTILE_BF = 256;  // bf16 values per tile (32 lanes x 8)
static constexpr int GWIN = 256;      // ring depth, in 64-row windows
// Publish the producer accumulator without a BF16 round trip: a residual
// computed from an already rounded raw Gram cannot recover that first loss.
// Same slots, watermarks and lifetime, now 20 MiB rather than 10 MiB.
__device__ __align__(16) float g_gram[GWIN * NGR * GTILES * GTILE_BF];
__device__ unsigned int g_gflag[NGR * 32];        // one 128B publication line per group
__device__ unsigned long long g_gprog[NGR * 16];  // windows retired, per consumer
__device__ unsigned int g_gdone[NGR];             // consumers finished, per group
// tile index ti = mi*(mi+1)/2 + gj, two bits per entry
#define GMI_TAB 0xffa94u
#define GGJ_TAB 0xe4910u

DI void gst_relaxed(unsigned long long* p, unsigned long long v) {
  asm volatile("st.relaxed.gpu.global.b64 [%0], %1;\n" ::"l"(p), "l"(v) : "memory");
}
DI void gst_release(unsigned long long* p, unsigned long long v) {
  asm volatile("st.release.gpu.global.b64 [%0], %1;\n" ::"l"(p), "l"(v) : "memory");
}
DI unsigned long long gld_acquire(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.acquire.gpu.global.b64 %0, [%1];\n" : "=l"(v) : "l"(p) : "memory");
  return v;
}
DI void gst_release(unsigned int* p, unsigned int v) {
  asm volatile("st.release.gpu.global.b32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory");
}
DI unsigned int gld_acquire(const unsigned int* p) {
  unsigned int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}
DI void gld_cg128(uint4& v, const void* p) {
  asm volatile("ld.global.cg.v4.b32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
}
DI void gld_cg64(uint2& v, const void* p) {
  asm volatile("ld.global.cg.v2.b32 {%0,%1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "l"(p) : "memory");
}
DI void gld_cg128_l2(uint4& v, const void* p) {
  asm volatile("ld.global.cg.L2::128B.v4.b32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
}
DI void gld_cg64_l2(uint2& v, const void* p) {
  asm volatile("ld.global.cg.L2::128B.v2.b32 {%0,%1}, [%2];\n"
               : "=r"(v.x), "=r"(v.y)
               : "l"(p)
               : "memory");
}
DI void gst_cs128(void* p, const int4& v) {
  asm volatile("st.global.cs.v4.b32 [%0], {%1,%2,%3,%4};\n" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z),
               "r"(v.w)
               : "memory");
}

// The last producer window may contain fewer than 64 logical rows.  Stage it
// with guarded cp.async zero-fill so every nonzero transaction stays inside
// the public B/C tensors while the MMA still sees a complete zero-padded tile.
__device__ __forceinline__ void gram_tail_load(__nv_bfloat16* bCur, __nv_bfloat16* cCur,
                                               const __nv_bfloat16* gB, const __nv_bfloat16* gC,
                                               int g, int row0, int T) {
  constexpr int VEC_PER_ROW = DSN / 8;
  constexpr int NVEC = QT * VEC_PER_ROW;
  for (int idx = threadIdx.x; idx < NVEC; idx += blockDim.x) {
    const int r = idx / VEC_PER_ROW;
    const int cc = idx - r * VEC_PER_ROW;
    const int t = row0 + r;
    const int bytes = (t < T) ? 16 : 0;
    const size_t off = (size_t)(t < T ? t : 0) * (NGR * DSN) + (size_t)g * DSN + cc * 8;
    cpa16(sptr(bCur + r * LDN + cc * 8), gB + off, bytes);
    cpa16(sptr(cCur + r * LDN + cc * 8), gC + off, bytes);
  }
  cpa_commit();
  cpa_wait();
  __syncthreads();
}

// One producer CTA per group.  It walks 64-row windows of B/C (two windows per
// 128-token chunk on the NH64 path, one per chunk on the NH128 path) and emits
// the ten causal tiles of each window.
template <int NWP, bool SPLIT_DIAG>
__device__ __forceinline__ void gram_producer(const CUtensorMap* tmaB, const CUtensorMap* tmaC,
                                              const __nv_bfloat16* gB, const __nv_bfloat16* gC,
                                              int g, int nwin, int T) {
  extern __shared__ __align__(128) char smem_raw[];
  __nv_bfloat16* sP = (__nv_bfloat16*)smem_raw;  // [2][2][QT][LDN]
  uint64_t* bar = (uint64_t*)(sP + 4 * QT * LDN);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int bc_shift = (g == NGR - 1) ? (LDN - DSN) : 0;
  const int bc_col = g * DSN - bc_shift;
  const int nfull = T / QT;
  constexpr int TBYTES = 2 * QT * LDN * (int)sizeof(__nv_bfloat16);

  if (tid == 0 && nfull > 0) {
    prefetch_tma(tmaB);
    prefetch_tma(tmaC);
    tma_barrier_init(&bar[0]);
    tma_barrier_init(&bar[1]);
  }
  __syncthreads();
  // A sequence shorter than one window has no full window at all: the two
  // mbarriers above are then never initialized, so the prologue transactions
  // must be skipped as well.  Every window of such a launch is a partial one
  // and is staged by the guarded cp.async path below instead.
  if (tid == 0 && nfull > 0) {
    tma_expect(&bar[0], TBYTES);
    tma_load_2d(sP, tmaB, &bar[0], bc_col, 0);
    tma_load_2d(sP + QT * LDN, tmaC, &bar[0], bc_col, 0);
    if (nfull > 1) {
      tma_expect(&bar[1], TBYTES);
      tma_load_2d(sP + 2 * QT * LDN, tmaB, &bar[1], bc_col, QT);
      tma_load_2d(sP + 3 * QT * LDN, tmaC, &bar[1], bc_col, QT);
    }
  }

  for (int c = 0; c < nwin; ++c) {
    const int b = c & 1;
    __nv_bfloat16* bCur = sP + 2 * b * QT * LDN + bc_shift;
    __nv_bfloat16* cCur = bCur + QT * LDN;
    if (c < nfull)
      tma_wait(&bar[b], (c >> 1) & 1);
    else
      gram_tail_load(bCur, cCur, gB, gC, g, c * QT, T);
    if (c >= GWIN) {  // do not lap the slowest head CTA of this group
      unsigned long long need = (unsigned long long)(c - GWIN + 9);
      if (warp == 0 && lane < 16) {
        while (gld_acquire(g_gprog + g * 16 + lane) < need) {
        }
      }
      __syncthreads();
    }
#pragma unroll
    for (int tw = warp; tw < GTILES; tw += NWP) {
      const int mi = (int)((GMI_TAB >> (2 * tw)) & 3u);
      const int gj = (int)((GGJ_TAB >> (2 * tw)) & 3u);
      float accS[2][4] = {};
      const uint32_t aCb = addr4<LDN>(cCur, 16 * mi, 0, lane);
      const uint32_t bSb = addr4b<LDN>(bCur, 16 * gj, 0, lane);
      uint32_t aC[4], bS[4];
      ldm4o<0>(aC, aCb);
      ldm4o<0>(bS, bSb);
#define GP_MMA()                                             \
  {                                                          \
    uint32_t f0[2] = {bS[0], bS[1]}, f1[2] = {bS[2], bS[3]}; \
    mma_f32(accS[0], aC, f0);                                \
    mma_f32(accS[1], aC, f1);                                \
  }
#define GP_STEP(nxt)                                \
  {                                                 \
    uint32_t aCn[4], bSn[4];                        \
    ldm4o<32 * (nxt)>(aCn, aCb);                    \
    ldm4o<32 * (nxt)>(bSn, bSb);                    \
    GP_MMA()                                        \
    _Pragma("unroll") for (int z = 0; z < 4; ++z) { \
      aC[z] = aCn[z];                               \
      bS[z] = bSn[z];                               \
    }                                               \
  }
      GP_STEP(1)
      GP_STEP(2)
      GP_STEP(3)
      GP_STEP(4)
      GP_STEP(5)
      GP_STEP(6)
      GP_STEP(7)
      GP_MMA()
#undef GP_STEP
#undef GP_MMA
      const float4 lo = make_float4(accS[0][0], accS[0][1], accS[0][2], accS[0][3]);
      const float4 hi = make_float4(accS[1][0], accS[1][1], accS[1][2], accS[1][3]);
      float* gout =
          g_gram +
          ((((size_t)g * GTILES + (size_t)tw) * GWIN + (size_t)(c & (GWIN - 1))) * GTILE_BF);
      // The first two diagonal tiles are consumed as independent low/high
      // halves. Store all tiles as two dense half tiles so each 16B lane
      // transaction coalesces, whether the reader needs one half or both.
      *(float4*)(gout + lane * 4) = lo;
      *(float4*)(gout + 128 + lane * 4) = hi;
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) gst_release(g_gflag + g * 32, (unsigned int)(c + 1));
    if (tid == 0 && c + 2 < nfull) {
      tma_expect(&bar[b], TBYTES);
      tma_load_2d(sP + 2 * b * QT * LDN, tmaB, &bar[b], bc_col, (c + 2) * QT);
      tma_load_2d(sP + (2 * b + 1) * QT * LDN, tmaC, &bar[b], bc_col, (c + 2) * QT);
    }
  }

  // Retire this group's ring after its consumers finish reading it.  The
  // terminal grid rendezvous below keeps all CTAs resident until every group
  // has returned the protocol to the zero state required by the next launch.
  if (tid == 0) {
    while (gld_acquire(g_gdone + g) < 16u) __nanosleep(128);
    g_gdone[g] = 0u;
    g_gflag[g * 32] = 0u;
#pragma unroll 1
    for (int i = 0; i < 16; ++i) g_gprog[g * 16 + i] = 0ull;
  }
}

}  // namespace regx

// -------------------------------------------------------------------------
// Rare large-step replay of the same 128-token SSD formula for one head's
// p-columns [p0, p0+pcnt).  Keep the intermediates in fp32 rather than
// narrowing Gram/weighted-x/state operands to bf16 when that narrowing can
// dominate the absolute error bound.
__device__ __noinline__ void precise_head(const __nv_bfloat16* x, const __nv_bfloat16* dt,
                                          const float* A, const __nv_bfloat16* B,
                                          const __nv_bfloat16* C, const __nv_bfloat16* D,
                                          const __nv_bfloat16* bias, __nv_bfloat16* y, int T,
                                          int NH, int h, int p0, int pcnt, char* scratch) {
  // At pcnt=64: 99,328 bytes, within the PT=64 consumer's existing
  // 99,352-byte allocation.  At the NH64 kernels' pcnt=32: 82,944 bytes,
  // within their 101,376-byte allocation.
  float* state = (float*)scratch;     // pcnt x 128
  float* score = state + pcnt * 128;  // 128 x 128
  float* steps = score + 128 * 128;
  float* prefix = steps + 128;
  const int tid = threadIdx.x, g = h / (NH / 8);
  const float ah = A[h], db = __bfloat162float(bias[h]);
  const float dh = __bfloat162float(D[h]);
  for (int q = tid; q < pcnt * 128; q += blockDim.x) state[q] = 0.f;
  __syncthreads();
  for (int start = 0; start < T; start += 128) {
    int count = min(128, T - start);
    if (tid < count) {
      float z = __bfloat162float(dt[(size_t)(start + tid) * NH + h]) + db;
      float d = z > 20.f ? z : __nv_log1pf(__nv_expf(z));
      steps[tid] = d;
      prefix[tid] = __fmul_rn(d, ah);
    }
    __syncthreads();
    if (tid == 0) {
      float sum = 0.f;
      for (int j = 0; j < count; ++j) {
        sum = __fadd_rn(sum, prefix[j]);
        prefix[j] = sum;
      }
    }
    __syncthreads();
    for (int q = tid; q < count * count; q += blockDim.x) {
      int i = q / count, j = q % count;
      float s = 0.f;
      if (j <= i) {
        const __nv_bfloat16* ci = C + ((size_t)(start + i) * 8 + g) * 128;
        const __nv_bfloat16* bj = B + ((size_t)(start + j) * 8 + g) * 128;
        if (count == 1) {
          // The one-row batched dot uses eight partial sums of sixteen
          // contiguous products, rather than the GEMM K traversal.
          for (int lane = 0; lane < 8; ++lane) {
            float partial = 0.f;
            for (int k = 0; k < 16; ++k) {
              int n = 16 * lane + k;
              partial = __fmaf_rn(__bfloat162float(ci[n]), __bfloat162float(bj[n]), partial);
            }
            score[lane] = partial;
          }
          for (int step = 4; step > 0; step >>= 1)
            for (int lane = 0; lane < step; ++lane)
              score[lane] = __fadd_rn(score[lane], score[lane + step]);
          s = score[0];
        } else {
          for (int n = 0; n < 128; ++n)
            s = __fmaf_rn(__bfloat162float(ci[n]), __bfloat162float(bj[n]), s);
        }
        // Decay over (j, i] from the interval's own step sum.  Subtracting
        // running prefixes cancels catastrophically once one step dominates
        // the fp32 mantissa (e.g. dt ~ 2^30): later prefixes all round to
        // the same value and the pair decay collapses to exp(0) = 1.
        // Accumulate the already-scaled A*step increments: a raw step sum
        // can overflow fp32 (steps near 2^127) even when the scaled
        // exponent is small and the decay finite.
        float e = 0.f;
        for (int k = j + 1; k <= i; ++k) e = __fadd_rn(e, __fmul_rn(ah, steps[k]));
        s = __fmul_rn(s, __nv_expf(e));
      }
      score[i * 128 + j] = s;
    }
    __syncthreads();
    for (int q = tid; q < count * pcnt; q += blockDim.x) {
      int i = q / pcnt, pl = q % pcnt, p = p0 + pl;
      float diag = 0.f, off = 0.f;
      for (int j = 0; j < count; ++j) {
        float xv = __bfloat162float(x[((size_t)(start + j) * NH + h) * 64 + p]);
        diag = __fmaf_rn(__fmul_rn(score[i * 128 + j], steps[j]), xv, diag);
      }
      const __nv_bfloat16* ci = C + ((size_t)(start + i) * 8 + g) * 128;
      for (int n = 0; n < 128; ++n)
        off = __fmaf_rn(__bfloat162float(ci[n]), state[pl * 128 + n], off);
      off = __fmul_rn(off, __nv_expf(prefix[i]));
      size_t out = ((size_t)(start + i) * NH + h) * 64 + p;
      float skip = __fmul_rn(dh, __bfloat162float(x[out]));
      y[out] = __float2bfloat16_rn(__fadd_rn(__fadd_rn(diag, off), skip));
    }
    __syncthreads();
    float total = prefix[count - 1], decay = __nv_expf(total);
    // Rewrite prefix[] as the suffix sums of A*step over (j, count-1]:
    // additions of same-sign terms only, so the state-update weights never
    // subtract two large prefixes (same cancellation as the score decay
    // above), and scaling each increment before the sum keeps the exponent
    // finite when a raw step sum would overflow fp32.
    __syncthreads();
    if (tid == 0) {
      float tail = 0.f;
      for (int j = count - 1; j >= 0; --j) {
        prefix[j] = tail;
        tail = __fadd_rn(tail, __fmul_rn(ah, steps[j]));
      }
    }
    __syncthreads();
    for (int q = tid; q < pcnt * 128; q += blockDim.x) {
      int p = p0 + q / 128, n = q % 128;
      float update = 0.f;
      for (int j = 0; j < count; ++j) {
        float w = __fmul_rn(steps[j], __nv_expf(prefix[j]));
        float xv = __bfloat162float(x[((size_t)(start + j) * NH + h) * 64 + p]);
        float bv = __bfloat162float(B[((size_t)(start + j) * 8 + g) * 128 + n]);
        update = __fmaf_rn(__fmul_rn(w, xv), bv, update);
      }
      state[q] = __fadd_rn(__fmul_rn(state[q], decay), update);
    }
    __syncthreads();
  }
}

namespace regx {

template <int PT, bool FULL_T = false, int CHUNK_UNROLL = 1, bool TRACK_PROGRESS = true,
          bool SPEC = false>
__global__ __launch_bounds__(8 * PT, 1) void ssd_kernel(
    const __nv_bfloat16* __restrict__ gx, const __nv_bfloat16* __restrict__ gdt,
    const float* __restrict__ gA, const __nv_bfloat16* __restrict__ gB,
    const __nv_bfloat16* __restrict__ gC, const __nv_bfloat16* __restrict__ gD,
    const __nv_bfloat16* __restrict__ gdtb, __nv_bfloat16* __restrict__ gy, int T, int NH,
    int nchunks, const __grid_constant__ CUtensorMap tmaB,
    const __grid_constant__ CUtensorMap tmaC) {
  constexpr int HEADS = (PT == 64) ? 128 : 64;  // PT=64 -> NH128, PT=32 -> NH64
  constexpr int HEADS_PER_GROUP = HEADS / NGR;
  constexpr int NSPL = HDP / PT;

  // Cooperative residency keeps every producer live while consumers poll.
  // The preceding launch's producers left each per-group protocol zeroed, so
  // no launch-start whole-grid reset or barrier is required.
  if (blockIdx.x >= (unsigned)(NH * NSPL)) {
    gram_producer<4 * (PT / 16), PT == 64>(&tmaB, &tmaC, gB, gC, (int)blockIdx.x - NH * NSPL,
                                           nchunks, T);
  } else {
    constexpr int LDX = PT + 8;
    constexpr int NCOL = PT / 16;  // 16-wide p column blocks
    constexpr int NW = 4 * NCOL;   // warps: one per (row block, p block)
    constexpr int NPY = 2;         // 8-wide p tiles per warp (y)
    constexpr int NPM = PT / 16;   // p Mtiles for the state
    constexpr int NSN = 4;         // 8-wide n tiles per warp (state)
    constexpr int NTHR = 32 * NW;
    // score-idle non-scan warp, lane 0
    constexpr int TMA_LEADER_TID = (NW == 16) ? (11 * 32) : (1 * 32);

    const int h = blockIdx.x / NSPL;
    const int ps = blockIdx.x - h * NSPL;
    const int p0 = ps * PT;
    const int g = h / HEADS_PER_GROUP;
    // The padded 136-column TMA box would cross the public 1024-column row for
    // the last group.  Skew that one valid window left by eight bf16 values;
    // consumers add the same shared offset, so their logical 128 columns are
    // unchanged while every global request remains inside [0, 1023].
    const int bc_shift = (g == NGR - 1) ? (LDN - DSN) : 0;
    const int bc_col = g * DSN - bc_shift;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    // warps map to SM quadrants as warp%4.  Put the *p/score-column* index on the
    // quadrant so every quadrant owns one of each row block mi; the causal score
    // tiles can then be spread evenly over the four schedulers (table below).
    // warps map to SM quadrants as warp%4.  With NCOL=4 (PT=64) the p-column
    // index rides on the quadrant, so every quadrant owns one of each row block.
    // With NCOL=2 (PT=32) only eight warps exist; pair row blocks {0,3} and
    // {1,2} on a quadrant instead so the y_diag k-depth (mi+1) is still equal
    // on all four schedulers.
    const int mi = (NW == 16) ? (warp >> 2) : (int)((0xaf50u >> (2 * warp)) & 3u);
    const int ng = warp & (NCOL - 1);  // p in [16*ng,+16) (y_off / y_diag)
    // Role split: warps 0-7 own a 32x16 slice of y (C.state + scores.x), warps
    // 8-15 own a 32x32 slice of the state update.  The two phases are
    // independent inside a window, so halving the warps per phase doubles each
    // warp's tile and cuts the ldmatrix operand stream by a third, while all
    // sixteen warps stay resident for latency hiding.
    const bool yrole = (warp < 8);
    const int myi = 2 * ((warp & 7) >> 2);             // y row blocks myi, myi+1
    const int pm0 = 2 * ((warp & 7) >> 2);             // state p blocks pm0, pm0+1
    const int nns = SPEC ? (warp & 3) : (warp / NPM);  // state n group
    // Causal pruning: only the 10 lower score tiles (j<=i) are needed.  Hand them
    // out so each quadrant computes 3,3,2,2 (NW=16) or 3,2,3,2 (NW=8) of them.
    // NW=16 splits two diagonal tiles into 16x8 halves to even out the n=8
    // subtile count; NW=8 instead gives three warps two whole tiles each.
    const int gj = (NW == 16) ? (int)((0x8d10d1212240ull >> (3 * warp)) & 7u) - 1
                              : (int)((0x8840u >> (2 * warp)) & 3u);
    const int gj1 = (NW == 16) ? 0 : (int)((0x1d00u >> (2 * warp)) & 3u);
    const int gn = (NW == 16) ? 0 : (int)((0x6a51u >> (2 * warp)) & 3u);
    const int gk = (NW == 16) ? (int)((0x55154ee0u >> (2 * warp)) & 3u) : (gn ? 1 : 0);
    const bool hasG = (NW == 16) ? (gk != 0) : (gn > 0);
    const bool gFull = (gk == 1);
    // Index of this warp's raw score tile(s) inside a published window.
    const int gtri = (mi * (mi + 1)) / 2;
    const int gti = gtri + (hasG ? gj : 0);
    const int gti1 = gtri + gj1;
    const int pm = warp & (NPM - 1);  // p Mtile for the state update
    const int nn = warp / NPM;        // n group for the state update

    extern __shared__ __align__(128) char smem_raw[];
    // Three rotating QT x DSN tiles.  With only two, C(c+1) could not be issued
    // until B(c) had been consumed by the state update, leaving the load barely
    // a third of a chunk to land; the spare tile lets B(c+1) go out at the very
    // top of chunk c and C(c+1) right after the first barrier.
    __nv_bfloat16* sTile = (__nv_bfloat16*)smem_raw;  // [3][QT][LDN]
    __nv_bfloat16* sSt = sTile + 3 * QT * LDN;
    __nv_bfloat16* sX = (__nv_bfloat16*)(sSt + PT * LDN);
    // y staging tile.  It replaces the pre-weighted x copy: the state update now
    // folds the decay weight into the A fragment instead, which frees the tile
    // and lets the epilogue write somewhere no MMA phase is reading.
    __nv_bfloat16* sY = sX + QT * LDX;
    __nv_bfloat16* sSc = sY + QT * LDX;
    float* sScan = (float*)(sSc + SCORE_WORDS);  // [2][4][QT]
    uint64_t* sTmaBar = (uint64_t*)(sScan + 8 * QT);
    uint64_t* sPhaseBar = sTmaBar + 1;
    float* sDtTotals = (float*)(sTmaBar + (FULL_T ? 2 : 1));

    // Keep the prefix scan in log2 units so every downstream decay uses the
    // native exp2 path without repeating an e-to-2 base conversion.
    const float Ah = gA[h] * 1.4426950408889634f;
    const float Dh = __bfloat162float(gD[h]);
    const float dtb = __bfloat162float(gdtb[h]);

    if (tid == 0) {
      prefetch_tma(&tmaB);
      prefetch_tma(&tmaC);
      tma_barrier_init(sTmaBar);
      // Every warp arrives, not just the score producers: with the group Gram
      // lifted out, all sixteen warps read the C tile and the state image for
      // exactly as long, so the four non-score warps are no longer guaranteed
      // to be past that phase when the tile is recycled below.
      if constexpr (FULL_T) barrier_init(sPhaseBar, NW);
    }
    __syncthreads();
    for (int i = tid; i < PT * LDN / 2; i += NTHR) ((uint32_t*)sSt)[i] = 0u;

    constexpr int XRC = PT / 8;            // 16B units per x row
    constexpr int XCH = QT * XRC;          // total 16B units per chunk
    constexpr int BCQ = (QT * 16) / NTHR;  // 16B units of B (and C) per thread

    const __nv_bfloat16* gBg = gB + (size_t)g * DSN;
    const __nv_bfloat16* gCg = gC + (size_t)g * DSN;
    const __nv_bfloat16* gxh = gx + (size_t)h * HDP + p0;
    __nv_bfloat16* gyh = gy + (size_t)h * HDP + p0;

    constexpr int XROW = HEADS * HDP;
    const bool xact = tid < XCH;
    const int xr_ = xact ? tid / XRC : 0;
    const int xcc_ = xact ? tid - xr_ * XRC : 0;
    const int xso_ = xr_ * LDX + xcc_ * 8;
    int4 rX;

#define ISSUE_ONE(c, dst, src)                                     \
  {                                                                \
    int t0 = (c) * QT;                                             \
    _Pragma("unroll") for (int q = 0; q < BCQ; ++q) {              \
      int idx = tid + q * NTHR;                                    \
      int r = idx >> 4, cc = idx & 15;                             \
      int t = t0 + r;                                              \
      int nb = (t < T) ? 16 : 0;                                   \
      size_t off = (size_t)(t < T ? t : 0) * (NGR * DSN) + cc * 8; \
      cpa16(sptr((dst) + r * LDN + cc * 8), (src) + off, nb);      \
    }                                                              \
  }
#define LOAD_X(c)                                                                               \
  if (xact) {                                                                                   \
    int t = (c) * QT + xr_;                                                                     \
    if constexpr (FULL_T)                                                                       \
      rX = *(const int4*)(gxh + (size_t)t * XROW + xcc_ * 8);                                   \
    else                                                                                        \
      rX = (t < T) ? *(const int4*)(gxh + (size_t)t * XROW + xcc_ * 8) : make_int4(0, 0, 0, 0); \
  }
#define STORE_X()             \
  if (xact) {                 \
    *(int4*)(sX + xso_) = rX; \
  }
#define LOAD_DT(cc, v)                                                          \
  {                                                                             \
    v = 0;                                                                      \
    if (warp < 2) {                                                             \
      int t = (cc) * QT + 32 * warp + lane;                                     \
      if constexpr (FULL_T) {                                                   \
        const __nv_bfloat16* p = gdt + (size_t)t * HEADS + h;                   \
        asm volatile("ld.global.cg.L2::256B.b16 %0, [%1];" : "=h"(v) : "l"(p)); \
      } else if (t < T) {                                                       \
        const __nv_bfloat16* p = gdt + (size_t)t * HEADS + h;                   \
        asm volatile("ld.global.cg.L2::256B.b16 %0, [%1];" : "=h"(v) : "l"(p)); \
      }                                                                         \
    }                                                                           \
  }
// The last half of the weight quarter is unused (64 bf16 values occupy
// 32 floats).  Use its final three floats for per-warp dt maxima and a flag.
// Large steps amplify operand narrowing; large prefixes lose small decays in
// the parallel log2 scan.  Both cases retain the fp32 reference intermediates
// in precise_head, after this head has finished consuming the Gram ring.
#define DT_SCAN(cc, buf, v)                                                                      \
  if (warp < 2) {                                                                                \
    int t = (cc) * QT + 32 * warp + lane;                                                        \
    float fv = __uint_as_float((uint32_t)(v) << 16);                                             \
    float d = FULL_T ? softplus_f(fv + dtb) : ((t < T) ? softplus_f(fv + dtb) : 0.f);            \
    float peak = d;                                                                              \
    _Pragma("unroll") for (int off = 16; off > 0; off >>= 1) peak =                              \
        fmaxf(peak, __shfl_down_sync(0xffffffffu, peak, off));                                   \
    if (lane == 0) sScan[(buf) * (4 * QT) + 4 * QT - 3 + warp] = peak;                           \
    float s = d * Ah;                                                                            \
    _Pragma("unroll") for (int off = 1; off < 32; off <<= 1) {                                   \
      float u = __shfl_up_sync(0xffffffffu, s, off);                                             \
      if (lane >= off) s += u;                                                                   \
    }                                                                                            \
    if (lane == 31) sDtTotals[warp] = s;                                                         \
    asm volatile("bar.sync 5, 64;" ::: "memory");                                                \
    float cc0 = s + (warp ? sDtTotals[0] : 0.f);                                                 \
    float tt = sDtTotals[0] + sDtTotals[1];                                                      \
    float* pb = sScan + (buf) * (4 * QT);                                                        \
    int i = 32 * warp + lane;                                                                    \
    if (i == 0)                                                                                  \
      pb[4 * QT - 1] =                                                                           \
          (fmaxf(pb[4 * QT - 3], pb[4 * QT - 2]) > 4096.f || fabsf(tt) > 65536.f) ? 1.f : 0.f;   \
    pb[i] = cc0;                                                                                 \
    pb[QT + i] = exp2f(cc0);                                                                     \
    pb[2 * QT + i] = d;                                                                          \
    reinterpret_cast<__nv_bfloat16*>(pb + 3 * QT)[i] = __float2bfloat16_rn(d * exp2f(tt - cc0)); \
  }

    unsigned short dv;
    // Highest published window of this group that this thread has observed.
    unsigned int gready = 0;
    LOAD_DT(0, dv)
    if constexpr (FULL_T) {
      if (tid == 0) {
        tma_expect(sTmaBar, 2 * QT * LDN * (int)sizeof(__nv_bfloat16));
        tma_load_2d(sTile, &tmaB, sTmaBar, bc_col, 0);
        tma_load_2d(sTile + QT * LDN, &tmaC, sTmaBar, bc_col, 0);
      }
    } else {
      if (T >= QT) {
        if (tid == 0) {
          tma_expect(sTmaBar, 2 * QT * LDN * (int)sizeof(__nv_bfloat16));
          tma_load_2d(sTile, &tmaB, sTmaBar, bc_col, 0);
          tma_load_2d(sTile + QT * LDN, &tmaC, sTmaBar, bc_col, 0);
        }
      } else {
        ISSUE_ONE(0, sTile + bc_shift, gBg)
        ISSUE_ONE(0, sTile + QT * LDN + bc_shift, gCg)
        cpa_commit();
      }
    }
    DT_SCAN(0, 0, dv)
    LOAD_X(0)

    float accSt[SPEC ? 2 : 1][NSN][4];
#pragma unroll
    for (int m = 0; m < (SPEC ? 2 : 1); ++m)
#pragma unroll
      for (int n = 0; n < NSN; ++n)
#pragma unroll
        for (int r = 0; r < 4; ++r) accSt[m][n][r] = 0.f;

    // Aligned in-scope launches do not wrap the 256-window ring.  Keeping each
    // (group,tile) stream contiguous lets the consumer advance one pointer and
    // gives L2 a sequential 512-byte window-to-window access pattern.
    bool precise_output = false;
    float peak_x = 0.f, peak_y = 0.f, peak_row_sum = 0.f;
    const float* gpWalk = nullptr;
    const float* gpWalk1 = nullptr;
    if constexpr (!TRACK_PROGRESS && FULL_T) {
      if (hasG) {
        gpWalk = g_gram + (((size_t)g * GTILES + (size_t)gti) * GWIN * GTILE_BF);
        if (NW == 8) gpWalk1 = g_gram + (((size_t)g * GTILES + (size_t)gti1) * GWIN * GTILE_BF);
      }
    }

    struct GramFrag {
      float4 lo, hi;
    };
    GramFrag gfrag, gfrag1;
    float4 ghalf;
    auto load_half = [&](const float* p) {
      uint4 bits;
      if constexpr (FULL_T && PT == 64)
        gld_cg128_l2(bits, p);
      else
        gld_cg128(bits, p);
      return punned<float4>(bits);
    };
    auto load_gram = [&](int gc) {
      if (hasG) {
        unsigned int want = (unsigned int)(gc + 1);
        while (gready < want) gready = gld_acquire(g_gflag + g * 32);
        const float* gp;
        if constexpr (TRACK_PROGRESS || !FULL_T) {
          gp = g_gram +
               ((((size_t)g * GTILES + (size_t)gti) * GWIN + (size_t)(gc & (GWIN - 1))) * GTILE_BF);
        } else {
          gp = gpWalk;
          gpWalk += GTILE_BF;
        }
        if (gFull) {
          gfrag.lo = load_half(gp + lane * 4);
          gfrag.hi = load_half(gp + 128 + lane * 4);
        } else {
          ghalf = load_half(gp + ((gk == 3) ? 128 : 0) + lane * 4);
        }
        if constexpr (NW == 8) {
          if (gn > 1) {
            const float* gq;
            if constexpr (TRACK_PROGRESS || !FULL_T) {
              gq = g_gram +
                   ((((size_t)g * GTILES + (size_t)gti1) * GWIN + (size_t)(gc & (GWIN - 1))) *
                    GTILE_BF);
            } else {
              gq = gpWalk1;
              gpWalk1 += GTILE_BF;
            }
            gfrag1.lo = load_half(gq + lane * 4);
            gfrag1.hi = load_half(gq + 128 + lane * 4);
          }
        }
      }
    };

    load_gram(0);

#pragma unroll CHUNK_UNROLL
    for (int c = 0; c < nchunks; ++c) {
      // X is independent of the B/C TMA completion.  Publish the register tile
      // and issue the next global load before waiting so their traffic overlaps
      // any residual TMA latency; the score-publication barrier below still
      // makes sX visible before y_diag/state consumers.
      STORE_X()
      if constexpr (PT == 64) peak_x = fmaxf(peak_x, peak8(rX));
      // Pick up this chunk's raw score tile as early as possible: the producer
      // is several chunks ahead, so the spin is normally skipped outright and
      // the L2 hit hides behind the C.state phase below.
      // Short launches cannot wrap the ring, so their specialization omits the
      // retirement store entirely.  Longer launches retain the exact coarse
      // watermark protocol: one relaxed store every 8 windows.
      if constexpr (TRACK_PROGRESS) {
        // GPU-scope release: the producer recycles a ring slot after an
        // ld.acquire.gpu of this counter, so the store must order this CTA's
        // earlier Gram reads (CTA-scoped by the barrier below) before it at
        // GPU scope; a relaxed store gives the producer no such guarantee.
        if (tid == 0 && (c & 7) == 0)
          gst_release(g_gprog + g * 16 + ((h % HEADS_PER_GROUP) * NSPL + ps),
                      (unsigned long long)c);
      }
      if constexpr (FULL_T) {
        // The later state-update overlap lets every warp require X here.
        if constexpr (NW == 16)
          asm volatile("bar.sync 6, 512;" ::: "memory");
        else
          asm volatile("bar.sync 6, 256;" ::: "memory");
      } else {
        if constexpr (NW == 16) {
          if (hasG)
            asm volatile("bar.arrive 6, 512;" ::: "memory");
          else
            asm volatile("bar.sync 6, 512;" ::: "memory");
        } else {
          if (hasG)
            asm volatile("bar.arrive 6, 256;" ::: "memory");
          else
            asm volatile("bar.sync 6, 256;" ::: "memory");
        }
      }
      if (c + 1 < nchunks) {
        LOAD_X(c + 1)
      }
      if (c + 1 < nchunks) {
        LOAD_DT(c + 1, dv)
      }
      // On aligned long sequences, drain the previous chunk while the current
      // B/C transaction is still in flight.  sY is not overwritten until the
      // epilogue below, so this moves global-store latency off the loop tail.
      if constexpr (FULL_T)
        if (c > 0) {
          int tp = (c - 1) * QT;
          for (int idx = tid; idx < XCH; idx += NTHR) {
            int r = idx / XRC, cc = idx % XRC;
            gst_cs128(gyh + (size_t)(tp + r) * XROW + cc * 8,
                      *(const int4*)(sY + r * LDX + cc * 8));
          }
        }
      __nv_bfloat16* bBase = sTile + ((c & 1) ? 2 : 0) * (QT * LDN);
      __nv_bfloat16* cBase = sTile + (QT * LDN);
      __nv_bfloat16* spareBase = sTile + ((c & 1) ? 0 : 2) * (QT * LDN);
      __nv_bfloat16* bCur = bBase + bc_shift;
      __nv_bfloat16* cCur = cBase + bc_shift;
      __nv_bfloat16* spare = spareBase + bc_shift;
      const bool curFull = FULL_T || (c + 1) * QT <= T;
      if constexpr (FULL_T) {
        tma_wait(sTmaBar, c & 1);
      } else {
        if (curFull) {
          tma_wait(sTmaBar, c & 1);
          if (c == 0) __syncthreads();
        } else {
          cpa_wait();
          __syncthreads();  // publish cp.async tail data
        }
      }

      // Start the first C.state fragment as soon as current C is ready.  The
      // next-B issue and pointer setup below give this LDSM useful lead before
      // the paired state-fragment load in y_off.
      uint32_t yoffAC[4];
      uint32_t yoffACb = 0;
      const int yrot = 2 * nn;  // first k tile this warp owns in registers
      if constexpr (NPY == 2) {
        if (c > 0) {
          yoffACb = addr4<LDN>(cCur, 16 * (SPEC ? myi : mi), 0, lane);
          if constexpr (SPEC) {
            if (yrole) ldm4o<0>(yoffAC, yoffACb);
          } else
            ldm4(yoffAC, yoffACb + (uint32_t)(32 * yrot));
        }
      }

      const float* sCum = sScan + (c & 1) * (4 * QT);
      precise_output |= sCum[4 * QT - 1] != 0.f;
      const float* sEcum = sCum + QT;
      const float* sDtf = sEcum + QT;
      const __nv_bfloat16* sWt = reinterpret_cast<const __nv_bfloat16*>(sDtf + QT);
      const bool hasNext = c + 1 < nchunks;
      const bool nextFull = (c + 2) * QT <= T;
      // B(c+1) goes out now: the spare tile has been free since the previous
      // chunk's state update finished.
      if (hasNext) {
        if constexpr (FULL_T) {
          if (tid == TMA_LEADER_TID) {
            tma_expect(sTmaBar, 2 * QT * LDN * (int)sizeof(__nv_bfloat16));
            tma_load_2d(spareBase, &tmaB, sTmaBar, bc_col, (c + 1) * QT);
          }
        } else {
          if (nextFull) {
            if (tid == TMA_LEADER_TID) {
              tma_expect(sTmaBar, 2 * QT * LDN * (int)sizeof(__nv_bfloat16));
              tma_load_2d(spareBase, &tmaB, sTmaBar, bc_col, (c + 1) * QT);
            }
          } else {
            ISSUE_ONE(c + 1, spare, gBg)
          }
        }
      }

      // ---------------- fp16 x and decay-weighted x --------------------
      const int ia0 = 16 * (SPEC ? myi : mi) + (lane >> 2);

      // ---------------- y_off (C.state) --------------------------------
      // The raw score block now arrives from the group's producer CTA, so this
      // phase is one uniform C.state chain for every warp instead of four
      // variants that also folded in a per-head copy of C.B^T.
      float accY[SPEC ? 2 : 1][NPY][4];
#pragma unroll
      for (int m = 0; m < (SPEC ? 2 : 1); ++m)
#pragma unroll
        for (int j = 0; j < NPY; ++j)
#pragma unroll
          for (int r = 0; r < 4; ++r) accY[m][j][r] = 0.f;

      // The public contract has a zero initial state, so C.state is exactly
      // zero in the first window.  Its uniform branch removes the first C-state
      // MMA chain without specializing on an exact sequence length.
#define YOFF_DECAY()                                       \
  {                                                        \
    float ea = sEcum[ia0], eb = sEcum[ia0 + 8];            \
    _Pragma("unroll") for (int pt = 0; pt < NPY; ++pt) {   \
      accY[0][pt][0] *= ea;                                \
      accY[0][pt][1] *= ea;                                \
      accY[0][pt][2] *= eb;                                \
      accY[0][pt][3] *= eb;                                \
    }                                                      \
    if constexpr (SPEC) {                                  \
      float e2 = sEcum[ia0 + 16], e3 = sEcum[ia0 + 24];    \
      _Pragma("unroll") for (int pt = 0; pt < NPY; ++pt) { \
        accY[1][pt][0] *= e2;                              \
        accY[1][pt][1] *= e2;                              \
        accY[1][pt][2] *= e3;                              \
        accY[1][pt][3] *= e3;                              \
      }                                                    \
    }                                                      \
  }
      // SPEC: two 16-row C fragments share one state fragment per k step, so a
      // 32x16 output tile costs 3 ldmatrix per step instead of 2 per 16x16.
#define YOFF_SPEC()                                            \
  if (c > 0) {                                                 \
    const uint32_t aCb0 = yoffACb;                             \
    const uint32_t aCb1 = aCb0 + (uint32_t)(16 * LDN * 2);     \
    const uint32_t bTb = addr4b<LDN>(sSt, 16 * ng, 0, lane);   \
    uint32_t aC0[4], aC1[4], bT[4];                            \
    aC0[0] = yoffAC[0];                                        \
    aC0[1] = yoffAC[1];                                        \
    aC0[2] = yoffAC[2];                                        \
    aC0[3] = yoffAC[3];                                        \
    ldm4o<0>(aC1, aCb1);                                       \
    ldm4o<0>(bT, bTb);                                         \
    _Pragma("unroll") for (int kt = 0; kt < 8; ++kt) {         \
      uint32_t n0[4], n1[4], nb[4];                            \
      if (kt < 7) {                                            \
        ldm4(n0, aCb0 + (uint32_t)(32 * (kt + 1)));            \
        ldm4(n1, aCb1 + (uint32_t)(32 * (kt + 1)));            \
        ldm4(nb, bTb + (uint32_t)(32 * (kt + 1)));             \
      }                                                        \
      uint32_t g0[2] = {bT[0], bT[1]}, g1[2] = {bT[2], bT[3]}; \
      mma_f32(accY[0][0], aC0, g0);                            \
      mma_f32(accY[0][1], aC0, g1);                            \
      mma_f32(accY[1][0], aC1, g0);                            \
      mma_f32(accY[1][1], aC1, g1);                            \
      if (kt < 7) {                                            \
        _Pragma("unroll") for (int z = 0; z < 4; ++z) {        \
          aC0[z] = n0[z];                                      \
          aC1[z] = n1[z];                                      \
          bT[z] = nb[z];                                       \
        }                                                      \
      }                                                        \
    }                                                          \
    YOFF_DECAY()                                               \
  }
      if constexpr (!SPEC) {
        if (c > 0) {
          if constexpr (NPY == 2) {
            const uint32_t aCb = yoffACb;
            const uint32_t bTb = addr4b<LDN>(sSt, 16 * ng, 0, lane);
            uint32_t bT[4];
            uint32_t(&aC)[4] = yoffAC;
            // k tiles 2*nn and 2*nn+1 come straight out of accSt in B-operand order
            // (same bf16 rounding the stmatrix publication applies), so the loop is
            // rotated to run them first and two ldmatrix per warp disappear.
            bT[0] = pack2f(accSt[0][0][0], accSt[0][0][1]);
            bT[1] = pack2f(accSt[0][1][0], accSt[0][1][1]);
            bT[2] = pack2f(accSt[0][0][2], accSt[0][0][3]);
            bT[3] = pack2f(accSt[0][1][2], accSt[0][1][3]);
#define PHA_N_MMA()                                          \
  {                                                          \
    uint32_t g0[2] = {bT[0], bT[1]}, g1[2] = {bT[2], bT[3]}; \
    mma_f32(accY[0][0], aC, g0);                             \
    mma_f32(accY[0][1], aC, g1);                             \
  }
#define PHA_N_AC(j) ldm4(aCn, aCb + (uint32_t)(32 * ((yrot + (j)) & 7)))
#define PHA_N_BT(j) ldm4(bTn, bTb + (uint32_t)(32 * ((yrot + (j)) & 7)))
            {
              uint32_t aCn[4], bTn[4];
              PHA_N_AC(1);
              PHA_N_MMA()
              _Pragma("unroll") for (int z = 0; z < 4; ++z) aC[z] = aCn[z];
              bT[0] = pack2f(accSt[0][2][0], accSt[0][2][1]);
              bT[1] = pack2f(accSt[0][3][0], accSt[0][3][1]);
              bT[2] = pack2f(accSt[0][2][2], accSt[0][2][3]);
              bT[3] = pack2f(accSt[0][3][2], accSt[0][3][3]);
              PHA_N_AC(2);
              PHA_N_BT(2);
              PHA_N_MMA()
              _Pragma("unroll") for (int z = 0; z < 4; ++z) {
                aC[z] = aCn[z];
                bT[z] = bTn[z];
              }
              _Pragma("unroll") for (int j = 3; j < 8; ++j) {
                uint32_t aC2[4], bT2[4];
                ldm4(aC2, aCb + (uint32_t)(32 * ((yrot + j) & 7)));
                ldm4(bT2, bTb + (uint32_t)(32 * ((yrot + j) & 7)));
                PHA_N_MMA()
                _Pragma("unroll") for (int z = 0; z < 4; ++z) {
                  aC[z] = aC2[z];
                  bT[z] = bT2[z];
                }
              }
              PHA_N_MMA()
            }
#undef PHA_N_BT
#undef PHA_N_AC
#undef PHA_N_MMA
          } else {
#pragma unroll
            for (int kt = 0; kt < 8; ++kt) {
              uint32_t aC[4];
              ldm4(aC, addr4<LDN>(cCur, 16 * mi, 16 * kt, lane));
              uint32_t bT[2];
              ldm2(bT, addr2<LDN>(sSt, 8 * ng, 16 * kt, lane));
              mma_f32(accY[0][0], aC, bT);
            }
          }
          YOFF_DECAY()
        }
      }

      // ---------------- scores = S_raw * decay * dt ---------------------
      // Emitted before the barrier so a single block-wide sync publishes the
      // staged x and the score tile together: two barriers per chunk, not three.
      if (hasG) {
        const int ia0 = 16 * mi + (lane >> 2);  // score tile row block
        int ib0 = ia0 + 8;
        float ca = sCum[ia0], cb = sCum[ib0];
        const float row_rel = exp2f(cb - ca);
        // Each float4 is a producer lane's m16n8 accumulator.
#define SCORE_UNPACK_FULL(FR, AS)          \
  memcpy(AS[0], &(FR).lo, sizeof(float4)); \
  memcpy(AS[1], &(FR).hi, sizeof(float4));
#define SCORE_EMIT_OFF(JJ, JT, AS)                  \
  {                                                 \
    int j0 = 16 * (JJ) + 8 * (JT) + 2 * (lane & 3); \
    float2 cjv = *(const float2*)(sCum + j0);       \
    float2 wvv = *(const float2*)(sDtf + j0);       \
    float cj0 = cjv.x, cj1 = cjv.y;                 \
    float w0 = wvv.x, w1 = wvv.y;                   \
    float q0 = exp2f(ca - cj0) * w0;                \
    float q1 = exp2f(ca - cj1) * w1;                \
    float v0 = AS[JT][0] * q0;                      \
    float v1 = AS[JT][1] * q1;                      \
    float v2 = AS[JT][2] * (row_rel * q0);          \
    float v3 = AS[JT][3] * (row_rel * q1);          \
    stage_pair(sSc + score_index(ia0, j0), v0, v1); \
    stage_pair(sSc + score_index(ib0, j0), v2, v3); \
  }
#define SCORE_EMIT_DIAG_LOW(JJ, AS, SLOT)                                   \
  {                                                                         \
    int jr = 2 * (lane & 3);                                                \
    int ir = lane >> 2;                                                     \
    int j0 = 16 * (JJ) + jr;                                                \
    float2 cjv = *(const float2*)(sCum + j0);                               \
    float2 wvv = *(const float2*)(sDtf + j0);                               \
    float cj0 = cjv.x, cj1 = cjv.y;                                         \
    float w0 = wvv.x, w1 = wvv.y;                                           \
    float v0 = (jr <= ir) ? AS[SLOT][0] * (exp2f(ca - cj0) * w0) : 0.f;     \
    float v1 = (jr + 1 <= ir) ? AS[SLOT][1] * (exp2f(ca - cj1) * w1) : 0.f; \
    float v2 = AS[SLOT][2] * (exp2f(cb - cj0) * w0);                        \
    float v3 = AS[SLOT][3] * (exp2f(cb - cj1) * w1);                        \
    stage_pair(sSc + score_index(ia0, j0), v0, v1);                         \
    stage_pair(sSc + score_index(ib0, j0), v2, v3);                         \
  }
#define SCORE_EMIT_DIAG_HIGH(JJ, AS, SLOT)                                  \
  {                                                                         \
    int jr = 2 * (lane & 3);                                                \
    int ir = lane >> 2;                                                     \
    int j0 = 16 * (JJ) + 8 + jr;                                            \
    float2 cjv = *(const float2*)(sCum + j0);                               \
    float2 wvv = *(const float2*)(sDtf + j0);                               \
    float cj0 = cjv.x, cj1 = cjv.y;                                         \
    float w0 = wvv.x, w1 = wvv.y;                                           \
    float v2 = (jr <= ir) ? AS[SLOT][2] * (exp2f(cb - cj0) * w0) : 0.f;     \
    float v3 = (jr + 1 <= ir) ? AS[SLOT][3] * (exp2f(cb - cj1) * w1) : 0.f; \
    stage_pair(sSc + score_index(ia0, j0), 0.f, 0.f);                       \
    stage_pair(sSc + score_index(ib0, j0), v2, v3);                         \
  }
#define SCORE_EMIT_TILE(JJ, FR)                                          \
  {                                                                      \
    float accS[2][4];                                                    \
    SCORE_UNPACK_FULL(FR, accS)                                          \
    if ((JJ) < mi) {                                                     \
      SCORE_EMIT_OFF(JJ, 0, accS) SCORE_EMIT_OFF(JJ, 1, accS)            \
    } else {                                                             \
      SCORE_EMIT_DIAG_LOW(JJ, accS, 0) SCORE_EMIT_DIAG_HIGH(JJ, accS, 1) \
    }                                                                    \
  }
        if constexpr (NW == 16) {
          if (gFull) {
            SCORE_EMIT_TILE(gj, gfrag)
          } else {
            float accS[1][4];
            memcpy(accS[0], &ghalf, sizeof(ghalf));
            if (gk == 3) {
              SCORE_EMIT_DIAG_HIGH(gj, accS, 0)
            } else {
              SCORE_EMIT_DIAG_LOW(gj, accS, 0)
            }
          }
        } else {
          SCORE_EMIT_TILE(gj, gfrag)
          if (gn > 1) {
            SCORE_EMIT_TILE(gj1, gfrag1)
          }
        }
#undef SCORE_EMIT_TILE
#undef SCORE_EMIT_DIAG_HIGH
#undef SCORE_EMIT_DIAG_LOW
#undef SCORE_EMIT_OFF
#undef SCORE_UNPACK_FULL
      }

      if (c + 1 < nchunks) load_gram(c + 1);

      if constexpr (!FULL_T) {
        if (!SPEC && !hasG)
          update_state_tile<NSN, LDX>(accSt[0], sX, bCur, sWt, sEcum, pm, nn, lane);
      }

      if constexpr (SPEC) {
        // The two phases of a window are independent: C.state consumes the
        // previous state image while the state update consumes this window's
        // x/B.  Splitting them across warp halves doubles each warp's output
        // tile, so the same MMA count reloads a third fewer operands.
        // Arrive before the phase work, not after: the wait below then finds
        // the barrier already satisfied instead of sixteen warps polling the
        // mbarrier at once.  The C tile stays live past this point, so its
        // reload moves behind the block barrier at the end of the window.
        if constexpr (FULL_T) {
          __syncwarp();
          if (lane == 0) barrier_arrive(sPhaseBar);
        }
        if (yrole) {
          YOFF_SPEC()
        } else
          update_state_tile2<NSN, LDX>(accSt, sX, bCur, sWt, sEcum, pm0, nns, lane);
        if constexpr (FULL_T) {
          if (c + 1 < nchunks) {
            DT_SCAN(c + 1, (c + 1) & 1, dv)
          }
          tma_wait(sPhaseBar, c & 1);
        } else {
          if (c + 1 < nchunks) {
            DT_SCAN(c + 1, (c + 1) & 1, dv)
          }
          __syncthreads();
        }
      } else if constexpr (FULL_T) {
        // Score warps can advance their register-only state update while other
        // warps finish publishing scores.  The scan warps likewise use the wait
        // shadow, then everyone observes sSc through the acquire below.
        // Publish every lane's score stores through a warp fence, then use a
        // single release arrival per warp.  The acquire wait below retains the
        // same transitive ordering for all score consumers.
        __syncwarp();
        if (lane == 0) barrier_arrive(sPhaseBar);
        update_state_tile<NSN, LDX>(accSt[0], sX, bCur, sWt, sEcum, pm, nn, lane);
        if (c + 1 < nchunks) {
          DT_SCAN(c + 1, (c + 1) & 1, dv)
        }
        tma_wait(sPhaseBar, c & 1);  // sSc ready; sSt and the C tile free
      } else if constexpr (!SPEC) {
        __syncthreads();
      }

      if (hasNext && !(SPEC && FULL_T)) {
        if constexpr (FULL_T) {
          if (tid == TMA_LEADER_TID) tma_load_2d(cBase, &tmaC, sTmaBar, bc_col, (c + 1) * QT);
        } else {
          if (nextFull) {
            if (tid == TMA_LEADER_TID) tma_load_2d(cBase, &tmaC, sTmaBar, bc_col, (c + 1) * QT);
          } else {
            ISSUE_ONE(c + 1, cCur, gCg)
            cpa_commit();
          }
        }
      }

      const bool stateFirst = !FULL_T && hasG && (mi == 3);
      if constexpr (!FULL_T) {
        if (stateFirst) update_state_tile<NSN, LDX>(accSt[0], sX, bCur, sWt, sEcum, pm, nn, lane);
      }

      // Bound errors in the intra-window score operands *before* cancellation
      // by measuring their absolute row sums. Only read live causal entries.
      if constexpr (PT == 64) {
        const int rows = min(QT, T - c * QT);
        for (int i = warp; i < rows; i += NW) {
          float s = 0.f;
          for (int j = lane; j <= i; j += 32) s += fabsf(__bfloat162float(sSc[score_index(i, j)]));
#pragma unroll
          for (int d = 16; d; d >>= 1) s += __shfl_down_sync(0xffffffffu, s, d);
          if (lane == 0) peak_row_sum = fmaxf(peak_row_sum, isfinite(s) ? s : CUDART_INF_F);
        }
      }

      // Reuse the X fragment for both score planes. Offsets are in bytes
      // for ldmatrix; adjacent tiles are 512 BF16, adjacent planes 256.
      auto score_mma = [&](float* y0, float* y1, uint32_t tile_addr, const uint32_t* x0,
                           const uint32_t* x1) {
        uint32_t hi[4], lo[4];
        ldm4(hi, tile_addr);
        ldm4o<512>(lo, tile_addr);
        mma_f32(y0, hi, x0);
        mma_f32(y1, hi, x1);
        mma_f32(y0, lo, x0);
        mma_f32(y1, lo, x1);
      };
      // ---------------- y_diag = scores . x -----------------------------
      if constexpr (SPEC) {
        if (yrole) {
          const uint32_t aSb0 = addr4<16>(sSc + score_tile(myi, 0), 0, 0, lane);
          const uint32_t aSb1 = addr4<16>(sSc + score_tile(myi + 1, 0), 0, 0, lane);
          const uint32_t bXb = addr4bt<LDX>(sX, 16 * ng, 0, lane);
          // One x fragment per k step feeds both row blocks; the upper block has
          // one extra causal k step.
#define YDIAG2_BODY(kt, BOTH)                                        \
  {                                                                  \
    uint32_t bX[4];                                                  \
    ldm4to<32 * (kt) * LDX>(bX, bXb);                                \
    uint32_t g0[2] = {bX[0], bX[1]}, g1[2] = {bX[2], bX[3]};         \
    if (BOTH) {                                                      \
      score_mma(accY[0][0], accY[0][1], aSb0 + 1024 * (kt), g0, g1); \
    }                                                                \
    {                                                                \
      score_mma(accY[1][0], accY[1][1], aSb1 + 1024 * (kt), g0, g1); \
    }                                                                \
  }
          if (myi == 2) {
            YDIAG2_BODY(0, true) YDIAG2_BODY(1, true) YDIAG2_BODY(2, true) YDIAG2_BODY(3, false)
          } else {
            YDIAG2_BODY(0, true) YDIAG2_BODY(1, false)
          }
#undef YDIAG2_BODY
        }
      } else {
        const uint32_t aSb = addr4<16>(sSc + score_tile(mi, 0), 0, 0, lane);
        const uint32_t bXb = addr4bt<LDX>(sX, 16 * ng, 0, lane);
#define YDIAG_BODY(kt)                                            \
  {                                                               \
    static_assert(NPY == 2);                                      \
    uint32_t bX[4];                                               \
    ldm4to<32 * (kt) * LDX>(bX, bXb);                             \
    uint32_t g0[2] = {bX[0], bX[1]}, g1[2] = {bX[2], bX[3]};      \
    score_mma(accY[0][0], accY[0][1], aSb + 1024 * (kt), g0, g1); \
  }
        // Score blocks with k>mi are identically zero (causal).  One uniform
        // branch chain, each arm fully unrolled, keeps the k loop branch-free.
        if (mi == 3) {
          YDIAG_BODY(0) YDIAG_BODY(1) YDIAG_BODY(2) YDIAG_BODY(3)
        } else if (mi == 2) {
          YDIAG_BODY(0) YDIAG_BODY(1) YDIAG_BODY(2)
        } else if (mi == 1) {
          YDIAG_BODY(0) YDIAG_BODY(1)
        } else {
          YDIAG_BODY(0)
        }
#undef YDIAG_BODY
      }

      // ---------------- epilogue: y = acc + D*x -------------------------
      if constexpr (SPEC) {
        if (yrole) {
#pragma unroll
          for (int rb = 0; rb < 2; ++rb) {
            int i0 = ia0 + 16 * rb;
            uint32_t yFrag[4];
#pragma unroll
            for (int pt = 0; pt < 2; ++pt) {
              int pc = 8 * (NPY * ng + pt) + 2 * (lane & 3);
              uint32_t xr0 = *(const uint32_t*)(sX + i0 * LDX + pc);
              uint32_t xr1 = *(const uint32_t*)(sX + (i0 + 8) * LDX + pc);
              float2 xa = __bfloat1622float2(punned<__nv_bfloat162>(xr0));
              float2 xb = __bfloat1622float2(punned<__nv_bfloat162>(xr1));
              __nv_bfloat162 y0 =
                  __floats2bfloat162_rn(accY[rb][pt][0] + Dh * xa.x, accY[rb][pt][1] + Dh * xa.y);
              __nv_bfloat162 y1 =
                  __floats2bfloat162_rn(accY[rb][pt][2] + Dh * xb.x, accY[rb][pt][3] + Dh * xb.y);
              yFrag[2 * pt] = punned<uint32_t>(y0);
              yFrag[2 * pt + 1] = punned<uint32_t>(y1);
            }
            uint32_t dst =
                sptr(sY + (16 * (myi + rb) + (lane & 15)) * LDX + 8 * (NPY * ng + (lane >> 4)));
            asm volatile(
                "stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 "
                "[%0], {%1,%2,%3,%4};\n" ::"r"(dst),
                "r"(yFrag[0]), "r"(yFrag[1]), "r"(yFrag[2]), "r"(yFrag[3]));
          }
        }
      } else if constexpr (NPY == 2) {
        // PT64 produces two adjacent 16x8 tiles per warp.  Publish the complete
        // 16x16 BF16 tile with one collective store instead of four STS.32s.
        uint32_t yFrag[4];
#pragma unroll
        for (int pt = 0; pt < 2; ++pt) {
          int pc = 8 * (NPY * ng + pt) + 2 * (lane & 3);
          uint32_t xr0 = *(const uint32_t*)(sX + ia0 * LDX + pc);
          uint32_t xr1 = *(const uint32_t*)(sX + (ia0 + 8) * LDX + pc);
          float2 xa = __bfloat1622float2(punned<__nv_bfloat162>(xr0));
          float2 xb = __bfloat1622float2(punned<__nv_bfloat162>(xr1));
          __nv_bfloat162 y0 =
              __floats2bfloat162_rn(accY[0][pt][0] + Dh * xa.x, accY[0][pt][1] + Dh * xa.y);
          __nv_bfloat162 y1 =
              __floats2bfloat162_rn(accY[0][pt][2] + Dh * xb.x, accY[0][pt][3] + Dh * xb.y);
          yFrag[2 * pt] = punned<uint32_t>(y0);
          yFrag[2 * pt + 1] = punned<uint32_t>(y1);
        }
        uint32_t dst = sptr(sY + (16 * mi + (lane & 15)) * LDX + 8 * (NPY * ng + (lane >> 4)));
        asm volatile(
            "stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 "
            "[%0], {%1,%2,%3,%4};\n" ::"r"(dst),
            "r"(yFrag[0]), "r"(yFrag[1]), "r"(yFrag[2]), "r"(yFrag[3]));
      } else {
#pragma unroll
        for (int pt = 0; pt < NPY; ++pt) {
          int pc = 8 * (NPY * ng + pt) + 2 * (lane & 3);
          uint32_t xr0 = *(const uint32_t*)(sX + ia0 * LDX + pc);
          uint32_t xr1 = *(const uint32_t*)(sX + (ia0 + 8) * LDX + pc);
          float2 xa = __bfloat1622float2(punned<__nv_bfloat162>(xr0));
          float2 xb = __bfloat1622float2(punned<__nv_bfloat162>(xr1));
          *(__nv_bfloat162*)(sY + ia0 * LDX + pc) =
              __floats2bfloat162_rn(accY[0][pt][0] + Dh * xa.x, accY[0][pt][1] + Dh * xa.y);
          *(__nv_bfloat162*)(sY + (ia0 + 8) * LDX + pc) =
              __floats2bfloat162_rn(accY[0][pt][2] + Dh * xb.x, accY[0][pt][3] + Dh * xb.y);
        }
      }
      if constexpr (!FULL_T) {
        if constexpr (!SPEC) {
          if (c + 1 < nchunks) {
            DT_SCAN(c + 1, (c + 1) & 1, dv)
          }
          if (hasG && !stateFirst)
            update_state_tile<NSN, LDX>(accSt[0], sX, bCur, sWt, sEcum, pm, nn, lane);
        }
      }
      // ---------------- publish fp16 state ------------------------------
      // SPEC publishes after the block barrier instead: the y warps read the
      // previous image during their C.state chain above, and the next window's
      // first barrier orders this store before any new read.
      if constexpr (!SPEC) {
#pragma unroll
        for (int n = 0; n < NSN; n += 2) {
          int nc = 8 * NSN * nn + 8 * n;
          uint32_t dst = sptr(sSt + (16 * pm + (lane & 15)) * LDN + nc + 8 * (lane >> 4));
          __nv_bfloat162 w00 = __floats2bfloat162_rn(accSt[0][n][0], accSt[0][n][1]);
          __nv_bfloat162 w01 = __floats2bfloat162_rn(accSt[0][n][2], accSt[0][n][3]);
          __nv_bfloat162 w10 = __floats2bfloat162_rn(accSt[0][n + 1][0], accSt[0][n + 1][1]);
          __nv_bfloat162 w11 = __floats2bfloat162_rn(accSt[0][n + 1][2], accSt[0][n + 1][3]);
          asm volatile(
              "stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 "
              "[%0], {%1,%2,%3,%4};\n" ::"r"(dst),
              "r"(punned<uint32_t>(w00)), "r"(punned<uint32_t>(w01)), "r"(punned<uint32_t>(w10)),
              "r"(punned<uint32_t>(w11)));
        }
      }

      __syncthreads();  // epilogue + next scan ready; sX/sXw readers finished
      if constexpr (PT == 64) {
        if (c * QT + xr_ < T) peak_y = fmaxf(peak_y, peak8(*(const int4*)(sY + xso_)));
      }
      if constexpr (SPEC) {
        if constexpr (FULL_T) {
          if (hasNext && tid == TMA_LEADER_TID)
            tma_load_2d(cBase, &tmaC, sTmaBar, bc_col, (c + 1) * QT);
        }
        if (!yrole) {
#pragma unroll
          for (int m = 0; m < 2; ++m)
#pragma unroll
            for (int n = 0; n < NSN; n += 2) {
              int nc = 8 * NSN * nns + 8 * n;
              uint32_t dst =
                  sptr(sSt + (16 * (pm0 + m) + (lane & 15)) * LDN + nc + 8 * (lane >> 4));
              __nv_bfloat162 w00 = __floats2bfloat162_rn(accSt[m][n][0], accSt[m][n][1]);
              __nv_bfloat162 w01 = __floats2bfloat162_rn(accSt[m][n][2], accSt[m][n][3]);
              __nv_bfloat162 w10 = __floats2bfloat162_rn(accSt[m][n + 1][0], accSt[m][n + 1][1]);
              __nv_bfloat162 w11 = __floats2bfloat162_rn(accSt[m][n + 1][2], accSt[m][n + 1][3]);
              asm volatile(
                  "stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 "
                  "[%0], {%1,%2,%3,%4};\n" ::"r"(dst),
                  "r"(punned<uint32_t>(w00)), "r"(punned<uint32_t>(w01)),
                  "r"(punned<uint32_t>(w10)), "r"(punned<uint32_t>(w11)));
            }
        }
      }
      if constexpr (!FULL_T) {
        int t0 = c * QT;
        for (int idx = tid; idx < XCH; idx += NTHR) {
          int r = idx / XRC, cc = idx % XRC;
          int t = t0 + r;
          if (t < T) {
            gst_cs128(gyh + (size_t)t * XROW + cc * 8, *(const int4*)(sY + r * LDX + cc * 8));
          }
        }
      }
    }
    if constexpr (FULL_T) {
      int tp = (nchunks - 1) * QT;
      for (int idx = tid; idx < XCH; idx += NTHR) {
        int r = idx / XRC, cc = idx % XRC;
        gst_cs128(gyh + (size_t)(tp + r) * XROW + cc * 8, *(const int4*)(sY + r * LDX + cc * 8));
      }
    }
    // Every ring read is upstream of the final loop barrier.  On the large-
    // magnitude path, finish draining y before reusing the existing shared
    // allocation for the fp32 128-token SSD calculation.
    if constexpr (PT == 64) {
      // Reuse the retired tile storage; all ring traffic and y drains finish
      // before any thread enters precise_head. Reduce once per head, not per
      // token, and make the replay decision uniform across the CTA.
      __syncthreads();
      float* peaks = reinterpret_cast<float*>(smem_raw);
#pragma unroll
      for (int d = 16; d; d >>= 1) {
        peak_x = fmaxf(peak_x, __shfl_down_sync(0xffffffffu, peak_x, d));
        peak_y = fmaxf(peak_y, __shfl_down_sync(0xffffffffu, peak_y, d));
      }
      if (lane == 0) {
        peaks[warp] = peak_x;
        peaks[NW + warp] = peak_y;
        peaks[2 * NW + warp] = peak_row_sum;
      }
      __syncthreads();
      if (warp == 0) {
        peak_x = lane < NW ? peaks[lane] : 0.f;
        peak_y = lane < NW ? peaks[NW + lane] : 0.f;
        peak_row_sum = lane < NW ? peaks[2 * NW + lane] : 0.f;
#pragma unroll
        for (int d = NW / 2; d; d >>= 1) {
          peak_x = fmaxf(peak_x, __shfl_down_sync(0xffffffffu, peak_x, d));
          peak_y = fmaxf(peak_y, __shfl_down_sync(0xffffffffu, peak_y, d));
          peak_row_sum = fmaxf(peak_row_sum, __shfl_down_sync(0xffffffffu, peak_row_sum, d));
        }
        if (lane == 0) {
          // Bound loss from staging the FP32 weighted score, not loss
          // from approximating the recurrence itself. For normal operands
          // |v-hi| <= u/(1-u)*|hi|, u=2^-8. Rounding that residual costs
          // at most u times its magnitude. Keep slack for the row-sum
          // reduction and the old absolute floor for tiny operands.
          double err = (0.000016 * peak_row_sum + QT * 4098.0 * 0x1p-126) * peak_x;
          precise_output |=
              Ah > 0.f || !isfinite(err) || !isfinite(peak_y) || err > 0.015 * (peak_y + 1.e-9);
        }
      }
      if (__syncthreads_or(precise_output)) {
        precise_head(gx, gdt, gA, gB, gC, gD, gdtb, gy, T, NH, h, 0, 64, smem_raw);
      }
    }
    if (tid == 0) atomicAdd(g_gdone + g, 1u);
#undef LOAD_DT
#undef DT_SCAN
#undef ISSUE_ONE
#undef LOAD_X
#undef STORE_X
  }

  // Keep every CTA resident until all groups have retired their ring reads
  // and reset the protocol.  The residency guard excludes two complete grids,
  // but without this rendezvous a later grid can overlap early CTA retirement.
  cooperative_groups::this_grid().sync();
}

// -------------------------------------------------------------------------
struct CooperativeLaunchState {
  std::once_flag initialize;
  int device = -1;
  int resident_capacity = 0;
};

static CooperativeLaunchState g_coop64;
static CooperativeLaunchState g_coop64_full;
static CooperativeLaunchState g_coop32;
static CooperativeLaunchState g_coop64_noprog;
static CooperativeLaunchState g_coop64_full_noprog;
static CooperativeLaunchState g_coop32_noprog;
static CooperativeLaunchState g_coop32_full;
static CooperativeLaunchState g_coop32_full_noprog;

static inline CUresult make_bc_tma(CUtensorMap* map, const __nv_bfloat16* ptr, int T) {
  const uint64_t dims[2] = {(uint64_t)(NGR * DSN), (uint64_t)T};
  const uint64_t strides[1] = {(uint64_t)(NGR * DSN * sizeof(__nv_bfloat16))};
  const uint32_t box[2] = {(uint32_t)LDN, (uint32_t)QT};
  const uint32_t elem_strides[2] = {1, 1};
  return cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)ptr, dims, strides,
                                box, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}
void Run(ffi::TensorView x, ffi::TensorView dt, ffi::TensorView A, ffi::TensorView Bm,
         ffi::TensorView Cm, ffi::TensorView D, ffi::TensorView dtb, ffi::TensorView y) {
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, x.device().device_id));
  int T = (int)x.shape()[0];
  int NH = (int)x.shape()[1];
  int nchunks = (T + QT - 1) / QT;
  if (T <= 0) return;

  const __nv_bfloat16* px = (const __nv_bfloat16*)x.data_ptr();
  const __nv_bfloat16* pdt = (const __nv_bfloat16*)dt.data_ptr();
  const float* pA = (const float*)A.data_ptr();
  const __nv_bfloat16* pB = (const __nv_bfloat16*)Bm.data_ptr();
  const __nv_bfloat16* pC = (const __nv_bfloat16*)Cm.data_ptr();
  const __nv_bfloat16* pD = (const __nv_bfloat16*)D.data_ptr();
  const __nv_bfloat16* pdb = (const __nv_bfloat16*)dtb.data_ptr();
  __nv_bfloat16* py = (__nv_bfloat16*)y.data_ptr();

  alignas(64) CUtensorMap tmaB{}, tmaC{};
  TVM_FFI_ICHECK_EQ(make_bc_tma(&tmaB, pB, T), CUDA_SUCCESS) << "failed to encode B tensor map";
  TVM_FFI_ICHECK_EQ(make_bc_tma(&tmaC, pC, T), CUDA_SUCCESS) << "failed to encode C tensor map";

  const int smem64 = (int)((3 * QT * LDN + 64 * LDN + 2 * QT * (64 + 8) + SCORE_WORDS) * 2 +
                           8 * QT * (int)sizeof(float) + 2 * sizeof(uint64_t) + 2 * sizeof(float));
  const int smem32 = (int)((3 * QT * LDN + 32 * LDN + 2 * QT * (32 + 8) + SCORE_WORDS) * 2 +
                           8 * QT * (int)sizeof(float) + 2 * sizeof(uint64_t) + 2 * sizeof(float));

  // cudaLaunchCooperativeKernel is a stream operation and is captured as a
  // cooperative kernel node.  There is deliberately no host launch counter or
  // other replay-varying state: the kernel resets its own protocol before the
  // grid-wide barrier on every execution, including every graph replay.
  auto launch_cooperative = [&](const void* kernel, int grid, int smem,
                                CooperativeLaunchState& state, int threads) {
    const int device = x.device().device_id;
    auto submit = [&] {
      void* args[] = {&px, &pdt, &pA, &pB, &pC, &pD, &pdb, &py, &T, &NH, &nchunks, &tmaB, &tmaC};
      const cudaError_t launch_status =
          cudaLaunchCooperativeKernel(kernel, dim3(grid), dim3(threads), args, smem, stream);
      const cudaError_t last_status = cudaGetLastError();
      TVM_FFI_CHECK_CUDA_ERROR(launch_status);
      TVM_FFI_CHECK_CUDA_ERROR(last_status);
    };
    bool submitted = false;
    std::call_once(state.initialize, [&] {
      TVM_FFI_CHECK_CUDA_ERROR(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
      int cooperative = 0, sms = 0, blocks_per_sm = 0;
      TVM_FFI_CHECK_CUDA_ERROR(
          cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device));
      TVM_FFI_ICHECK(cooperative != 0)
          << "SSD group-Gram kernel requires cooperative-launch support";
      TVM_FFI_CHECK_CUDA_ERROR(
          cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
      TVM_FFI_CHECK_CUDA_ERROR(
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, threads, smem));
      // Validate this first launch before committing the state: an exception
      // here leaves the once-flag retriable, so a failed first use (e.g. a
      // device that cannot hold the whole cooperative grid) does not pin the
      // kernel's launch state to a device it never launched on.
      const int capacity = blocks_per_sm * sms;
      TVM_FFI_ICHECK_LE(grid, capacity)
          << "SSD cooperative grid exceeds whole-grid residency capacity";
      TVM_FFI_ICHECK_GT(2 * grid, capacity)
          << "SSD ring is unsafe when two cooperative grids can co-reside";
      submit();  // A submission error must leave the once-flag retriable too.
      submitted = true;
      state.device = device;
      state.resident_capacity = capacity;
    });
    if (submitted) return;
    TVM_FFI_ICHECK_EQ(state.device, device)
        << "SSD cooperative launch state was initialized on another device";
    TVM_FFI_ICHECK_LE(grid, state.resident_capacity)
        << "SSD cooperative grid exceeds whole-grid residency capacity";
    TVM_FFI_ICHECK_GT(2 * grid, state.resident_capacity)
        << "SSD ring is unsafe when two cooperative grids can co-reside";

    submit();
  };

  // Eight group-Gram producers ride on the SMs left idle by the 128 consumer
  // CTAs.  Cooperative launch makes all 136 blocks resident before execution.

  if (NH >= 96) {
    if ((T & 63) == 0) {
      if (nchunks > GWIN)
        launch_cooperative((const void*)ssd_kernel<64, true, 4, true, false>, NH + NGR, smem64,
                           g_coop64_full, 512);
      else
        launch_cooperative((const void*)ssd_kernel<64, true, 4, false, false>, NH + NGR, smem64,
                           g_coop64_full_noprog, 512);
    } else {
      if (nchunks > GWIN)
        launch_cooperative((const void*)ssd_kernel<64, false, 1, true, false>, NH + NGR, smem64,
                           g_coop64, 512);
      else
        launch_cooperative((const void*)ssd_kernel<64, false, 1, false, false>, NH + NGR, smem64,
                           g_coop64_noprog, 512);
    }
  } else {
    if ((T & 63) == 0) {
      if (nchunks > GWIN)
        launch_cooperative((const void*)ssd_kernel<32, true, 4, true>, NH * 2 + NGR, smem32,
                           g_coop32_full, 256);
      else
        launch_cooperative((const void*)ssd_kernel<32, true, 4, false>, NH * 2 + NGR, smem32,
                           g_coop32_full_noprog, 256);
    } else {
      if (nchunks > GWIN)
        launch_cooperative((const void*)ssd_kernel<32, false, 1, true>, NH * 2 + NGR, smem32,
                           g_coop32, 256);
      else
        launch_cooperative((const void*)ssd_kernel<32, false, 1, false>, NH * 2 + NGR, smem32,
                           g_coop32_noprog, 256);
    }
  }
}
}  // namespace regx
#undef DI

void Run(ffi::TensorView x, ffi::TensorView dt, ffi::TensorView A, ffi::TensorView Bmat,
         ffi::TensorView Cmat, ffi::TensorView D, ffi::TensorView dt_bias, ffi::TensorView y) {
  auto dev = x.device();
  TVM_FFI_ICHECK_EQ((int)dev.device_type, (int)kDLCUDA) << "x must be a CUDA tensor";
  auto same_device = [&](const ffi::TensorView& t, const char* name) {
    TVM_FFI_ICHECK(t.device().device_type == dev.device_type &&
                   t.device().device_id == dev.device_id)
        << name << " must reside on the same CUDA device as x";
  };
  same_device(dt, "dt");
  same_device(A, "A");
  same_device(Bmat, "B");
  same_device(Cmat, "C");
  same_device(D, "D");
  same_device(dt_bias, "dt_bias");
  same_device(y, "y");
  auto address = [](const ffi::TensorView& t) { return reinterpret_cast<uintptr_t>(t.data_ptr()); };
  TVM_FFI_ICHECK(((address(x) | address(Bmat) | address(Cmat) | address(y)) & 15) == 0)
      << "x, B, C and y must be 16-byte aligned";
  const size_t tokens = x.shape()[0], heads = x.shape()[1];
  const size_t dt_bytes = tokens * heads * sizeof(bf16);
  const size_t y_bytes = dt_bytes * 64, bc_bytes = tokens * 8 * 128 * sizeof(bf16);
  const uintptr_t dst = address(y);
  auto disjoint = [&](const ffi::TensorView& input, size_t bytes, const char* name) {
    const uintptr_t src = address(input);
    TVM_FFI_ICHECK(!bytes || !y_bytes || (src <= dst ? dst - src >= bytes : src - dst >= y_bytes))
        << "y must not overlap " << name;
  };
  disjoint(x, y_bytes, "x");
  disjoint(dt, dt_bytes, "dt");
  disjoint(A, heads * sizeof(float), "A");
  disjoint(Bmat, bc_bytes, "B");
  disjoint(Cmat, bc_bytes, "C");
  disjoint(D, heads * sizeof(bf16), "D");
  disjoint(dt_bias, heads * sizeof(bf16), "dt_bias");
  // The module binds its per-kernel launch state (shared-memory opt-in,
  // cooperative residency capacity) to the first CUDA device that runs it.
  // Enforce that binding here so a direct native caller on another device
  // gets this clean refusal — mirroring the Python-side guard's
  // "native_state_other_device" — instead of launching against state that
  // was configured for a different device.
  static std::atomic<int> bound_device{-1};
  static std::mutex bind_mutex;
  auto launch = [&] {
    // Launches below use this device's stream; make it current first (a
    // launch whose stream belongs to a non-current device is rejected).
    ffi::CUDADeviceGuard device_guard(dev.device_id);
    if ((int)x.shape()[1] >= 96)
      regx::Run(x, dt, A, Bmat, Cmat, D, dt_bias, y);
    else
      RunPeer(x, dt, A, Bmat, Cmat, D, dt_bias, y);
  };
  int bound = bound_device.load(std::memory_order_acquire);
  if (bound == -1) {
    // First-use path: validate and submit the launch under the lock, and
    // commit the binding only after submission succeeds.  A failed
    // first-launch validation (e.g. the cooperative residency check) must
    // leave the device unbound so a later call on an eligible device is not
    // refused by a binding that never launched anything.
    std::lock_guard<std::mutex> lock(bind_mutex);
    bound = bound_device.load(std::memory_order_relaxed);
    if (bound == -1) {
      launch();
      bound_device.store(dev.device_id, std::memory_order_release);
      return;
    }
  }
  TVM_FFI_ICHECK_EQ(bound, dev.device_id)
      << "SSD prefill native launch state is bound to CUDA device " << bound
      << "; run other devices from a separate process";
  launch();
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, Run);
