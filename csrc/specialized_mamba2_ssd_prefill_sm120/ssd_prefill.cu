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
#include <climits>
#include <cstddef>
#include <cstring>
#include <initializer_list>
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

extern "C" __device__ float __nv_exp2f(float);
namespace win1306 {
#define WI __device__ __forceinline__
using Q = long long;
struct alignas(8) Slot {
  Q pref[64];
  float step[64], row[64], write[64], xm[2][64];
  double total, dz, rounding[2][4];
  unsigned state_abs1438[2];
  // Scan-only mailbox. part holds S/X/B first, then mass/fault.
  float part[4][2][3], me[2][2], carry[2], end[2][3];
};
static_assert(sizeof(Slot) <= 2048 && offsetof(Slot, total) == 1792);
static_assert(offsetof(Slot, rounding) == 1808 && offsetof(Slot, state_abs1438) == 1872);
static_assert(offsetof(Slot, part) == 1880 && sizeof(Slot) == 2024);
WI float mul(float x, float y) {
  float z;
  asm("mul.rn.f32 %0,%1,%2;" : "=f"(z) : "f"(x), "f"(y));
  return z;
}
WI float add(float x, float y) {
  float z;
  asm("add.rn.f32 %0,%1,%2;" : "=f"(z) : "f"(x), "f"(y));
  return z;
}
WI double wide(float x) {
  double z;
  asm("cvt.f64.f32 %0,%1;" : "=d"(z) : "f"(x));
  return z;
}
WI float from(unsigned short u) { return __uint_as_float(unsigned(u) << 16); }
WI unsigned short bits(bf16 x) { return punned<unsigned short>(x); }
WI bool normal_source(unsigned short u) {
  unsigned m = u & 32767u;
  return !m || (m >= (79u << 7) && m <= (139u << 7));
}
WI bool a_source(float a) {
  unsigned u = __float_as_uint(a), m = u & 0x7fffffffu;
  return !m || ((u >> 31) && m >= 0x800000u && m < 0x7f800000u);
}
WI float checked(float x, int& bad, int mask) {
  if ((__float_as_uint(x) & 0x7fffffffu) >= 0x7b800000u) bad |= mask;  // >=2^120 or special
  return x;
}
WI unsigned short plane_bits(float x) {
  unsigned short u = bits(__float2bfloat16_rn(x));
  return (u & 0x7f80u) ? u : (u & 0x8000u);
}
WI unsigned short plane(float x, int low, int& bad, int mask) {
  checked(x, bad, mask);
  unsigned short h = plane_bits(x);
  unsigned short v = low ? plane_bits(add(x, -from(h))) : h;
  checked(from(v), bad, mask);
  return v;
}
WI unsigned word(float a, float b, int low, int& bad, int mask) {
  return unsigned(plane(a, low, bad, mask)) | (unsigned(plane(b, low, bad, mask)) << 16);
}
WI float atten(Q t, int& bad, int mask) {
  float v = __nv_exp2f(mul(__ll2float_rn(t), 0x1p-32f));
  if (!(v >= 0 && v <= 1)) bad |= mask;
  return v;
}
WI float weight(const Slot& s, int i, int j, int& bad, int mask) {
  return checked(mul(s.step[j], atten(s.pref[i] - s.pref[j], bad, mask)), bad, mask);
}
struct Budget {
  double M = 0, E = 0, mass = 0, x = 0, round = 0, peak = 0, score = 0;
};
WI double up(double x, int& bad, int mask) {
  double y = __dadd_rn(__dmul_rn(x, 1. + 0x1p-40), 0x1p-1022);
  if (!(x >= 0 && y < 0x1p120)) bad |= mask;
  return y;
}
// Called by whole warps after a retained FP32 state update.
template <int Tiles>
WI void publish1438(Slot& slot, int k, const float (&state)[Tiles][4]) {
  unsigned m = 0;
#pragma unroll
  for (int n = 0; n < Tiles; n++) {
#pragma unroll
    for (int q = 0; q < 4; q++) m = max(m, __float_as_uint(state[n][q]) & 0x7fffffffu);
  }
  m = __reduce_max_sync(~0u, m);
  if (!(threadIdx.x & 31)) atomicMax(&slot.state_abs1438[k], m);
}
WI void tighten1438(Budget& z, const Slot& slot, int k) {
  unsigned m = slot.state_abs1438[k];
  if (m >= 0x7b800000u || !(z.M >= 0. && z.M < 0x1p120 && z.E >= 0. && z.E < 0x1p120)) return;
  double a = __dadd_rn(wide(__uint_as_float(m)), z.E);
  double upper = __dadd_rn(__dmul_rn(a, 1. + 0x1p-40), 0x1p-1022);
  if (a >= 0. && upper < 0x1p120) z.M = fmin(z.M, upper);
}
WI double boost(double x, int& bad, int mask) {
  if (!(x >= 0 && x < .25)) {
    bad |= mask;
    return 0.;
  }
  return up(x / (1. - x), bad, mask);
}
WI double gamma_n(int j, int& bad, int mask) {
  double v = j * 0x1p-24;
  return up(v / (1. - v), bad, mask);
}
WI double paired(int k, int& bad, int mask) {
  return up(0x1p-16 + (1. + 3. * 0x1p-8) * gamma_n(2 * k + 1, bad, mask), bad, mask);
}
template <bool Peer>
WI float& bstat(Slot& s, bf16* tile, int i) {
  if constexpr (Peer)
    return reinterpret_cast<float*>((char*)&s + 2048)[i];
  else
    return *reinterpret_cast<float*>(tile + 72 * i + 64);
}
template <bool Peer>
WI double& cstat(Slot& s, bf16* tile, int i) {
  if constexpr (Peer)
    return reinterpret_cast<double*>((char*)&s + 2304)[i];
  else
    return *reinterpret_cast<double*>(tile + 72 * i + 68);
}
// Read the original cp.async payload, not an MMA operand/result.
template <int Units>
WI unsigned short staged_bits(uint32_t base, int row, int col) {
  unsigned short v;
  uint32_t addr = addrU<Units>(base, row, col >> 3) + 2u * (col & 7);
  asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(addr) : "memory");
  return v;
}
// Collective. Source operands are inspected as bits, not reconstructed from
// Gram, weighted-X or an absolute peak. tile padding is not TMA storage.
template <bool Peer, bool Staged = false>
WI void setup(Slot& s, bf16* tile, const bf16* x, const bf16* dt, const bf16* B, const bf16* C,
              const bf16* bias, const bf16* D, const float* A, int start, int n, int nh, int h,
              int p0, int& bad, uint32_t aB = 0, uint32_t aX = 0, int row0 = 0,
              const uint2* groupBC = nullptr) {
  static_assert(!Staged || Peer);
  constexpr int halves = Peer ? 1 : 2, all = Peer ? 1 : 3;
  int tid = threadIdx.x, l = tid & 31, w = tid >> 5;
  auto plusRU = [](float a, float b) {
    float r;
    asm("add.rp.f32 %0,%1,%2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
  };
  if (tid < 64) {
    float step = 0, inc = 0, loss = 0;
    if (tid < n) {
      auto du = bits(dt[size_t(start + tid) * nh + h]), bu = bits(bias[h]);
      if (!normal_source(du) || !normal_source(bu) || !normal_source(bits(D[h])) || !a_source(A[h]))
        bad |= all;
      float d = from(du), b = from(bu), z = add(d, b);
      if (!(z >= -32 && z <= 4096)) bad |= all;
      // Residual of two binary32 inputs summed in binary64 as in bound1220.
      loss = __double2float_ru(fabs(__dadd_rn(wide(d), wide(b)) - wide(z)));
      step = z > 20.f ? z : add(fmaxf(z, 0.f), __nv_log1pf(__nv_expf(-fabsf(z))));
      if (!(step >= 0x1p-126f && step <= 4096.f)) bad |= all;
      inc = mul(mul(A[h], step), 1.4426950408889634f);
      if (!(fabsf(inc) <= 65536.f)) {
        bad |= all;
        inc = 0.f;
      }
    }
    s.pref[tid] = __double2ll_rn(wide(inc) * 0x1p32);
    s.row[tid] = inc;
    s.step[tid] = step;
    s.write[tid] = loss;
  }
  // Each warp owns complete rows. Use original BF payload even if a later
  // MMA will erase it. Only independent source-row statistics here.
  if constexpr (!Peer) {
    // These two full warps acquired the Gram publication for this window.
    int i = tid - 64;
    if (unsigned(i) < unsigned(n)) {
      uint2 rec = groupBC[i];
      if (rec.x >> 31) bad |= all;
      bstat<false>(s, tile, i) = __uint_as_float(rec.x & 0x7fffffffu);
      cstat<false>(s, tile, i) = wide(__uint_as_float(rec.y));
    }
  }
  for (int i = w; i < n; i += blockDim.x / 32) {
    if constexpr (Peer) {
      size_t bc = (size_t(start + i) * 8 + h / (nh / 8)) * 128;
      float bm = 0, cn = 0;
      for (int j = l; j < 128; j += 32) {
        unsigned short b;
        if constexpr (Staged)
          b = staged_bits<16>(aB, row0 + i, j);
        else
          b = bits(B[bc + j]);
        auto c = bits(C[bc + j]);
        if (!normal_source(b) || !normal_source(c)) bad |= all;
        bm = fmaxf(bm, fabsf(from(b)));
        cn = plusRU(cn, fabsf(from(c)));
      }
      for (int d = 16; d; d /= 2) {
        bm = fmaxf(bm, __shfl_down_sync(~0u, bm, d));
        cn = plusRU(cn, __shfl_down_sync(~0u, cn, d));
      }
      if (!l) {
        bstat<Peer>(s, tile, i) = bm;
        cstat<Peer>(s, tile, i) = wide(cn);
      }
    }
    for (int k = 0; k < halves; k++) {
      unsigned short u;
      if constexpr (Staged)
        u = staged_bits<4>(aX, row0 + i, l);
      else
        u = bits(x[(size_t(start + i) * nh + h) * 64 + p0 + 32 * k + l]);
      if (!normal_source(u)) bad |= 1 << k;
      float xm = fabsf(from(u));
      for (int d = 16; d; d /= 2) xm = fmaxf(xm, __shfl_down_sync(~0u, xm, d));
      if (!l) s.xm[k][i] = xm;
    }
  }
  __syncthreads();
  if (w == 0) {
    Q a = s.pref[l], b = s.pref[l + 32];
    float mass = plusRU(fabsf(s.row[l]), fabsf(s.row[l + 32]));
    // Old fmax starts at +0 and ignores NaNs. Losses are magnitudes.
    auto losskey = [](float x) {
      unsigned u = __float_as_uint(x) & 0x7fffffffu;
      return u <= 0x7f800000u ? u : 0u;
    };
    unsigned dz = max(losskey(s.write[l]), losskey(s.write[l + 32]));
#pragma unroll
    for (int d = 1; d < 32; d *= 2) {
      Q x = __shfl_up_sync(~0u, a, d), y = __shfl_up_sync(~0u, b, d);
      if (l >= d) {
        a += x;
        b += y;
      }
    }
    Q carry = __shfl_sync(~0u, a, 31);
    s.pref[l] = a;
    s.pref[l + 32] = carry + b;
#pragma unroll
    for (int d = 16; d; d /= 2) mass = plusRU(mass, __shfl_down_sync(~0u, mass, d));
    dz = __reduce_max_sync(~0u, dz);
    if (!l) {
      double mag = __dmul_ru(wide(mass), 1. + 0x1p-46);
      s.total = mag == 0. ? 0. : -mag;
      s.dz = wide(__uint_as_float(dz));
      if (!(mag <= 65536.)) bad |= all;
    }
  }
  if (tid < halves) s.state_abs1438[tid] = 0u;
  if (tid < halves * 4) s.rounding[tid / 4][tid % 4] = 0.;
  __syncthreads();
  if (tid < 64) {
    s.row[tid] = atten(s.pref[tid], bad, all);
    s.write[tid] = weight(s, n - 1, tid, bad, all);  // padded ticks repeat the last prefix
  }
  __syncthreads();
}
template <bool Peer>
WI void scan(Budget& z, Slot& s, bf16* tile, int k, int start, int n, float D, int& bad) {
  int mask = 1 << k;
  auto U = [&](double a) { return up(a, bad, mask); };
  auto V = [&](double a) { return boost(a, bad, mask); };
  constexpr double f = 0x1p-126, u = 0x1p-24, v = 0x1p-53;
  double rs = U(s.dz + 12 * u + 0x1p-32), rg = U(rs + 5 * u);
  if (!(rg < .001)) bad |= mask;
  double L = fmin(128., U(fabs(s.total) + 1.)), lf = U(L * u + 6 * u), af = V(lf);
  double rr = V(U(rs + V(rg) * L + n * 0x1p-32 + 1024 * v));
  double rw = V(U(lf + 3 * u + paired(64, bad, mask)));
  double ry = V(U(lf + 2 * u + fmax(paired(128, bad, mask), gamma_n(129, bad, mask)) +
                  paired(64, bad, mask)));
  auto Qup = [&](float a) {
    if (!(a >= 0 && a <= 1)) bad |= mask;
    return fmin(1., U(wide(a) * (1 + af) + 2 * f));
  };
  double S = 0, X = 0, B = 0;
  for (int i = 0; i < n; i++) {
    S = fmax(S, wide(s.step[i]));
    X = fmax(X, wide(s.xm[k][i]));
    B = fmax(B, wide(bstat<Peer>(s, tile, i)));
  }
  if (!(fmax(S, fmax(X, B)) <= 4096)) bad |= mask;
  z.x = fmax(z.x, X);
  double Z = U(2 * (1 + double(start) * 0x1p36) + z.E), N = U(n * (1 + S) * (1 + X));
  double aw = U(64 * f * (1 + Z + N * (1 + B))), H = 0, sm = 0, inh = 0, fm = 0;
  for (int i = 0; i < n; i++) {
    double c = cstat<Peer>(s, tile, i), xi = wide(s.xm[k][i]);
    if (!(c >= 0 && c <= 128 * 4096)) bad |= mask;
    double d = Qup(atten(s.pref[i] - (i ? s.pref[i - 1] : 0), bad, mask)), p = Qup(s.row[i]);
    H = U(d * H + wide(s.step[i]) * xi * wide(bstat<Peer>(s, tile, i)));
    sm = U(p * z.M + H);
    inh = U(p * z.E);
    fm = U(sm + inh);
    double ay = U(64 * f * (128 + (1 + c) * (1 + Z) + N * (128 + (1 + c) * (1 + B))));
    z.mass = fmax(z.mass, U(c * (inh + ry * fm + rr * sm) + ay + 1024 * v * fabs(wide(D)) * xi));
    U(c * fm);
  }
  z.M = U((1 + rr) * sm + aw);
  z.E = U(inh + rw * fm + rr * sm + aw);
}

// One full warp per original rank tile. Only warp 0 commits Budget.
// Barrier 1 is scan-private; nonparticipants can proceed to the CTA barrier.
template <bool Peer>
WI void scan1423(Budget& z, Slot& s, bf16* tile, int start, int n, float D, int& priorBad) {
  constexpr int stride = Peer ? 1 : 2, width = 32 / stride;
  const int lane = threadIdx.x & 31, k = lane % stride, r = lane / stride, mask = 1 << k;
  const int w = threadIdx.x >> 5, tiles = (n + width - 1) / width;
  const int base = w * width, i = base + r, count = min(width, n - base);
  auto meet = [&]() {
    if (tiles > 1)
      asm volatile("barrier.cta.sync 1, %0;" ::"r"(32 * tiles) : "memory");
    else
      __syncwarp();
  };
  int bad = 0;
  // No .ftz. Do not substitute ordinary +, *, / or reciprocal estimates.
  auto A = [](float x, float y) {
    float t;
    asm("add.rp.f32 %0,%1,%2;" : "=f"(t) : "f"(x), "f"(y));
    return t;
  };
  auto T = [](float x, float y) {
    float t;
    asm("mul.rp.f32 %0,%1,%2;" : "=f"(t) : "f"(x), "f"(y));
    return t;
  };
  auto sublo = [](float x, float y) {
    float t;
    asm("sub.rm.f32 %0,%1,%2;" : "=f"(t) : "f"(x), "f"(y));
    return t;
  };
  auto divhi = [](float x, float y) {
    float t;
    asm("div.rp.f32 %0,%1,%2;" : "=f"(t) : "f"(x), "f"(y));
    return t;
  };
  auto F = [](double x) {
    float t;
    asm("cvt.rp.f32.f64 %0,%1;" : "=f"(t) : "d"(x));
    return t;
  };
  auto mx = [](float x, float y) {
    float t;
    asm("max.f32 %0,%1,%2;" : "=f"(t) : "f"(x), "f"(y));
    return t;
  };
  auto groupmax = [&](float x) {
    for (int d = 16; d >= stride; d /= 2) x = mx(x, __shfl_xor_sync(~0u, x, d));
    return x;
  };
  auto get = [&](float x, int rank) { return __shfl_sync(~0u, x, rank * stride + k); };
  // nextUp(1) covers both old lift constants; minsub covers both old pads.
  constexpr float g = 0x1.000002p0f, eta = 0x1p-149f, u = 0x1p-24f;
  auto U = [&](float x) {
    float y = A(T(x, g), eta);
    // Nonnegative finite y < 2^120. Accept -0 as an input.
    if (!((__float_as_uint(x) <= 0x7f800000u || __float_as_uint(x) == 0x80000000u) &&
          __float_as_uint(y) < 0x7b800000u))
      bad |= mask;
    return y;
  };
  auto V = [&](float x) {
    if (!(x >= 0.f && x < .25f)) {
      bad |= mask;
      return 0.f;
    }
    return U(divhi(x, sublo(1.f, x)));
  };
  float rs = U(A(A(F(s.dz), 12 * u), 0x1p-32f)), rg = U(A(rs, 5 * u));
  if (!(wide(rg) < .001)) bad |= mask;
  float L = fminf(128.f, U(A(F(fabs(s.total)), 1.f)));
  float lf = U(A(T(L, u), 6 * u)), af = V(lf);
  float nf = __int2float_ru(n);
  float rr = V(U(A(A(A(rs, T(V(rg), L)), T(nf, 0x1p-32f)), 0x1p-43f)));
  float p64 = F(paired(64, bad, mask));
  float rw = V(U(A(A(lf, 3 * u), p64)));
  float p128 = F(paired(128, bad, mask)), gam = F(gamma_n(129, bad, mask));
  float ry = V(U(A(A(A(lf, 2 * u), mx(p128, gam)), p64)));
  auto Qup = [&](float a) {
    if (!(a >= 0 && a <= 1)) bad |= mask;
    return fminf(1.f, U(A(T(a, A(1.f, af)), 0x1p-125f)));
  };
  float S = 0, X = 0, B = 0;
  if (r < count) {
    S = mx(S, s.step[i]);
    X = mx(X, s.xm[k][i]);
    B = mx(B, bstat<Peer>(s, tile, i));
  }
  S = groupmax(S);
  X = groupmax(X);
  B = groupmax(B);
  float M, E;
  if (tiles == 1) {
    M = F(__shfl_sync(~0u, z.M, k));
    E = F(__shfl_sync(~0u, z.E, k));
  } else {
    if (!r) {
      s.part[w][k][0] = S;
      s.part[w][k][1] = X;
      s.part[w][k][2] = B;
      if (!w) {
        s.me[k][0] = F(z.M);
        s.me[k][1] = F(z.E);
      }
    }
    meet();
    for (int t = 0; t < tiles; t++) {
      S = mx(S, s.part[t][k][0]);
      X = mx(X, s.part[t][k][1]);
      B = mx(B, s.part[t][k][2]);
    }
    M = s.me[k][0];
    E = s.me[k][1];
  }
  if (!(mx(S, mx(X, B)) <= 4096.f)) bad |= mask;
  if (!w && !r) z.x = fmax(z.x, wide(X));
  float Z = U(A(T(2.f, A(1.f, T(__int2float_ru(start), 0x1p36f))), E));
  float N = U(T(T(nf, A(1.f, S)), A(1.f, X)));
  float aw = U(T(0x1p-120f, A(A(1.f, Z), T(N, A(1.f, B)))));
  float H = 0, sm = 0, inh = 0, fm = 0, mass = 0;
  {
    float d = 0, J = 0;
    if (r < count) {
      d = Qup(atten(s.pref[i] - (i ? s.pref[i - 1] : 0), bad, mask));
      J = T(T(s.step[i], s.xm[k][i]), bstat<Peer>(s, tile, i));
    }
    float carry = 0.f;
    auto ck = [&](float x) {
      if (!(x >= 0.f && isfinite(x))) bad |= mask;
      return x;
    };
    float a = 1.f, b = 0.f;
    if (r < count) {
      if (!(d >= 0.f && d <= 1.f)) bad |= mask;
      ck(J);
      a = ck(T(g, d));
      b = ck(A(ck(T(g, J)), eta));
    }
#pragma unroll
    for (int dist = 1; dist < width; dist *= 2) {
      float al = __shfl_up_sync(~0u, a, dist * stride);
      float bl = __shfl_up_sync(~0u, b, dist * stride);
      if (r >= dist && r < count) {
        b = ck(A(ck(T(a, bl)), b));
        a = ck(T(a, al));
      }
    }
    float al = __shfl_up_sync(~0u, a, stride), bl = __shfl_up_sync(~0u, b, stride);
    // Only the last H of a nonterminal (necessarily full) tile is a
    // dependency for the next warp. Do not replace it by a composed map.
    auto rowH = [&](float in) {
      float prev = ck(in);
      if (r) prev = ck(A(ck(T(al, prev)), bl));
      return U(A(T(d, prev), J));
    };
    for (int t = 0; t + 1 < tiles; t++) {
      if (w == t && r == width - 1) s.carry[k] = rowH(carry);
      meet();
      if (w == t + 1) carry = s.carry[k];
      // Retire every successor lane's load before its last rank can overwrite.
      __syncwarp();
    }
    if (r < count) {
      H = rowH(carry);
      double cd = cstat<Peer>(s, tile, i);
      if (!(cd >= 0. && cd <= 128 * 4096)) bad |= mask;
      float c = F(cd), xi = s.xm[k][i], p = Qup(s.row[i]);
      sm = U(A(T(p, M), H));
      inh = U(T(p, E));
      fm = U(A(sm, inh));
      float ay = U(T(0x1p-120f, A(A(128.f, T(A(1.f, c), A(1.f, Z))),
                                  T(N, A(128.f, T(A(1.f, c), A(1.f, B)))))));
      mass = mx(mass,
                U(A(A(T(c, A(A(inh, T(ry, fm)), T(rr, sm))), ay), T(T(0x1p-43f, fabsf(D)), xi))));
      U(T(c, fm));
    }
  }
  int last = count - 1;
  sm = get(sm, last);
  inh = get(inh, last);
  fm = get(fm, last);
  mass = groupmax(mass);
  const unsigned faults = __ballot_sync(~0u, bad != 0);
  const unsigned members = Peer ? ~0u : (0x55555555u << k);
  if (tiles > 1) {
    // The first carry rendezvous retired every part[][][] maxima reader.
    if (!r) {
      s.part[w][k][0] = mass;
      s.part[w][k][1] = float((faults & members) != 0);
      if (w == tiles - 1) {
        s.end[k][0] = sm;
        s.end[k][1] = inh;
        s.end[k][2] = fm;
      }
    }
    meet();
    if (!w && !r) {
      for (int t = 0; t < tiles; t++) {
        mass = mx(mass, s.part[t][k][0]);
        if (s.part[t][k][1] != 0.f) bad |= mask;
      }
      sm = s.end[k][0];
      inh = s.end[k][1];
      fm = s.end[k][2];
    }
  }
  if (!w && !r) {
    if (faults & members) bad |= mask;
    z.mass = fmax(z.mass, wide(mass));
    z.M = wide(U(A(T(A(1.f, rr), sm), aw)));
    z.E = wide(U(A(A(A(inh, T(rw, fm)), T(rr, sm)), aw)));
    priorBad |= bad;
  }
}

struct Peaks {
  float pack = 0, skip = 0, raw = 0, y = 0;
  WI unsigned short finish(float a, float x, float D, int& bad, int mask) {
    float b = checked(mul(D, x), bad, mask);
    if ((__float_as_uint(D) & 0x7fffffff) && (__float_as_uint(x) & 0x7fffffff) &&
        (__float_as_uint(b) & 0x7fffffff) < 0x00800000u)
      bad |= mask;
    checked(a, bad, mask);
    float r = checked(add(a, b), bad, mask);
    auto u = bits(__float2bfloat16_rn(r));
    float q = checked(from(u), bad, mask);
    pack = fmaxf(pack, fabsf(add(q, -r)));
    skip = fmaxf(skip, fabsf(b));
    raw = fmaxf(raw, fabsf(a));
    y = fmaxf(y, fabsf(q));
    return u;
  }
  WI void reduce(Slot& s, int k) {
    float a[4] = {pack, skip, raw, y};
    for (int j = 0; j < 4; j++) {
      for (int d = 16; d; d /= 2) a[j] = fmaxf(a[j], __shfl_down_sync(~0u, a[j], d));
      if (!(threadIdx.x & 31))
        atomicMax((unsigned long long*)&s.rounding[k][j],
                  (unsigned long long)__double_as_longlong(wide(a[j])));
    }
  }
};
WI void collect(Budget& z, const Slot& s, int k) {
  const double* a = s.rounding[k];
  z.peak = fmax(z.peak, a[3]);
  z.round = fmax(z.round, a[0] + 0x1p-23 * (a[1] + a[2]));
}
WI bool refuses(const Budget& z, double row) {
  auto nn = [](double v) { return isfinite(v) && v >= 0.; };
  if (!(nn(z.mass) && nn(row) && nn(z.x) && nn(z.round) && nn(z.peak))) return true;

  // Positive score sums use upward FP32 additions, widened exactly. Retain
  // the original lift; collect's pack/skip/raw expression is still RN.
  // These are bounds on the actual aggregates, not an absolute surrogate.
  constexpr double lift = 1. + 0x1p-40;
  double score = __dmul_ru(row, lift), rounding = __dmul_ru(z.round, lift);
  double c = __dadd_ru(__dmul_ru(__ddiv_ru(1., 62500.), score), 64. * 4098 * 0x1p-126);
  if (!(nn(score) && nn(rounding) && nn(c))) return true;
  double e = __dadd_ru(__dadd_ru(z.mass, __dmul_ru(c, z.x)), rounding);
  double cap = __dmul_rd(__ddiv_rd(3., 200.), z.peak);
  if (!(nn(e) && nn(cap) && e <= cap)) return true;

  constexpr double u = 0x1p-8, eta = 0x1p-133;
  double upper = __dadd_ru(z.peak, e);
  // Ensure the reference rounds to finite BF16 even at the upper endpoint.
  if (!(nn(upper) && upper <= 0x1.fep127)) return true;
  double L0 = fmax(0., __dsub_rd(z.peak, e));
  double B1 = __dadd_ru(e, __dadd_ru(__dmul_ru(u, upper), eta));
  double L1 = fmax(0., __dsub_rd(__dmul_rd(1. - u, L0), eta));
  return !(nn(L0) && nn(B1) && nn(L1) && e <= __ddiv_rd(L0, 50.) && B1 <= __ddiv_rd(L1, 50.));
}
// All lanes enter or none do. Parallel across p, strictly serial across n
// and t. Shared coefficients are rebuilt from original bits ONLY on refusal.
__device__ __noinline__ void ordered(const bf16* x, const bf16* dt, const float* A, const bf16* B,
                                     const bf16* C, const bf16* D, const bf16* bias, bf16* y, int T,
                                     int nh, int h, int p0, char* mem) {
  int l = threadIdx.x, g = h / (nh / 8);
  double* st = (double*)mem;
  double *steps = st + 4096, *dec = steps + 64;
  for (int i = l; i < 4096; i += blockDim.x) st[i] = 0.;
  double db = wide(from(bits(bias[h]))), dh = wide(from(bits(D[h]))), ah = wide(A[h]);
  for (int t = 0; t < T; t += 64) {
    int count = min(64, T - t);
    if (l < count) {
      double v = __dadd_rn(wide(from(bits(dt[size_t(t + l) * nh + h]))), db);
      double s = v > 20. ? v : log1p(exp(v));
      steps[l] = s;
      dec[l] = exp(__dmul_rn(s, ah));
    }
    __syncthreads();
    if (l < 32)
      for (int j = 0; j < count; j++) {
        size_t off = (size_t(t + j) * nh + h) * 64 + p0 + l, bc = (size_t(t + j) * 8 + g) * 128;
        double xv = wide(from(bits(x[off]))), dx = __dmul_rn(steps[j], xv), sum = 0.;
        for (int i = 0; i < 128; i++) {
          double old = st[i * 32 + l], inj = __dmul_rn(dx, wide(from(bits(B[bc + i]))));
          double a = old == 0. ? inj : __dadd_rn(__dmul_rn(old, dec[j]), inj);
          st[i * 32 + l] = a;
          sum = __dadd_rn(sum, __dmul_rn(a, wide(from(bits(C[bc + i])))));
        }
        y[off] = __double2bfloat16(__dadd_rn(sum, __dmul_rn(dh, xv)));
      }
    __syncthreads();
  }
}
constexpr int SH = 62464;
constexpr int DualSH = 78880;
static_assert(sizeof(size_t) == 8);
template <int W>
struct alignas(16) Scratch {
  bf16 b[64][128], c[64][128], x[64][32];
  double step[64], dec[64], dx[64][32], mail[2][W][32];
};
using Max = Scratch<16>;
static_assert(offsetof(Max, c) == 16384 && offsetof(Max, x) == 32768 &&
              offsetof(Max, step) == 36864 && offsetof(Max, dec) == 37376 &&
              offsetof(Max, dx) == 37888 && offsetof(Max, mail) == 54272 && sizeof(Max) == SH &&
              sizeof(Scratch<8>) == 58368);
#define ARGS                                                                                  \
  const bf16 *x, const bf16 *dt, const float *A, const bf16 *B, const bf16 *C, const bf16 *D, \
      const bf16 *bias, bf16 *y
#define PTRS x, dt, A, B, C, D, bias, y

template <int W>
__device__ __noinline__ void pipeline(ARGS, int T, int nh, int h, int p0, char* mem) {
  static_assert(W == 8 || W == 16);
  constexpr int K = 128 / W;
  int l = threadIdx.x, w = l >> 5, p = l & 31, g = h / (nh / 8);
  auto& q = *reinterpret_cast<Scratch<W>*>(mem);
  double state[K] = {};
  double db = wide(from(bits(bias[h]))), dh = wide(from(bits(D[h]))), ah = wide(A[h]);
  for (int t = 0; t < T;) {
    int n = min(64, T - t);
    if (l < n) {
      double v = __dadd_rn(wide(from(bits(dt[size_t(t + l) * nh + h]))), db);
      double s = v > 20. ? v : log1p(exp(v));
      q.step[l] = s;
      q.dec[l] = exp(__dmul_rn(s, ah));
    }
    for (int i = l; i < n * 128; i += 32 * W) {
      int j = i / 128, a = i % 128;
      size_t bc = (size_t(t + j) * 8 + g) * 128 + a;
      q.b[j][a] = B[bc];
      q.c[j][a] = C[bc];
    }
    __syncthreads();
    for (int j = w; j < n; j += W) {
      auto v = x[(size_t(t + j) * nh + h) * 64 + p0 + p];
      q.x[j][p] = v;
      q.dx[j][p] = __dmul_rn(q.step[j], wide(from(bits(v))));
    }
    __syncthreads();
    for (int s = 0; s < n + W - 1; s++) {
      int j = s - w;
      if (unsigned(j) < unsigned(n)) {
        double sum = w ? q.mail[(s & 1) ^ 1][w - 1][p] : 0.;
        double dx = q.dx[j][p], dec = q.dec[j];
#pragma unroll
        for (int a = 0; a < K; a++) {
          int i = w * K + a;
          double old = state[a], inj = __dmul_rn(dx, wide(from(bits(q.b[j][i]))));
          double v = old == 0. ? inj : __dadd_rn(__dmul_rn(old, dec), inj);
          state[a] = v;
          sum = __dadd_rn(sum, __dmul_rn(v, wide(from(bits(q.c[j][i])))));
        }
        if (w != W - 1)
          q.mail[s & 1][w][p] = sum;
        else {
          double xv = wide(from(bits(q.x[j][p])));
          y[(size_t(t + j) * nh + h) * 64 + p0 + p] =
              __double2bfloat16(__dadd_rn(sum, __dmul_rn(dh, xv)));
        }
      }
      __syncthreads();
    }
    t += n;
  }
}

#undef PTRS
#undef ARGS
#define ARGS                                                                                  \
  const bf16 *x, const bf16 *dt, const float *A, const bf16 *B, const bf16 *C, const bf16 *D, \
      const bf16 *bias, bf16 *y
#define PTRS x, dt, A, B, C, D, bias, y

struct alignas(16) Cert {
  unsigned short bb[64][128], cc[64][128], xx[64][32];
  float sf[64], df[64], S[64], J[64], ds[64], dd[64];
  float cn[64], xm[64], bm[64], tops[16][4];
  int fault;
};
static_assert(alignof(Cert) == 16 && offsetof(Cert, cc) == 16384 && offsetof(Cert, xx) == 32768 &&
              offsetof(Cert, sf) == 36864 && sizeof(Cert) == 39440 && sizeof(Cert) < SH);
static_assert(2 * sizeof(Cert) == DualSH && DualSH % 16 == 0 && sizeof(Max) <= DualSH &&
              DualSH <= 100376);
#define I __device__ __forceinline__

// Test before reuse, not just at the final radius: zero factors must
// never erase a finite but out-of-envelope operand.
struct Out1530 {
  int bad = 0;
  I float ck(float z) {
    if (!(z >= 0.f && z < 0x1p100f)) bad = 4;
    return z;
  }
  I float A(float x, float y) { return ck(__fadd_ru(x, y)); }
  I float V(float x, float y) { return ck(__fmul_ru(x, y)); }
  I float rad(float x, float u) { return A(V(x, u), 0x1p-149f); }
};
I float radius(float S, float J, float ds, float dd, float sf, float df, float X, float B, float C,
               float D, float M, float& E, float N, int& bad) {
  Out1530 o;
  constexpr float uf = 0x1p-24f, ur = 0x1p-53f;
  o.ck(X);
  o.ck(B);
  o.ck(C);
  o.ck(M);
  o.ck(E);
  o.ck(N);
  o.ck(S);
  o.ck(J);
  o.ck(ds);
  o.ck(dd);
  float xf = o.V(sf, X), xr = o.V(S, X);
  float fx = o.rad(xf, uf), rx = o.rad(xr, ur);
  float dx = o.A(o.V(ds, X), o.A(fx, rx));
  float fi = o.V(o.A(xf, fx), B), ri = o.V(o.A(xr, rx), B);
  float ifr = o.rad(fi, uf), irr = o.rad(ri, ur);
  float ei = o.A(o.V(dx, B), o.A(ifr, irr));
  float of = o.V(M, df), ore = o.V(o.A(M, E), J);
  float off = o.rad(of, uf), orr = o.rad(ore, ur);
  float eo = o.A(o.A(o.V(J, E), o.V(M, dd)), o.A(off, orr));
  float af = o.A(o.A(of, off), o.A(fi, ifr));
  float ar = o.A(o.A(ore, orr), o.A(ri, irr));
  E = o.A(o.A(ei, eo), o.A(o.rad(af, uf), o.rad(ar, ur)));
  float skip = o.V(fabsf(D), X);
  float mf = o.A(o.V(C, N), skip), ne = o.A(N, E), mr = o.A(o.V(C, ne), skip);
  constexpr float res = 520 * 0x1p-149f;
  float pf = o.A(o.V(18 * 0x1p-24f, mf), res);
  float pr = o.A(o.V(260 * 0x1p-53f, mr), res);
  float R = o.A(o.V(C, E), o.A(pf, pr));
  bad |= o.bad;
  return R;
}
I float warpmax(float v) {
  for (int d = 16; d; d /= 2) v = fmaxf(v, __shfl_down_sync(~0u, v, d));
  return v;
}
template <int W, bool Audit, bool Dual = false>
__device__ __noinline__ int tentative(ARGS, int T, int nh, int h, int k, char* mem, double* stats,
                                      float* output) {
  static_assert(!Dual || W == 8);
  const int group = Dual ? int(threadIdx.x) / 256 : 0;
  const int tid = Dual ? int(threadIdx.x) % 256 : int(threadIdx.x);
  if constexpr (Dual) k = group;
  auto& q = *reinterpret_cast<Cert*>(mem + group * sizeof(Cert));
  const int l = tid % 32, w = tid / 32, g = h / (nh / 8);
  constexpr int NP = 32 / W;
  float st[NP][4] = {};
  const float Dv = from(bits(D[h])), bv = from(bits(bias[h]));
  int bad = 0;
  float E = 0, M = 0, top[4] = {};
  if (!tid)
    q.fault =
        (!normal_source(bits(D[h])) || !normal_source(bits(bias[h])) || !a_source(A[h])) ? 1 : 0;
  __syncthreads();
  for (int base = 0; base < T;) {
    int count = min(64, T - base);
    if (tid < count) {
      auto a = bits(dt[size_t(base + tid) * nh + h]);
      double z = __dadd_rn(wide(from(a)), wide(bv));
      double s = z > 20. ? z : log1p(exp(z)), d = exp(__dmul_rn(s, wide(A[h])));
      float sf = __double2float_rn(s), df = __double2float_rn(d);
      q.sf[tid] = sf;
      q.df[tid] = df;
      q.S[tid] = __double2float_ru(s);
      q.J[tid] = __double2float_ru(d);
      q.ds[tid] = __double2float_ru(nextafter(fabs(__dsub_rn(s, double(sf))), CUDART_INF));
      q.dd[tid] = __double2float_ru(nextafter(fabs(__dsub_rn(d, double(df))), CUDART_INF));
      int fault = (!normal_source(a) ? 1 : 0) |
                  (!(z >= -32 && z <= 4096 && s >= 0 && s <= 4096 && d >= 0 && d <= 1) ? 2 : 0);
      if (fault) atomicOr(&q.fault, fault);
    }
    // Payload copies only. Rows >= count are neither produced nor consumed.
    for (int u = tid; u < count * 128; u += 32 * W) {
      int j = u / 128, i = u % 128;
      size_t bc = (size_t(base + j) * 8 + g) * 128 + i;
      q.bb[j][i] = bits(B[bc]);
      q.cc[j][i] = bits(C[bc]);
    }
    for (int u = tid; u < count * 32; u += 32 * W) {
      int j = u / 32, p = u % 32;
      q.xx[j][p] = bits(x[(size_t(base + j) * nh + h) * 64 + 32 * k + p]);
    }
    __syncthreads();
    for (int j = w; j < count; j += W) {
      float c[4], bm = 0;
      int fault = 0;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        auto bi = q.bb[j][4 * l + i], ci = q.cc[j][4 * l + i];
        if (!normal_source(bi) || !normal_source(ci)) fault = 1;
        c[i] = from(ci);
        bm = fmaxf(bm, fabsf(from(bi)));
      }
      auto xi = q.xx[j][l];
      if (!normal_source(xi)) fault = 1;
      float xm = warpmax(fabsf(from(xi)));
      bm = warpmax(bm);
      float cn =
          __fadd_ru(__fadd_ru(fabsf(c[0]), fabsf(c[1])), __fadd_ru(fabsf(c[2]), fabsf(c[3])));
      for (int a = 1; a < 32; a *= 2) cn = __fadd_ru(cn, __shfl_down_sync(~0u, cn, a));
      unsigned src = __ballot_sync(~0u, fault != 0);
      if (!l) {
        q.cn[j] = cn;
        q.bm[j] = bm;
        q.xm[j] = xm;
        if (src) atomicOr(&q.fault, 1);
      }
    }
    __syncthreads();
    for (int j = 0; j < count; j++) {
      size_t off = (size_t(base + j) * nh + h) * 64 + 32 * k;
      float sf = q.sf[j], df = q.df[j];
      float b[4], c[4], m = 0, raw[NP];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        b[i] = from(q.bb[j][4 * l + i]);
        c[i] = from(q.cc[j][4 * l + i]);
      }
#pragma unroll
      for (int a = 0; a < NP; a++) {
        int col = w + a * W;
        float xv = from(q.xx[j][col]), dx = mul(sf, xv), pr[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
          float old = st[a][i], inj = mul(dx, b[i]);
          float v = old == 0 ? inj : add(mul(old, df), inj);
          st[a][i] = v;
          pr[i] = mul(v, c[i]);
          m = fmaxf(m, fabsf(v));
          if (!isfinite(v)) bad |= 4;
        }
        float sum = add(add(pr[0], pr[1]), add(pr[2], pr[3]));
#pragma unroll
        for (int dist = 1; dist < 32; dist *= 2) sum = add(sum, __shfl_down_sync(~0u, sum, dist));
        if (!l) raw[a] = add(sum, mul(Dv, xv));
      }
      float N = warpmax(m);
      if (!l) {
        float R = radius(q.S[j], q.J[j], q.ds[j], q.dd[j], sf, df, q.xm[j], q.bm[j], q.cn[j], Dv, M,
                         E, N, bad);
        M = N;
        Out1530 o;
        float P = 0, qe = 0;
#pragma unroll
        for (int a = 0; a < NP; a++) {
          int col = w + a * W;
          float y0 = raw[a];
          bf16 packed = __float2bfloat16_rn(y0);
          float v = from(bits(packed));
          if (!(isfinite(v) && isfinite(y0))) bad |= 4;
          y[off + col] = packed;
          if constexpr (Audit)
            output[size_t(base + j) * (Dual ? 64 : 32) + (Dual ? 32 * k : 0) + col] = y0;
          P = fmaxf(P, fabsf(y0));
          qe = fmaxf(qe, fmaxf(__fsub_ru(v, y0), __fsub_ru(y0, v)));
        }
        o.ck(P);
        o.ck(qe);
        float B0 = o.A(R, qe);
        float L0 = o.ck(fmaxf(0.f, __fsub_rd(P, R)));
        float pmax = o.A(P, R);
        float B1 = o.A(B0, o.A(o.V(0x1p-8f, pmax), 0x1p-133f));
        float L1 = o.ck(fmaxf(0.f, __fsub_rd(__fmul_rd(1.f - 0x1p-8f, L0), 0x1p-133f)));
        top[0] = fmaxf(top[0], B0);
        top[1] = fmaxf(top[1], B1);
        top[2] = fmaxf(top[2], L0);
        top[3] = fmaxf(top[3], L1);
        bad |= o.bad;
      }
    }
    __syncthreads();  // all window readers retired
    base += count;
  }
  int wb = __reduce_or_sync(~0u, bad);
  if (!l) {
    for (int i = 0; i < 4; i++) q.tops[w][i] = top[i];
    if (wb) atomicOr(&q.fault, wb);
  }
  __syncthreads();
  if (!tid) {
    for (int a = 1; a < W; a++)
      for (int i = 0; i < 4; i++) top[i] = fmaxf(top[i], q.tops[a][i]);
    if (!q.fault && !(top[0] <= __fdiv_rd(top[2], 50.f) && top[1] <= __fdiv_rd(top[3], 50.f)))
      q.fault = 8;
    if constexpr (Audit) {
      auto z = stats + (2 * h + k) * 5;
      z[0] = q.fault;
      for (int i = 0; i < 4; i++) z[i + 1] = top[i];
    }
  }
  __syncthreads();
  int verdict = q.fault;
  __syncthreads();
  return verdict;
}

// Separate from Out1530: auxiliary coefficients have no 2^100 domain cap.
struct Aux1734 {
  int bad = 0;
  I float ck(float z) {
    if (!(z >= 0 && isfinite(z))) bad = 4;
    return z;
  }
  I float A(float x, float y) {
    ck(x);
    ck(y);
    return ck(__fadd_ru(x, y));
  }
  I float V(float x, float y) {
    ck(x);
    ck(y);
    return ck(__fmul_ru(x, y));
  }
};
template <int W, bool Audit, bool Dual = false>
__device__ __noinline__ int tentative_parallel(ARGS, int T, int nh, int h, int k, char* mem,
                                               double* stats, float* output) {
  static_assert(!Dual || W == 8);
  const int group = Dual ? int(threadIdx.x) / 256 : 0;
  const int tid = Dual ? int(threadIdx.x) % 256 : int(threadIdx.x);
  if constexpr (Dual) k = group;
  auto& q = *reinterpret_cast<Cert*>(mem + group * sizeof(Cert));
  const int l = tid % 32, w = tid / 32, g = h / (nh / 8);
  constexpr int NP = 32 / W;
  static_assert(W == 8 || W == 16);
  constexpr int K = 128 / W;
  const int rank = l % W, col = w + (l / W) * W;
  float st[K] = {};
  const float Dv = from(bits(D[h])), bv = from(bits(bias[h]));
  int bad = 0;
  float E = 0, M = 0, top[4] = {};
  if (!tid)
    q.fault =
        (!normal_source(bits(D[h])) || !normal_source(bits(bias[h])) || !a_source(A[h])) ? 1 : 0;
  __syncthreads();
  for (int base = 0; base < T;) {
    int count = min(64, T - base);
    if (tid < count) {
      auto a = bits(dt[size_t(base + tid) * nh + h]);
      double z = __dadd_rn(wide(from(a)), wide(bv));
      double s = z > 20. ? z : log1p(exp(z)), d = exp(__dmul_rn(s, wide(A[h])));
      float sf = __double2float_rn(s), df = __double2float_rn(d);
      q.sf[tid] = sf;
      q.df[tid] = df;
      q.S[tid] = __double2float_ru(s);
      q.J[tid] = __double2float_ru(d);
      q.ds[tid] = __double2float_ru(nextafter(fabs(__dsub_rn(s, double(sf))), CUDART_INF));
      q.dd[tid] = __double2float_ru(nextafter(fabs(__dsub_rn(d, double(df))), CUDART_INF));
      int fault = (!normal_source(a) ? 1 : 0) |
                  (!(z >= -32 && z <= 4096 && s >= 0 && s <= 4096 && d >= 0 && d <= 1) ? 2 : 0);
      if (fault) atomicOr(&q.fault, fault);
    }
    // Payload copies only. Rows >= count are neither produced nor consumed.
    for (int u = tid; u < count * 128; u += 32 * W) {
      int j = u / 128, i = u % 128;
      size_t bc = (size_t(base + j) * 8 + g) * 128 + i;
      q.bb[j][i] = bits(B[bc]);
      q.cc[j][i] = bits(C[bc]);
    }
    for (int u = tid; u < count * 32; u += 32 * W) {
      int j = u / 32, p = u % 32;
      q.xx[j][p] = bits(x[(size_t(base + j) * nh + h) * 64 + 32 * k + p]);
    }
    __syncthreads();
    for (int j = w; j < count; j += W) {
      float c[4], bm = 0;
      int fault = 0;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        auto bi = q.bb[j][4 * l + i], ci = q.cc[j][4 * l + i];
        if (!normal_source(bi) || !normal_source(ci)) fault = 1;
        c[i] = from(ci);
        bm = fmaxf(bm, fabsf(from(bi)));
      }
      auto xi = q.xx[j][l];
      if (!normal_source(xi)) fault = 1;
      float xm = warpmax(fabsf(from(xi)));
      bm = warpmax(bm);
      float cn =
          __fadd_ru(__fadd_ru(fabsf(c[0]), fabsf(c[1])), __fadd_ru(fabsf(c[2]), fabsf(c[3])));
      for (int a = 1; a < 32; a *= 2) cn = __fadd_ru(cn, __shfl_down_sync(~0u, cn, a));
      unsigned src = __ballot_sync(~0u, fault != 0);
      if (!l) {
        q.cn[j] = cn;
        q.bm[j] = bm;
        q.xm[j] = xm;
        if (src) atomicOr(&q.fault, 1);
      }
    }
    __syncthreads();
    float n0 = 0, p0 = 0, e0 = 0, n1 = 0, p1 = 0, e1 = 0;
    for (int j = 0; j < count; j++) {
      size_t off = (size_t(base + j) * nh + h) * 64 + 32 * k;
      float sf = q.sf[j], df = q.df[j];
      // Each subwarp owns one old output column. Local sums and subwarp
      // shuffles reproduce the original balanced 128-leaf projection tree.
      float xv = from(q.xx[j][col]), dx = mul(sf, xv);
      float part[K / 4], m = 0.f;
      static_assert(CHAR_BIT == 8 && sizeof(unsigned short) == 2 && sizeof(uint32_t) == 4);
      static_assert(alignof(Cert) == 16 && sizeof(Cert) == 39440 && sizeof(Cert) % 16 == 0);
      static_assert(offsetof(Cert, bb) == 0 && offsetof(Cert, cc) == 16384);
      static_assert(sizeof(q.bb[0]) == 256 && sizeof(q.cc[0]) == 256);
      static_assert(W * K == 128 && K % 8 == 0 && (K * sizeof(unsigned short)) % 16 == 0);
#pragma unroll
      for (int g8 = 0; g8 < K / 8; g8++) {
        const int idx = rank * K + 8 * g8;
        uint32_t bw[4], cw[4];
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(bw[0]), "=r"(bw[1]), "=r"(bw[2]), "=r"(bw[3])
                     : "r"(peer_sptr(&q.bb[j][idx]))
                     : "memory");
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(cw[0]), "=r"(cw[1]), "=r"(cw[2]), "=r"(cw[3])
                     : "r"(peer_sptr(&q.cc[j][idx]))
                     : "memory");
#pragma unroll
        for (int half = 0; half < 2; half++) {
          const int g4 = 2 * g8 + half;
          float pr[4];
#pragma unroll
          for (int i = 0; i < 4; i++) {
            const int word = 2 * half + i / 2, shift = 16 * (i & 1);
            const auto bi = static_cast<unsigned short>(bw[word] >> shift);
            const auto ci = static_cast<unsigned short>(cw[word] >> shift);
            float old = st[4 * g4 + i], inj = mul(dx, from(bi));
            float v = old == 0.f ? inj : add(mul(old, df), inj);
            st[4 * g4 + i] = v;
            if (!isfinite(v)) bad |= 4;
            m = fmaxf(m, fabsf(v));
            pr[i] = mul(v, from(ci));
          }
          part[g4] = add(add(pr[0], pr[1]), add(pr[2], pr[3]));
        }
      }
#pragma unroll
      for (int d = 1; d < K / 4; d *= 2) {
#pragma unroll
        for (int i = 0; i < K / 4; i += 2 * d) part[i] = add(part[i], part[i + d]);
      }
      float sum = part[0];
#pragma unroll
      for (int d = 1; d < W; d *= 2) sum = add(sum, __shfl_down_sync(~0u, sum, d, W));
      float P = 0.f, qe = 0.f;
      if (!rank) {
        float y0 = add(sum, mul(Dv, xv));
        bf16 packed = __float2bfloat16_rn(y0);
        float v = from(bits(packed));
        if (!(isfinite(v) && isfinite(y0))) bad |= 4;
        y[off + col] = packed;
        if constexpr (Audit)
          output[size_t(base + j) * (Dual ? 64 : 32) + (Dual ? 32 * k : 0) + col] = y0;
        P = fmaxf(0.f, fabsf(y0));
        qe = fmaxf(0.f, fmaxf(__fsub_ru(v, y0), __fsub_ru(y0, v)));
      }
      // Only nonnegative maxima initialized at +0 reach these reductions.
      // Invalid operands retain their independent sticky fault bits.
      float N = __uint_as_float(__reduce_max_sync(~0u, __float_as_uint(m) & 0x7fffffffu));
      P = __uint_as_float(__reduce_max_sync(~0u, __float_as_uint(P) & 0x7fffffffu));
      qe = __uint_as_float(__reduce_max_sync(~0u, __float_as_uint(qe) & 0x7fffffffu));
      if (l == (j & 31)) {
        if (j < 32) {
          n0 = N;
          p0 = P;
          e0 = qe;
        } else {
          n1 = N;
          p1 = P;
          e1 = qe;
        }
      }
    }
    for (int tile = 0; tile < count; tile += 32) {
      int j = tile + l, last = min(32, count - tile) - 1;
      float N = tile ? n1 : n0, P = tile ? p1 : p0, qe = tile ? e1 : e0;
      float prev = __shfl_up_sync(~0u, N, 1);
      if (!l) prev = M;
      float a = 1, b = 0;
      Aux1734 aux;
      if (j < count) {
        float S = q.S[j], J = q.J[j], sf = q.sf[j], df = q.df[j], ds = q.ds[j], dd = q.dd[j];
        float X = q.xm[j], B = q.bm[j], M = prev;
        constexpr float uf = 0x1p-24f, ur = 0x1p-53f, G = 1.f + 11.f * 0x1p-23f;
        if (!(sf >= 0 && isfinite(sf) && df >= 0 && isfinite(df) && J <= 1)) bad |= 4;
        Out1530 o;
        float xf = o.V(sf, X), xr = o.V(S, X);
        float fx = o.rad(xf, uf), rx = o.rad(xr, ur);
        float dx = o.A(o.V(ds, X), o.A(fx, rx));
        float fi = o.V(o.A(xf, fx), B), ri = o.V(o.A(xr, rx), B);
        float ifr = o.rad(fi, uf), irr = o.rad(ri, ur);
        float ei = o.A(o.V(dx, B), o.A(ifr, irr));

        float of = o.V(M, df), off = o.rad(of, uf);
        float p = o.rad(o.A(o.A(of, off), o.A(fi, ifr)), uf);
        float k = o.A(ri, irr), mdd = o.V(M, dd);
        bad |= o.bad;
        a = aux.V(G, J);
        b = aux.A(ei, mdd);
        b = aux.A(b, off);
        b = aux.A(b, p);
        b = aux.A(b, aux.V(ur, k));
        b = aux.A(b, aux.V(aux.V(J, M), 0x1.000002p-52f));
        b = aux.A(b, 3.f * 0x1p-149f);
        b = aux.V(G, aux.A(b, 1024.f * 0x1p-149f));
      }
#pragma unroll
      for (int d = 1; d < 32; d *= 2) {
        float al = __shfl_up_sync(~0u, a, d), bl = __shfl_up_sync(~0u, b, d);
        if (l >= d && j < count) {
          b = aux.A(aux.V(a, bl), b);
          a = aux.V(a, al);
        }
      }
      float al = __shfl_up_sync(~0u, a, 1), bl = __shfl_up_sync(~0u, b, 1);
      float U = E;
      if (j < count) {
        if (l) U = aux.A(aux.V(al, E), bl);
        float R = radius(q.S[j], q.J[j], q.ds[j], q.dd[j], q.sf[j], q.df[j], q.xm[j], q.bm[j],
                         q.cn[j], Dv, prev, U, N, bad);
        Out1530 o;
        o.ck(P);
        o.ck(qe);
        float B0 = o.A(R, qe);
        float L0 = o.ck(fmaxf(0.f, __fsub_rd(P, R)));
        float pmax = o.A(P, R);
        float B1 = o.A(B0, o.A(o.V(0x1p-8f, pmax), 0x1p-133f));
        float L1 = o.ck(fmaxf(0.f, __fsub_rd(__fmul_rd(1.f - 0x1p-8f, L0), 0x1p-133f)));
        top[0] = fmaxf(top[0], B0);
        top[1] = fmaxf(top[1], B1);
        top[2] = fmaxf(top[2], L0);
        top[3] = fmaxf(top[3], L1);
        bad |= o.bad;
      }
      bad |= aux.bad;
      M = __shfl_sync(~0u, N, last);
      E = __shfl_sync(~0u, U, last);
    }
    __syncthreads();  // all window readers retired
    base += count;
  }
  for (int i = 0; i < 4; i++) top[i] = warpmax(top[i]);
  int wb = __reduce_or_sync(~0u, bad);
  if (!l) {
    for (int i = 0; i < 4; i++) q.tops[w][i] = top[i];
    if (wb) atomicOr(&q.fault, wb);
  }
  __syncthreads();
  if (!tid) {
    for (int a = 1; a < W; a++)
      for (int i = 0; i < 4; i++) top[i] = fmaxf(top[i], q.tops[a][i]);
    if (!q.fault && !(top[0] <= __fdiv_rd(top[2], 50.f) && top[1] <= __fdiv_rd(top[3], 50.f)))
      q.fault = 8;
    if constexpr (Audit) {
      auto z = stats + (2 * h + k) * 5;
      z[0] = q.fault;
      for (int i = 0; i < 4; i++) z[i + 1] = top[i];
    }
  }
  __syncthreads();
  int verdict = q.fault;
  __syncthreads();
  return verdict;
}

template <int W, bool Dual = false>
__device__ __noinline__ int tentative_recover(ARGS, int T, int nh, int h, int k, char* mem) {
  int verdict = tentative_parallel<W, false, Dual>(PTRS, T, nh, h, k, mem, nullptr, nullptr);
  if constexpr (Dual) {
    // Whole CTA recovery: both half groups must enter every shared barrier.
    if (__syncthreads_or(verdict != 0))
      verdict = tentative<W, false, true>(PTRS, T, nh, h, k, mem, nullptr, nullptr);
  } else {
    if (verdict) verdict = tentative<W, false>(PTRS, T, nh, h, k, mem, nullptr, nullptr);
  }
  return verdict;
}

struct alignas(16) Record2051 {
  int fault;
  float top[4];
  int pad[3];
};
static_assert(sizeof(Record2051) == 32 && sizeof(int) == 4);
WI Record2051* rec2051(int* routes, int nh, int id) {
  return reinterpret_cast<Record2051*>(routes + 2 * nh) + 16 * id;
}
template <int W, int GROUP>
struct alignas(16) GroupWindow {
  static_assert((W == 8 || W == 16) && (GROUP == 2 || GROUP == 4) && W % GROUP == 0);
  Cert cert;
  float row[GROUP][3][32 / W][64];
  int owner_bad[GROUP];
};
static __device__ __forceinline__ void load_bf16_bits8(unsigned short (&dst)[4], const bf16* src) {
  unsigned long long bits;
  static_assert(sizeof(bits) == 8 && sizeof(dst) == 8 && sizeof(bf16) == 2,
                "expected an eight-byte, four-bf16 payload");
  asm("ld.global.u64 %0, [%1];"
      : "=l"(bits)
      : "l"(static_cast<unsigned long long>(__cvta_generic_to_global(src)))
      : "memory");
  memcpy(dst, &bits, sizeof(bits));
}

template <int W, int GROUP>
__device__ __forceinline__ void grouped2051(ARGS, int T, int nh, Record2051* records) {
  const int id = blockIdx.y, h = id / 2, k = id % 2;
  static_assert((W == 8 && GROUP == 2) || (W == 16 && GROUP == 4));

  constexpr int NP = 32 / W;
  using Window = GroupWindow<W, GROUP>;
  constexpr int NW = GROUP * NP, NT = 32 * NW;
  static_assert(NT == 256 && sizeof(Window) == 45600 && W / GROUP == 4);
  static_assert(offsetof(Window, row) == sizeof(Cert));
  static_assert(offsetof(Window, owner_bad) == sizeof(Cert) + GROUP * 3 * NP * 64 * sizeof(float));
  static_assert(sizeof(Window) == offsetof(Window, owner_bad) + 16 && sizeof(Window) <= 51744);
  extern __shared__ __align__(16) unsigned char storage2043[];
  Window& window = *reinterpret_cast<Window*>(storage2043);
  auto& q = window.cert;
  const int tid = threadIdx.x, l = tid % 32, physical = tid / 32;
  const int localOwner = physical / NP, w = physical % NP;
  const int owner = int(blockIdx.x) * GROUP + localOwner;
  const int col = owner + w * W, g = h / (nh / 8);
  float st[4] = {};
  const float Dv = from(bits(D[h])), bv = from(bits(bias[h]));
  int bad = 0;
  float E = 0, M = 0, top[4] = {};
  // Only sticky words need clearing. Each valid payload/stat/row cell
  // has one producer before its first read; tail cells are never read.
  if (tid < GROUP) window.owner_bad[tid] = 0;
  if (!tid)
    q.fault =
        (!normal_source(bits(D[h])) || !normal_source(bits(bias[h])) || !a_source(A[h])) ? 1 : 0;
  __syncthreads();  // publish source and owner fault initialization
  for (int base = 0; base < T;) {
    int count = min(64, T - base);
    if (tid < count) {
      auto a = bits(dt[size_t(base + tid) * nh + h]);
      double z = __dadd_rn(wide(from(a)), wide(bv));
      double s = z > 20. ? z : log1p(exp(z)), d = exp(__dmul_rn(s, wide(A[h])));
      float sf = __double2float_rn(s), df = __double2float_rn(d);
      q.sf[tid] = sf;
      q.df[tid] = df;
      q.S[tid] = __double2float_ru(s);
      q.J[tid] = __double2float_ru(d);
      q.ds[tid] = __double2float_ru(nextafter(fabs(__dsub_rn(s, double(sf))), CUDART_INF));
      q.dd[tid] = __double2float_ru(nextafter(fabs(__dsub_rn(d, double(df))), CUDART_INF));
      int fault = (!normal_source(a) ? 1 : 0) |
                  (!(z >= -32 && z <= 4096 && s >= 0 && s <= 4096 && d >= 0 && d <= 1) ? 2 : 0);
      if (fault) atomicOr(&q.fault, fault);
    }
    // One warp stages and validates one complete logical row.
    for (int j = physical; j < count; j += NW) {
      static_assert(sizeof(bf16) == 2 && sizeof(unsigned short) == 2);
      alignas(8) unsigned short bs[4], cs[4];
      size_t bc = (size_t(base + j) * 8 + g) * 128 + 4 * l;
      load_bf16_bits8(bs, B + bc);
      load_bf16_bits8(cs, C + bc);
      memcpy(&q.bb[j][4 * l], bs, sizeof(bs));
      memcpy(&q.cc[j][4 * l], cs, sizeof(cs));
      auto xi = bits(x[(size_t(base + j) * nh + h) * 64 + 32 * k + l]);
      q.xx[j][l] = xi;
      float c[4], bm = 0;
      int fault = 0;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        auto bi = bs[i], ci = cs[i];
        if (!normal_source(bi) || !normal_source(ci)) fault = 1;
        c[i] = from(ci);
        bm = fmaxf(bm, fabsf(from(bi)));
      }
      if (!normal_source(xi)) fault = 1;
      float xm = warpmax(fabsf(from(xi)));
      bm = warpmax(bm);
      float cn =
          __fadd_ru(__fadd_ru(fabsf(c[0]), fabsf(c[1])), __fadd_ru(fabsf(c[2]), fabsf(c[3])));
      for (int a = 1; a < 32; a *= 2) cn = __fadd_ru(cn, __shfl_down_sync(~0u, cn, a));
      unsigned src = __ballot_sync(~0u, fault != 0);
      if (!l) {
        q.cn[j] = cn;
        q.bm[j] = bm;
        q.xm[j] = xm;
        if (src) atomicOr(&q.fault, 1);
      }
    }
    __syncthreads();  // coefficients, payload and statistics ready

    for (int j = 0; j < count; j++) {
      float xv = from(q.xx[j][col]), dx = mul(q.sf[j], xv);
      float pr[4], m = 0.f;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        int n = 4 * l + i;
        float old = st[i], inj = mul(dx, from(q.bb[j][n]));
        float v = old == 0.f ? inj : add(mul(old, q.df[j]), inj);
        st[i] = v;
        if (!isfinite(v)) bad |= 4;
        m = fmaxf(m, fabsf(v));
        pr[i] = mul(v, from(q.cc[j][n]));
      }
      float sum = add(add(pr[0], pr[1]), add(pr[2], pr[3]));
#pragma unroll
      for (int d = 1; d < 32; d *= 2) sum = add(sum, __shfl_down_sync(~0u, sum, d));
      float N = __uint_as_float(__reduce_max_sync(~0u, __float_as_uint(m) & 0x7fffffffu));
      if (!l) {
        float y0 = add(sum, mul(Dv, xv));
        bf16 packed = __float2bfloat16_rn(y0);
        float v = from(bits(packed));
        if (!(isfinite(v) && isfinite(y0))) bad |= 4;
        size_t p = (size_t(base + j) * nh + h) * 64 + 32 * k + col;
        y[p] = packed;
        window.row[localOwner][0][w][j] = N;
        window.row[localOwner][1][w][j] = fmaxf(0.f, fabsf(y0));
        window.row[localOwner][2][w][j] = fmaxf(0.f, fmaxf(__fsub_ru(v, y0), __fsub_ru(y0, v)));
      }
    }
    __syncthreads();  // all NP column records ready for each owner
    if (w == 0)
      for (int tile = 0; tile < count; tile += 32) {
        int j = tile + l, last = min(32, count - tile) - 1;
        float N = 0, P = 0, qe = 0;
        if (j < count)
          for (int a = 0; a < NP; a++) {
            N = fmaxf(N, window.row[localOwner][0][a][j]);
            P = fmaxf(P, window.row[localOwner][1][a][j]);
            qe = fmaxf(qe, window.row[localOwner][2][a][j]);
          }
        float prev = __shfl_up_sync(~0u, N, 1);
        if (!l) prev = M;
        float a = 1, b = 0;
        Aux1734 aux;
        if (j < count) {
          float S = q.S[j], J = q.J[j], sf = q.sf[j], df = q.df[j], ds = q.ds[j], dd = q.dd[j];
          float X = q.xm[j], B = q.bm[j], M = prev;
          constexpr float uf = 0x1p-24f, ur = 0x1p-53f, G = 1.f + 11.f * 0x1p-23f;
          if (!(sf >= 0 && isfinite(sf) && df >= 0 && isfinite(df) && J <= 1)) bad |= 4;
          Out1530 o;
          float xf = o.V(sf, X), xr = o.V(S, X);
          float fx = o.rad(xf, uf), rx = o.rad(xr, ur);
          float dx = o.A(o.V(ds, X), o.A(fx, rx));
          float fi = o.V(o.A(xf, fx), B), ri = o.V(o.A(xr, rx), B);
          float ifr = o.rad(fi, uf), irr = o.rad(ri, ur);
          float ei = o.A(o.V(dx, B), o.A(ifr, irr));

          float of = o.V(M, df), off = o.rad(of, uf);
          float p = o.rad(o.A(o.A(of, off), o.A(fi, ifr)), uf);
          float k = o.A(ri, irr), mdd = o.V(M, dd);
          bad |= o.bad;
          a = aux.V(G, J);
          b = aux.A(ei, mdd);
          b = aux.A(b, off);
          b = aux.A(b, p);
          b = aux.A(b, aux.V(ur, k));
          b = aux.A(b, aux.V(aux.V(J, M), 0x1.000002p-52f));
          b = aux.A(b, 3.f * 0x1p-149f);
          b = aux.V(G, aux.A(b, 1024.f * 0x1p-149f));
        }
#pragma unroll
        for (int d = 1; d < 32; d *= 2) {
          float al = __shfl_up_sync(~0u, a, d), bl = __shfl_up_sync(~0u, b, d);
          if (l >= d && j < count) {
            b = aux.A(aux.V(a, bl), b);
            a = aux.V(a, al);
          }
        }
        float al = __shfl_up_sync(~0u, a, 1), bl = __shfl_up_sync(~0u, b, 1);
        float U = E;
        if (j < count) {
          if (l) U = aux.A(aux.V(al, E), bl);
          float R = radius(q.S[j], q.J[j], q.ds[j], q.dd[j], q.sf[j], q.df[j], q.xm[j], q.bm[j],
                           q.cn[j], Dv, prev, U, N, bad);
          Out1530 o;
          o.ck(P);
          o.ck(qe);
          float B0 = o.A(R, qe);
          float L0 = o.ck(fmaxf(0.f, __fsub_rd(P, R)));
          float pmax = o.A(P, R);
          float B1 = o.A(B0, o.A(o.V(0x1p-8f, pmax), 0x1p-133f));
          float L1 = o.ck(fmaxf(0.f, __fsub_rd(__fmul_rd(1.f - 0x1p-8f, L0), 0x1p-133f)));
          top[0] = fmaxf(top[0], B0);
          top[1] = fmaxf(top[1], B1);
          top[2] = fmaxf(top[2], L0);
          top[3] = fmaxf(top[3], L1);
          bad |= o.bad;
        }
        bad |= aux.bad;
        M = __shfl_sync(~0u, N, last);
        E = __shfl_sync(~0u, U, last);
      }

    __syncthreads();  // all owners retired before any window reuse
    base += count;
  }
  int wb = __reduce_or_sync(~0u, bad);
  if (!l && wb) atomicOr(&window.owner_bad[localOwner], wb);
  if (!w)
    for (int i = 0; i < 4; i++) {
      float v = warpmax(top[i]);
      if (!l) records[owner].top[i] = v;
    }
  __syncthreads();  // retire every physical warp's owner-local OR
  if (!w && !l) records[owner].fault = q.fault | window.owner_bad[localOwner];
}

template <int W>
WI int vote2051(const Record2051* r) {
  int f = 0;
  float v[4] = {};
  for (int w = 0; w < W; w++) {
    f |= r[w].fault;
    for (int i = 0; i < 4; i++) v[i] = fmaxf(v[i], r[w].top[i]);
  }
  if (!f && !(v[0] <= __fdiv_rd(v[2], 50.f) && v[1] <= __fdiv_rd(v[3], 50.f))) f = 8;
  return f;
}

__global__ void worker2051(ARGS, int T, int nh, int* routes, const int* tile_status = nullptr) {
  if (tile_status && *tile_status == 0) return;
  int id = blockIdx.y, width = routes[id];
  if (!width) return;
  auto r = rec2051(routes, nh, id);
  if (width == 8)
    grouped2051<8, 2>(PTRS, T, nh, r);
  else
    grouped2051<16, 4>(PTRS, T, nh, r);
}
template <int BW>
__global__ void finalize2051(ARGS, int T, int nh, int* routes, const int* tile_status = nullptr) {
  if (tile_status && *tile_status == 0) return;
  extern __shared__ __align__(16) char scratch2051[];
  const int tid = threadIdx.x;
  const int h = BW == 8 ? int(blockIdx.x) / 2 : int(blockIdx.x);
  if constexpr (BW == 16) {
    // A pair of 8s exists iff BOTH primary refusals were metric.
    if (routes[2 * h] == 8 && routes[2 * h + 1] == 8) {
      const int k = tid / 256;
      int f = 0;
      if (!(tid % 256)) f = vote2051<8>(rec2051(routes, nh, 2 * h + k));
      if (!__syncthreads_or(f != 0)) return;
      f = tentative<8, false, true>(PTRS, T, nh, h, k, scratch2051, nullptr, nullptr);
      int* m = reinterpret_cast<int*>(scratch2051);
      if (!(tid % 256)) m[k] = f != 0;
      __syncthreads();
      int mask = m[0] | (m[1] << 1);
      __syncthreads();
      for (int j = 0; j < 2; j++)
        if (mask & (1 << j)) {
          pipeline<16>(PTRS, T, nh, h, 32 * j, scratch2051);
          __syncthreads();
        }
      return;
    }
  }
  const int first = BW == 8 ? (int(blockIdx.x) & 1) : 0;
  for (int j = first; j < (BW == 8 ? first + 1 : 2); j++) {
    if (!routes[2 * h + j]) continue;
    int f = 0;
    if (!tid) f = vote2051<BW>(rec2051(routes, nh, 2 * h + j));
    if (__syncthreads_or(f != 0)) {
      if (tentative<BW, false>(PTRS, T, nh, h, j, scratch2051, nullptr, nullptr))
        pipeline<BW>(PTRS, T, nh, h, 32 * j, scratch2051);
      __syncthreads();
    }
  }
}
template <int BW>
constexpr int finalBytes2051() {
  static_assert(BW == 8 || BW == 16);
  static_assert(sizeof(Scratch<8>) >= sizeof(Cert) && DualSH >= sizeof(Scratch<16>));
  return BW == 8 ? int(sizeof(Scratch<8>)) : DualSH;
}
template <int BW>
static void prepare2051() {
  // 45600 bytes fits default dynamic shared; no worker opt-in or dummy query.
  TVM_FFI_CHECK_CUDA_ERROR(cudaFuncSetAttribute(
      finalize2051<BW>, cudaFuncAttributeMaxDynamicSharedMemorySize, finalBytes2051<BW>()));
}
template <int BW>
static void finish2051(ARGS, int T, int nh, int* routes, cudaStream_t stream,
                       const int* tile_status = nullptr) {
  TVM_FFI_CHECK_CUDA_ERROR(cudaGetLastError());
  worker2051<<<dim3(4, 2 * nh), 256, 45600, stream>>>(PTRS, T, nh, routes, tile_status);
  TVM_FFI_CHECK_CUDA_ERROR(cudaGetLastError());
  finalize2051<BW><<<(BW == 8 ? 2 * nh : nh), 32 * BW, finalBytes2051<BW>(), stream>>>(
      PTRS, T, nh, routes, tile_status);
  TVM_FFI_CHECK_CUDA_ERROR(cudaGetLastError());
}

#undef I
#undef ARGS
#undef PTRS
static_assert(SH <= 101376 && SH <= 100376);
#undef WI
}  // namespace win1306

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

  extern __shared__ __align__(16) char smem[];
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

  using namespace win1306;
  Slot& slot = *reinterpret_cast<Slot*>(sCum);
  int bad = 0;
  Budget budget;
  double previous = 0.;
  float scorePeak = 0.f;
  const float D_h = from(bits(Dg[h]));
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
  issue_C(0);
  issue_X(0);
  issue_B(0);
  // Warm the two chunks that steady-state c+2 prefetching cannot reach, while
  // the initial async B/C/X copies are already in flight.
#pragma unroll
  for (int d = 1; d <= 2; ++d)
    if (d < nchunks) l2_prefetch(d);

  const int loop_chunks = ONE_CHUNK ? 1 : nchunks;
#pragma unroll 2
  for (int c = 0; c < loop_chunks; ++c) {
    const int t0 = c * 128;

    const bool has_next = c + 1 < nchunks;
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
    auto put_state = [&](uint32_t sbbase, int low) {
#pragma unroll
      for (int nt = 0; nt < NT_S; nt += 2) {
        int n0 = WN * N_PART + nt * 8, p0 = 16 * WP;
        uint32_t o = addrU<16>(sbbase, p0 + (lane & 15), (n0 >> 3) + (lane >> 4));
        uint32_t v0 = word(Sacc[nt][0], Sacc[nt][1], low, bad, 1);
        uint32_t v1 = word(Sacc[nt][2], Sacc[nt][3], low, bad, 1);
        uint32_t v2 = word(Sacc[nt + 1][0], Sacc[nt + 1][1], low, bad, 1);
        uint32_t v3 = word(Sacc[nt + 1][2], Sacc[nt + 1][3], low, bad, 1);
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
    };

    auto state_up = [&](int kb, float decay) {
      float delta[NT_S][4] = {};
      for (int low = 0; low < 2; low++) {
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
          int k = kb + kk, j = 16 * kk + 2 * t4;
          uint32_t xa[4], af[4];
          peer_ldm4t(xa, addrU<XU>(aX, 16 * k + (lane & 15), 2 * WP + (lane >> 4)));
          auto wx = [&](unsigned v, int t) {
            return word(checked(mul(from(v), slot.write[t]), bad, 1),
                        checked(mul(from(v >> 16), slot.write[t + 1]), bad, 1), low, bad, 1);
          };
          af[0] = wx(xa[0], j);
          af[1] = wx(xa[2], j);
          af[2] = wx(xa[1], j + 8);
          af[3] = wx(xa[3], j + 8);
          for (int nn = 0; nn < NT_S / 2; nn++) {
            uint32_t bb[4];
            peer_ldm4t(
                bb, addrU<16>(aB, 16 * k + (lane & 15), (WN * N_PART + nn * 16) / 8 + (lane >> 4)));
            mmaf(delta[2 * nn], af, bb[0], bb[1]);
            mmaf(delta[2 * nn + 1], af, bb[2], bb[3]);
          }
        }
        for (int n = 0; n < NT_S; n++)
          for (int q = 0; q < 4; q++) checked(delta[n][q], bad, 1);
      }
      for (int n = 0; n < NT_S; n++)
        for (int q = 0; q < 4; q++)
          Sacc[n][q] = checked(add(checked(mul(Sacc[n][q], decay), bad, 1), delta[n][q]), bad, 1);
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
        if (k < 7)
#pragma unroll
          for (int q = 0; q < 4; ++q) bb[q] = bn[q];
      }
    };

    __syncthreads();  // current B/X arrived before any cross-warp use
    for (int half = 0; half < 2; half++) {
      int start = t0 + 64 * half, n = min(64, T - start);
      if (n <= 0) break;
      setup<true, HD_BLK == 32>(slot, nullptr, xg, dtg, Bg, Cg, dtbg, Dg, Ag, start, n, NH, h,
                                pslot * HD_BLK, bad, aB, aX, 64 * half);
      if (warp < (n + 31) / 32) scan1423<true>(budget, slot, nullptr, start, n, D_h, bad);
      if (!tid) {
        if (half && !(fabs(previous + slot.total) <= 65536.)) bad |= 1;
        previous = slot.total;
      }
      if (start) {
        put_state(aSb0, 0);
        put_state(aSb1, 1);
      }
      __syncthreads();
      if (HALF == half) {
        if (start) {
          y_off(aSb0);
          y_off(aSb1);
          for (int nt = 0; nt < NT_Y; nt++)
            for (int q = 0; q < 4; q++) {
              int r = 16 * (I & 3) + gid + 8 * (q >> 1);
              Yacc[nt][q] = checked(mul(checked(Yacc[nt][q], bad, 1), slot.row[r]), bad, 1);
            }
        }
        uint32_t residual[4][4];
        float rows[2] = {};
        for (int low = 0; low < 2; low++) {
          for (int jj = 0; jj <= (I & 3); jj++) {
            int J = 4 * half + jj;
            uint32_t mf[4];
            if (!low) {
              float ga[2][4] = {};
              gram_one(J, ga);
              for (int nb = 0; nb < 2; nb++)
                for (int rb = 0; rb < 2; rb++) {
                  int r = 16 * (I & 3) + gid + 8 * rb, j = 16 * jj + 8 * nb + 2 * t4;
                  float v[2] = {};
                  for (int t = 0; t < 2; t++)
                    if (r < n && j + t <= r)
                      v[t] = checked(
                          mul(checked(ga[nb][2 * rb + t], bad, 1), weight(slot, r, j + t, bad, 1)),
                          bad, 1);
                  unsigned hi = word(v[0], v[1], 0, bad, 1);
                  mf[2 * nb + rb] = hi;
                  residual[jj][2 * nb + rb] = word(v[0], v[1], 1, bad, 1);
                  rows[rb] = __fadd_ru(rows[rb], __fadd_ru(fabsf(from(hi)), fabsf(from(hi >> 16))));
                }
            } else
              for (int q = 0; q < 4; q++) mf[q] = residual[jj][q];
            for (int pp = 0; pp < NT_Y / 2; pp++) {
              uint32_t xb[4];
              peer_ldm4t(xb, addrU<XU>(aX, 16 * J + (lane & 15), 2 * pp + (lane >> 4)));
              mmaf(Yacc[2 * pp], mf, xb[0], xb[1]);
              mmaf(Yacc[2 * pp + 1], mf, xb[2], xb[3]);
            }
          }
          for (int nt = 0; nt < NT_Y; nt++)
            for (int q = 0; q < 4; q++) checked(Yacc[nt][q], bad, 1);
        }
        for (int rb = 0; rb < 2; rb++) {
          float v = rows[rb];
          v = __fadd_ru(v, __shfl_xor_sync(~0u, v, 1));
          v = __fadd_ru(v, __shfl_xor_sync(~0u, v, 2));
          scorePeak = fmaxf(scorePeak, v);
        }
      }
      // y_off reads published state snapshots, not another warp's Sacc.
      // Warps without Y rows in this half may advance their register state.
      // The following barrier retires snapshot readers and state publishers.
      if (T - start > 64) {
        state_up(half * 4, slot.row[n - 1]);
        publish1438(slot, 0, Sacc);
      }
      __syncthreads();  // retire coeffs and register delta before new snapshot
      if (T - start > 64 && !tid) tighten1438(budget, slot, 0);
    }
    // ---------------- epilogue ----------------
    __syncthreads();
    Peaks rp;
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
        out[2 * q] = 0;
        if (t0 + row0 < T) {
          unsigned lo = rp.finish(Yacc[nt + q][0], f0.x, D_h, bad, 1);
          unsigned hi = rp.finish(Yacc[nt + q][1], f0.y, D_h, bad, 1);
          out[2 * q] = lo | (hi << 16);
        }
        out[2 * q + 1] = 0;
        if (t0 + row1 < T) {
          unsigned lo = rp.finish(Yacc[nt + q][2], f1.x, D_h, bad, 1);
          unsigned hi = rp.finish(Yacc[nt + q][3], f1.y, D_h, bad, 1);
          out[2 * q + 1] = lo | (hi << 16);
        }
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

    // The temporaries have two physical slots, even in RunPeer. Retire Y
    // readers before reusing sSb1. Keep maxima local to each 64-row window.
    if (tid < 8) slot.rounding[tid / 4][tid % 4] = 0.;
    __syncthreads();
    rp.reduce(slot, HALF);
    __syncthreads();
    if (!tid) {
      collect(budget, slot, 0);
      collect(budget, slot, 1);
    }
    __syncthreads();
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

  // Reduce across quadrants, not just the last warp's rows.
  auto* sums = reinterpret_cast<double*>(smem);
  for (int d = 16; d; d /= 2) scorePeak = fmaxf(scorePeak, __shfl_down_sync(~0u, scorePeak, d));
  if (!lane) sums[warp] = wide(scorePeak);
  __syncthreads();
  const int priorNonmetric = __syncthreads_or(bad);
  if (!tid) {
    for (int w = 0; w < 8; w++) budget.score = fmax(budget.score, sums[w]);
    bad |= refuses(budget, budget.score);
  }
  if (__syncthreads_or(bad)) {
    bool exact = true;
    if constexpr (HD_BLK == 32) {
      // tentative takes a half index; pipeline takes its first column.
      if (!priorNonmetric)
        exact = win1306::tentative_recover<8>(xg, dtg, Ag, Bg, Cg, Dg, dtbg, yg, T, NH, h, pslot,
                                              smem) != 0;
    }
    if (exact)
      win1306::pipeline<8>(xg, dtg, Ag, Bg, Cg, Dg, dtbg, yg, T, NH, h, pslot * HD_BLK, smem);
  }
}

template <int HD_BLK, bool FULL_T = false, bool ONE_CHUNK = false, bool DIRECT_DECAY = false>
__global__ __launch_bounds__(256, 1) void ssd_primary2051(
    const bf16* __restrict__ xg, const bf16* __restrict__ dtg, const float* __restrict__ Ag,
    const bf16* __restrict__ Bg, const bf16* __restrict__ Cg, const bf16* __restrict__ Dg,
    const bf16* __restrict__ dtbg, bf16* __restrict__ yg, int T, int NH, int nchunks, int* routes) {
  constexpr int P_BLOCKS = HD_BLK / 16;
  constexpr int N_PART = 16 * P_BLOCKS;
  constexpr int NT_Y = HD_BLK / 8;
  constexpr int NT_S = N_PART / 8;
  constexpr int XU = HD_BLK / 8;
  constexpr int HDSPLIT = 64 / HD_BLK;
  constexpr int HEADS = (HD_BLK == 32) ? 64 : 128;
  constexpr bool L2E = (HD_BLK <= 32);  // decay math in log2 units (fewer MUFU inputs)

  extern __shared__ __align__(16) char smem[];
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
  if (!tid) routes[2 * h + pslot] = 0;
  const int gq = h / (HEADS >> 3);

  using namespace win1306;
  Slot& slot = *reinterpret_cast<Slot*>(sCum);
  int bad = 0;
  Budget budget;
  double previous = 0.;
  float scorePeak = 0.f;
  const float D_h = from(bits(Dg[h]));
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
  issue_C(0);
  issue_X(0);
  issue_B(0);
  // Warm the two chunks that steady-state c+2 prefetching cannot reach, while
  // the initial async B/C/X copies are already in flight.
#pragma unroll
  for (int d = 1; d <= 2; ++d)
    if (d < nchunks) l2_prefetch(d);

  const int loop_chunks = ONE_CHUNK ? 1 : nchunks;
#pragma unroll 2
  for (int c = 0; c < loop_chunks; ++c) {
    const int t0 = c * 128;

    const bool has_next = c + 1 < nchunks;
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
    auto put_state = [&](uint32_t sbbase, int low) {
#pragma unroll
      for (int nt = 0; nt < NT_S; nt += 2) {
        int n0 = WN * N_PART + nt * 8, p0 = 16 * WP;
        uint32_t o = addrU<16>(sbbase, p0 + (lane & 15), (n0 >> 3) + (lane >> 4));
        uint32_t v0 = word(Sacc[nt][0], Sacc[nt][1], low, bad, 1);
        uint32_t v1 = word(Sacc[nt][2], Sacc[nt][3], low, bad, 1);
        uint32_t v2 = word(Sacc[nt + 1][0], Sacc[nt + 1][1], low, bad, 1);
        uint32_t v3 = word(Sacc[nt + 1][2], Sacc[nt + 1][3], low, bad, 1);
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
    };

    auto state_up = [&](int kb, float decay) {
      float delta[NT_S][4] = {};
      for (int low = 0; low < 2; low++) {
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
          int k = kb + kk, j = 16 * kk + 2 * t4;
          uint32_t xa[4], af[4];
          peer_ldm4t(xa, addrU<XU>(aX, 16 * k + (lane & 15), 2 * WP + (lane >> 4)));
          auto wx = [&](unsigned v, int t) {
            return word(checked(mul(from(v), slot.write[t]), bad, 1),
                        checked(mul(from(v >> 16), slot.write[t + 1]), bad, 1), low, bad, 1);
          };
          af[0] = wx(xa[0], j);
          af[1] = wx(xa[2], j);
          af[2] = wx(xa[1], j + 8);
          af[3] = wx(xa[3], j + 8);
          for (int nn = 0; nn < NT_S / 2; nn++) {
            uint32_t bb[4];
            peer_ldm4t(
                bb, addrU<16>(aB, 16 * k + (lane & 15), (WN * N_PART + nn * 16) / 8 + (lane >> 4)));
            mmaf(delta[2 * nn], af, bb[0], bb[1]);
            mmaf(delta[2 * nn + 1], af, bb[2], bb[3]);
          }
        }
        for (int n = 0; n < NT_S; n++)
          for (int q = 0; q < 4; q++) checked(delta[n][q], bad, 1);
      }
      for (int n = 0; n < NT_S; n++)
        for (int q = 0; q < 4; q++)
          Sacc[n][q] = checked(add(checked(mul(Sacc[n][q], decay), bad, 1), delta[n][q]), bad, 1);
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
        if (k < 7)
#pragma unroll
          for (int q = 0; q < 4; ++q) bb[q] = bn[q];
      }
    };

    __syncthreads();  // current B/X arrived before any cross-warp use
    for (int half = 0; half < 2; half++) {
      int start = t0 + 64 * half, n = min(64, T - start);
      if (n <= 0) break;
      setup<true, HD_BLK == 32>(slot, nullptr, xg, dtg, Bg, Cg, dtbg, Dg, Ag, start, n, NH, h,
                                pslot * HD_BLK, bad, aB, aX, 64 * half);
      if (warp < (n + 31) / 32) scan1423<true>(budget, slot, nullptr, start, n, D_h, bad);
      if (!tid) {
        if (half && !(fabs(previous + slot.total) <= 65536.)) bad |= 1;
        previous = slot.total;
      }
      if (start) {
        put_state(aSb0, 0);
        put_state(aSb1, 1);
      }
      __syncthreads();
      if (HALF == half) {
        if (start) {
          y_off(aSb0);
          y_off(aSb1);
          for (int nt = 0; nt < NT_Y; nt++)
            for (int q = 0; q < 4; q++) {
              int r = 16 * (I & 3) + gid + 8 * (q >> 1);
              Yacc[nt][q] = checked(mul(checked(Yacc[nt][q], bad, 1), slot.row[r]), bad, 1);
            }
        }
        uint32_t residual[4][4];
        float rows[2] = {};
        for (int low = 0; low < 2; low++) {
          for (int jj = 0; jj <= (I & 3); jj++) {
            int J = 4 * half + jj;
            uint32_t mf[4];
            if (!low) {
              float ga[2][4] = {};
              gram_one(J, ga);
              for (int nb = 0; nb < 2; nb++)
                for (int rb = 0; rb < 2; rb++) {
                  int r = 16 * (I & 3) + gid + 8 * rb, j = 16 * jj + 8 * nb + 2 * t4;
                  float v[2] = {};
                  for (int t = 0; t < 2; t++)
                    if (r < n && j + t <= r)
                      v[t] = checked(
                          mul(checked(ga[nb][2 * rb + t], bad, 1), weight(slot, r, j + t, bad, 1)),
                          bad, 1);
                  unsigned hi = word(v[0], v[1], 0, bad, 1);
                  mf[2 * nb + rb] = hi;
                  residual[jj][2 * nb + rb] = word(v[0], v[1], 1, bad, 1);
                  rows[rb] = __fadd_ru(rows[rb], __fadd_ru(fabsf(from(hi)), fabsf(from(hi >> 16))));
                }
            } else
              for (int q = 0; q < 4; q++) mf[q] = residual[jj][q];
            for (int pp = 0; pp < NT_Y / 2; pp++) {
              uint32_t xb[4];
              peer_ldm4t(xb, addrU<XU>(aX, 16 * J + (lane & 15), 2 * pp + (lane >> 4)));
              mmaf(Yacc[2 * pp], mf, xb[0], xb[1]);
              mmaf(Yacc[2 * pp + 1], mf, xb[2], xb[3]);
            }
          }
          for (int nt = 0; nt < NT_Y; nt++)
            for (int q = 0; q < 4; q++) checked(Yacc[nt][q], bad, 1);
        }
        for (int rb = 0; rb < 2; rb++) {
          float v = rows[rb];
          v = __fadd_ru(v, __shfl_xor_sync(~0u, v, 1));
          v = __fadd_ru(v, __shfl_xor_sync(~0u, v, 2));
          scorePeak = fmaxf(scorePeak, v);
        }
      }
      // y_off reads published state snapshots, not another warp's Sacc.
      // Warps without Y rows in this half may advance their register state.
      // The following barrier retires snapshot readers and state publishers.
      if (T - start > 64) {
        state_up(half * 4, slot.row[n - 1]);
        publish1438(slot, 0, Sacc);
      }
      __syncthreads();  // retire coeffs and register delta before new snapshot
      if (T - start > 64 && !tid) tighten1438(budget, slot, 0);
    }
    // ---------------- epilogue ----------------
    __syncthreads();
    Peaks rp;
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
        out[2 * q] = 0;
        if (t0 + row0 < T) {
          unsigned lo = rp.finish(Yacc[nt + q][0], f0.x, D_h, bad, 1);
          unsigned hi = rp.finish(Yacc[nt + q][1], f0.y, D_h, bad, 1);
          out[2 * q] = lo | (hi << 16);
        }
        out[2 * q + 1] = 0;
        if (t0 + row1 < T) {
          unsigned lo = rp.finish(Yacc[nt + q][2], f1.x, D_h, bad, 1);
          unsigned hi = rp.finish(Yacc[nt + q][3], f1.y, D_h, bad, 1);
          out[2 * q + 1] = lo | (hi << 16);
        }
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

    // The temporaries have two physical slots, even in RunPeer. Retire Y
    // readers before reusing sSb1. Keep maxima local to each 64-row window.
    if (tid < 8) slot.rounding[tid / 4][tid % 4] = 0.;
    __syncthreads();
    rp.reduce(slot, HALF);
    __syncthreads();
    if (!tid) {
      collect(budget, slot, 0);
      collect(budget, slot, 1);
    }
    __syncthreads();
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

  // Reduce across quadrants, not just the last warp's rows.
  auto* sums = reinterpret_cast<double*>(smem);
  for (int d = 16; d; d /= 2) scorePeak = fmaxf(scorePeak, __shfl_down_sync(~0u, scorePeak, d));
  if (!lane) sums[warp] = wide(scorePeak);
  __syncthreads();
  const int priorNonmetric = __syncthreads_or(bad);
  if (!tid) {
    for (int w = 0; w < 8; w++) budget.score = fmax(budget.score, sums[w]);
    bad |= refuses(budget, budget.score);
  }
  if (__syncthreads_or(bad)) {
    static_assert(HD_BLK == 32);
    if (!priorNonmetric) {
      if (!tid) routes[2 * h + pslot] = 8;
    } else
      win1306::pipeline<8>(xg, dtg, Ag, Bg, Cg, Dg, dtbg, yg, T, NH, h, pslot * HD_BLK, smem);
  }
}

// Atomic: the Python wrapper does not serialize calls, so two host threads
// can reach the first-call attribute setup concurrently.  The setup itself is
// idempotent; the atomic only removes the unsynchronized int data race.
static std::atomic<int> g_init[2] = {0, 0};

static void* effective1306(const ffi::TensorView& t) {
  uintptr_t base = reinterpret_cast<uintptr_t>(t.data_ptr());
  uint64_t offset = t.byte_offset();
  TVM_FFI_ICHECK(base && offset <= UINTPTR_MAX - base) << "bad DLPack base/offset";
  return reinterpret_cast<void*>(base + offset);
}
static void stream1306(int dev, cudaStream_t stream) {
  CUcontext cur = nullptr, owner = nullptr;
  CUdevice current = -1;
  TVM_FFI_ICHECK(cuCtxGetCurrent(&cur) == CUDA_SUCCESS && cur &&
                 cuCtxGetDevice(&current) == CUDA_SUCCESS && current == dev)
      << "wrong CUDA context device";
  if (stream && stream != cudaStreamLegacy && stream != cudaStreamPerThread) {
    TVM_FFI_ICHECK(cuStreamGetCtx(reinterpret_cast<CUstream>(stream), &owner) == CUDA_SUCCESS &&
                   owner == cur)
        << "stream belongs to another CUDA context";
  }
}
void RunPeer(ffi::TensorView x, ffi::TensorView dt, ffi::TensorView A, ffi::TensorView Bmat,
             ffi::TensorView Cmat, ffi::TensorView D, ffi::TensorView dt_bias, ffi::TensorView y) {
  const int T = (int)x.shape()[0];
  const int NH = (int)x.shape()[1];
  const int nchunks = (T + 127) / 128;
  auto dev = x.device();
  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(dev.device_type, dev.device_id);

  stream1306(dev.device_id, stream);
  const bf16* xp = (const bf16*)effective1306(x);
  const bf16* dtp = (const bf16*)effective1306(dt);
  const float* Ap = (const float*)effective1306(A);
  const bf16* Bp = (const bf16*)effective1306(Bmat);
  const bf16* Cp = (const bf16*)effective1306(Cmat);
  const bf16* Dp = (const bf16*)effective1306(D);
  const bf16* dbp = (const bf16*)effective1306(dt_bias);
  bf16* yp = (bf16*)effective1306(y);
  auto& initialized = g_init[NH >= 96 ? 0 : 1];
  const bool needs_setup = !initialized.load();

  TVM_FFI_ICHECK_EQ(NH, 64);
  {
    constexpr int SH = 65536 + 2 * 128 * 32 * 2 + 2 * 32 * 128 * 2 + 3 * 2 * 128 * 4;
    if (needs_setup) {
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
    }
    TVM_FFI_CHECK_CUDA_ERROR(cudaGetLastError());  // report pending error before <<<
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
  // While Run is still unbound, a failed submission can be retried on another device.
  if (needs_setup) initialized.store(1);
}

static void RunPeer2051(ffi::TensorView x, ffi::TensorView dt, ffi::TensorView A,
                        ffi::TensorView Bmat, ffi::TensorView Cmat, ffi::TensorView D,
                        ffi::TensorView dt_bias, ffi::TensorView y, int* routes,
                        cudaStream_t& accepted_stream) {
  const int T = (int)x.shape()[0];
  const int NH = (int)x.shape()[1];
  const int nchunks = (T + 127) / 128;
  auto dev = x.device();
  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(dev.device_type, dev.device_id);

  stream1306(dev.device_id, stream);
  accepted_stream = stream;
  const bf16* xp = (const bf16*)effective1306(x);
  const bf16* dtp = (const bf16*)effective1306(dt);
  const float* Ap = (const float*)effective1306(A);
  const bf16* Bp = (const bf16*)effective1306(Bmat);
  const bf16* Cp = (const bf16*)effective1306(Cmat);
  const bf16* Dp = (const bf16*)effective1306(D);
  const bf16* dbp = (const bf16*)effective1306(dt_bias);
  bf16* yp = (bf16*)effective1306(y);
  static std::atomic<int> initialized{0};
  const bool needs_setup = !initialized.load();

  TVM_FFI_ICHECK_EQ(NH, 64);
  {
    constexpr int SH = 65536 + 2 * 128 * 32 * 2 + 2 * 32 * 128 * 2 + 3 * 2 * 128 * 4;
    if (needs_setup) {
      win1306::prepare2051<8>();
      TVM_FFI_CHECK_CUDA_ERROR(cudaFuncSetAttribute(
          ssd_primary2051<32>, cudaFuncAttributeMaxDynamicSharedMemorySize, SH));
      TVM_FFI_CHECK_CUDA_ERROR(cudaFuncSetAttribute(
          ssd_primary2051<32, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SH));
      TVM_FFI_CHECK_CUDA_ERROR(cudaFuncSetAttribute(
          ssd_primary2051<32, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SH));
      TVM_FFI_CHECK_CUDA_ERROR(cudaFuncSetAttribute(
          ssd_primary2051<32, true, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SH));
      TVM_FFI_CHECK_CUDA_ERROR(cudaFuncSetAttribute(ssd_primary2051<32, false, false, true>,
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                    SH));
    }
    TVM_FFI_CHECK_CUDA_ERROR(cudaGetLastError());  // report pending error before <<<
    if (T <= 128)
      ssd_primary2051<32, false, true>
          <<<NH * 2, 256, SH, stream>>>(xp, dtp, Ap, Bp, Cp, Dp, dbp, yp, T, NH, nchunks, routes);
    else if (T >= 12288) {
      if ((T & 127) == 0)
        ssd_primary2051<32, true, false, true>
            <<<NH * 2, 256, SH, stream>>>(xp, dtp, Ap, Bp, Cp, Dp, dbp, yp, T, NH, nchunks, routes);
      else
        ssd_primary2051<32, false, false, true>
            <<<NH * 2, 256, SH, stream>>>(xp, dtp, Ap, Bp, Cp, Dp, dbp, yp, T, NH, nchunks, routes);
    } else if ((T & 127) == 0)
      ssd_primary2051<32, true>
          <<<NH * 2, 256, SH, stream>>>(xp, dtp, Ap, Bp, Cp, Dp, dbp, yp, T, NH, nchunks, routes);
    else
      ssd_primary2051<32>
          <<<NH * 2, 256, SH, stream>>>(xp, dtp, Ap, Bp, Cp, Dp, dbp, yp, T, NH, nchunks, routes);
  }
  TVM_FFI_CHECK_CUDA_ERROR(cudaGetLastError());
  // While Run is still unbound, a failed submission can be retried on another device.
  if (needs_setup) initialized.store(1);
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
// One lock for the payload and all flag/progress/done words, across
// every NH128 specialization in this module. Never clear a stranded lock.
static __device__ __align__(8) unsigned long long ring_owner = 0ull;
static constexpr unsigned long long ring_poison = ~0ull;
__device__ __align__(16) float g_gram[GWIN * NGR * GTILES * GTILE_BF];
// Published and retired with the same owner's Gram window.
__device__ __align__(16) uint2 g_bc1952[NGR * GWIN * QT];
static_assert(sizeof(g_bc1952) == 1048576);
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
    const int nr = min(QT, T - c * QT);
    for (int i = warp; i < nr; i += NWP) {
      float bm = 0.f, cn = 0.f;
      int invalid = 0;
      auto ru = [](float a, float b) {
        float z;
        asm("add.rp.f32 %0,%1,%2;" : "=f"(z) : "f"(a), "f"(b));
        return z;
      };
      for (int j = lane; j < DSN; j += 32) {
        auto b = win1306::bits(bCur[i * LDN + j]), v = win1306::bits(cCur[i * LDN + j]);
        invalid |= !win1306::normal_source(b) || !win1306::normal_source(v);
        bm = fmaxf(bm, fabsf(win1306::from(b)));
        cn = ru(cn, fabsf(win1306::from(v)));
      }
      for (int d = 16; d; d /= 2) {
        bm = fmaxf(bm, __shfl_down_sync(~0u, bm, d));
        cn = ru(cn, __shfl_down_sync(~0u, cn, d));
      }
      unsigned f = __ballot_sync(~0u, invalid);
      if (!lane)
        g_bc1952[(g * GWIN + (c & (GWIN - 1))) * QT + i] =
            make_uint2(__float_as_uint(bm) | (unsigned(f != 0) << 31), __float_as_uint(cn));
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
namespace regx {

template <int PT>
DI void wpub(bf16* st, float (&s)[4][4], int pm, int nn, int lane, int low, int& bad) {
  for (int n = 0; n < 4; n += 2) {
    int mask = 1 << (pm / 2);
    unsigned v[4];
    for (int q = 0; q < 4; q++)
      v[q] =
          win1306::word(s[n + q / 2][2 * (q & 1)], s[n + q / 2][2 * (q & 1) + 1], low, bad, mask);
    uint32_t dst = sptr(st + (16 * pm + (lane & 15)) * LDN + 32 * nn + 8 * n + 8 * (lane >> 4));
    peer_stm4(dst, v[0], v[1], v[2], v[3]);
  }
}
DI void wproject(float (&y)[2][4], float (&s)[4][4], bf16* C, bf16* st, int mi, int ng, int nn,
                 int lane, int low, int& bad) {
  uint32_t ac = addr4<LDN>(C, 16 * mi, 0, lane), bt = addr4b<LDN>(st, 16 * ng, 0, lane);
  for (int j = 0; j < 8; j++) {
    uint32_t a[4], b[4], off = 32u * ((2 * nn + j) & 7);
    ldm4(a, ac + off);
    if (j < 2) {
      for (int q = 0; q < 4; q++)
        b[q] = win1306::word(s[2 * j + (q & 1)][2 * (q / 2)], s[2 * j + (q & 1)][2 * (q / 2) + 1],
                             low, bad, 1 << (ng / 2));
    } else
      ldm4(b, bt + off);
    mma_f32(y[0], a, b);
    mma_f32(y[1], a, b + 2);
  }
}
template <int LDX>
DI void wupdate(float (&s)[4][4], bf16* hi, bf16* lo, bf16* b, int pm, int nn, int lane,
                float decay, int& bad) {
  using namespace win1306;
  int mask = 1 << (pm / 2);
  float delta[4][4] = {};
  const uint32_t bn = addr4bt<LDN>(b, 32 * nn, 0, lane);
#pragma unroll
  for (int low = 0; low < 2; low++) {
    const uint32_t ax = addr4t<LDX>(low ? lo : hi, 16 * pm, 0, lane);
#pragma unroll
    for (int k = 0; k < 4; k++) {
      uint32_t a[4];
      ldm4to<0>(a, ax + 32u * k * LDX);
#pragma unroll
      for (int pair = 0; pair < 2; pair++) {
        uint32_t bb[4];
        ldm4to<0>(bb, bn + 32u * k * LDN + 32u * pair);
        mma_f32(delta[2 * pair], a, bb);
        mma_f32(delta[2 * pair + 1], a, bb + 2);
      }
    }
    // Check the high-only partial before the low plane can cancel it.
#pragma unroll
    for (int n = 0; n < 4; n++)
      for (int q = 0; q < 4; q++) checked(delta[n][q], bad, mask);
  }
#pragma unroll
  for (int n = 0; n < 4; n++)
    for (int q = 0; q < 4; q++)
      s[n][q] = checked(add(checked(mul(s[n][q], decay), bad, mask), delta[n][q]), bad, mask);
}

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

  using namespace win1306;
  int repairMask = 0, nonmetricMask = 0;
  static_assert(PT == 64, "regx ownership path is NH128 only");
  unsigned long long id;
  asm volatile("mov.u64 %0, %%gridid;" : "=l"(id));
  const unsigned long long token = id + 1ull;
  const bool eligible = token != 0ull && token != ring_poison;
  if (eligible && blockIdx.x == 0 && threadIdx.x == 0) {
    unsigned long long old;
    asm volatile("atom.acq_rel.gpu.global.cas.b64 %0, [%1], %2, %3;"
                 : "=l"(old)
                 : "l"(&ring_owner), "l"(0ull), "l"(token)
                 : "memory");
    // Only possible for a stranded or diagnostic owner, not a live peer.
    if (old == token) gst_release(&ring_owner, ring_poison);
  }
  cooperative_groups::this_grid().sync();
  if (!eligible || gld_acquire(&ring_owner) != token) {
    if (blockIdx.x < (unsigned)NH) {
      extern __shared__ __align__(128) char smem_raw[];
      for (int k = 0; k < 2; ++k)
        ordered(gx, gdt, gA, gB, gC, gD, gdtb, gy, T, NH, blockIdx.x, 32 * k, smem_raw);
    }
    return;
  }
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

    Slot& slot = *reinterpret_cast<Slot*>(sScan);
    int bad = 0;
    Budget budget;
    float scorePeak = 0.f;
    const float Dh = from(bits(gD[h]));
    if (tid == 0) {
      prefetch_tma(&tmaB);
      prefetch_tma(&tmaC);
      tma_barrier_init(sTmaBar);
      // Every warp arrives, not just the score producers: with the group Gram
      // lifted out, all sixteen warps read the C tile and the state image for
      // exactly as long, so the four non-score warps are no longer guaranteed
      // to be past that phase when the tile is recycled below.
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
    // Highest published window of this group that this thread has observed.
    unsigned int gready = 0;
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

      int count = min(QT, T - c * QT);
      static_assert(NW == 16 && ((0x55154ee0u >> 4) & 15u) == 14u);
      setup<false>(slot, sX, gx, gdt, gB, gC, gdtb, gD, gA, c * QT, count, NH, h, 0, bad, 0, 0, 0,
                   g_bc1952 + (g * GWIN + (c & (GWIN - 1))) * QT);
      if (warp < (count + 15) / 16) scan1423<false>(budget, slot, sX, c * QT, count, Dh, bad);
      if (c + 1 < nchunks) {
        LOAD_X(c + 1)
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

      const bool hasNext = c + 1 < nchunks;
      const bool nextFull = c + 2 <= T / QT;
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

      const int ia0 = 16 * mi + (lane >> 2), khalf = ng / 2, mask = 1 << khalf;
      float accY[2][4] = {};
      if (c) {
        wproject(accY, accSt[0], cCur, sSt, mi, ng, nn, lane, 0, bad);
        __syncthreads();
        wpub<PT>(sSt, accSt[0], pm, nn, lane, 1, bad);
        __syncthreads();
        wproject(accY, accSt[0], cCur, sSt, mi, ng, nn, lane, 1, bad);
        for (int pt = 0; pt < 2; pt++)
          for (int q = 0; q < 4; q++)
            accY[pt][q] = checked(mul(checked(accY[pt][q], bad, mask), slot.row[ia0 + 8 * (q / 2)]),
                                  bad, mask);
      }
      if (hasG) {
        auto emit = [&](int jj, int half, float4 f) {
          int j = 16 * jj + 8 * half + 2 * (lane & 3);
          float v[4] = {f.x, f.y, f.z, f.w};
          for (int rb = 0; rb < 2; rb++) {
            int i = ia0 + 8 * rb;
            for (int q = 0; q < 2; q++) {
              float& a = v[2 * rb + q];
              a = i < count && j + q <= i
                      ? checked(mul(checked(a, bad, 3), weight(slot, i, j + q, bad, 3)), bad, 3)
                      : 0.f;
            }
            auto dst = sSc + score_index(i, j);
            for (int low = 0; low < 2; low++)
              *reinterpret_cast<unsigned*>(dst + 256 * low) =
                  word(v[2 * rb], v[2 * rb + 1], low, bad, 3);
          }
        };
        if (gFull) {
          emit(gj, 0, gfrag.lo);
          emit(gj, 1, gfrag.hi);
        } else
          emit(gj, gk == 3, ghalf);
      }

      if (hasNext) load_gram(c + 1);
      __syncthreads();  // retire low readers, old Y drain; publish scores
      if (hasNext) {
        for (int pair = tid; pair < QT * PT / 2; pair += NTHR) {
          int r = pair / (PT / 2), p = 2 * (pair % (PT / 2)), off = r * LDX + p, m = 1 << (p / 32);
          unsigned x = *reinterpret_cast<unsigned*>(sX + off);
          float a = checked(mul(from(x), slot.write[r]), bad, m);
          float b = checked(mul(from(x >> 16), slot.write[r]), bad, m);
          *reinterpret_cast<unsigned*>(sY + off) = word(a, b, 0, bad, m);
          *reinterpret_cast<unsigned*>(sSt + off) = word(a, b, 1, bad, m);
        }
        __syncthreads();
        wupdate<LDX>(accSt[0], sY, sSt, bCur, pm, nn, lane, slot.row[count - 1], bad);
        publish1438(slot, pm / 2, accSt[0]);
        __syncthreads();  // no weighted-X reader remains; both norms published
        if (tid < 2) tighten1438(budget, slot, tid);
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

      for (int i = warp; i < count; i += NW) {
        float r = 0.f;
        for (int j = lane; j <= i; j += 32)
          r = __fadd_ru(r, fabsf(from(bits(sSc[score_index(i, j)]))));
        for (int d = 16; d; d /= 2) r = __fadd_ru(r, __shfl_down_sync(~0u, r, d));
        if (!lane) scorePeak = fmaxf(scorePeak, r);
      }
      uint32_t sa = addr4<16>(sSc + score_tile(mi, 0), 0, 0, lane);
      uint32_t xb = addr4bt<LDX>(sX, 16 * ng, 0, lane);
      for (int low = 0; low < 2; low++) {
        for (int j = 0; j <= mi; j++) {
          uint32_t v[4], xx[4];
          ldm4(v, sa + 1024u * j + 512u * low);
          ldm4to<0>(xx, xb + 32u * j * LDX);
          mma_f32(accY[0], v, xx);
          mma_f32(accY[1], v, xx + 2);
        }
        for (int pt = 0; pt < 2; pt++)
          for (int q = 0; q < 4; q++) checked(accY[pt][q], bad, mask);
      }
      Peaks rp;
      uint32_t yy[4] = {};
      for (int pt = 0; pt < 2; pt++)
        for (int rb = 0; rb < 2; rb++) {
          int r = ia0 + 8 * rb, p = 16 * ng + 8 * pt + 2 * (lane & 3);
          if (r < count) {
            unsigned x = *reinterpret_cast<unsigned*>(sX + r * LDX + p);
            unsigned u = rp.finish(accY[pt][2 * rb], from(x), Dh, bad, mask);
            yy[2 * pt + rb] =
                u | (unsigned(rp.finish(accY[pt][2 * rb + 1], from(x >> 16), Dh, bad, mask)) << 16);
          }
        }
      peer_stm4(sptr(sY + (16 * mi + (lane & 15)) * LDX + 16 * ng + 8 * (lane >> 4)), yy[0], yy[1],
                yy[2], yy[3]);
      if (hasNext) wpub<PT>(sSt, accSt[0], pm, nn, lane, 0, bad);
      rp.reduce(slot, khalf);
      __syncthreads();
      if (tid < 2) collect(budget, slot, tid);
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
      __syncthreads();
    }
    if constexpr (FULL_T) {
      int tp = (nchunks - 1) * QT;
      for (int idx = tid; idx < XCH; idx += NTHR) {
        int r = idx / XRC, cc = idx % XRC;
        gst_cs128(gyh + (size_t)(tp + r) * XROW + cc * 8, *(const int4*)(sY + r * LDX + cc * 8));
      }
    }

    __syncthreads();  // all global output drains done
    double* peaks = reinterpret_cast<double*>(smem_raw);
    if (!lane) peaks[warp] = wide(scorePeak);
    __syncthreads();
    const int prior0 = __syncthreads_or(bad & 1);
    const int prior1 = __syncthreads_or(bad & 2);
    nonmetricMask = (prior0 != 0) | ((prior1 != 0) << 1);
    if (tid < 2) {
      double r = 0.;
      for (int w = 0; w < NW; w++) r = fmax(r, peaks[w]);
      if (refuses(budget, r)) bad |= 1 << tid;
    }
    int a = __syncthreads_or(bad & 1), b = __syncthreads_or(bad & 2);
    repairMask = (a != 0) | ((b != 0) << 1);
    if (tid == 0) {
      // The CTA votes precede this handoff. Each RMW acquires the
      // preceding completion and releases the accumulated chain.
      unsigned prior;
      asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"
                   : "=r"(prior)
                   : "l"(g_gdone + g)
                   : "memory");
    }
#undef LOAD_DT
#undef DT_SCAN
#undef ISSUE_ONE
#undef LOAD_X
#undef STORE_X
  }

  // Rendezvous after reset within THIS execution. Occupancy bounds are
  // not a stream dependency or a proof of isolation across graph replays.
  cooperative_groups::this_grid().sync();
  if (blockIdx.x == 0 && threadIdx.x == 0) gst_release(&ring_owner, 0ull);
  // No later ring access. Rewriting only the refused half does not need
  // producer residency, another launch or a decision carried across replays.
  if (repairMask) {
    extern __shared__ __align__(128) char smem_raw[];
    if (repairMask == 3 && nonmetricMask == 0) {
      const int tid = threadIdx.x;
      int v = win1306::tentative_recover<8, true>(gx, gdt, gA, gB, gC, gD, gdtb, gy, T, NH,
                                                  blockIdx.x, 0, smem_raw);
      // Both half certificates and their readers have retired.
      int* slot = reinterpret_cast<int*>(smem_raw);
      if (!(tid % 256)) slot[tid / 256] = v != 0;
      __syncthreads();
      int exactMask = slot[0] | (slot[1] << 1);
      __syncthreads();  // retire all mask reads before exact scratch reuse
      for (int k = 0; k < 2; k++)
        if (exactMask & (1 << k)) {
          win1306::pipeline<16>(gx, gdt, gA, gB, gC, gD, gdtb, gy, T, NH, blockIdx.x, 32 * k,
                                smem_raw);
          __syncthreads();
        }
    } else {
      for (int k = 0; k < 2; k++)
        if (repairMask & (1 << k)) {
          bool exact = true;
          if (!(nonmetricMask & (1 << k)))
            exact = win1306::tentative_recover<16>(gx, gdt, gA, gB, gC, gD, gdtb, gy, T, NH,
                                                   blockIdx.x, k, smem_raw) != 0;
          if (exact)
            win1306::pipeline<16>(gx, gdt, gA, gB, gC, gD, gdtb, gy, T, NH, blockIdx.x, 32 * k,
                                  smem_raw);
        }
    }
  }
}

template <int PT, bool FULL_T = false, int CHUNK_UNROLL = 1, bool TRACK_PROGRESS = true,
          bool SPEC = false>
__global__ __launch_bounds__(8 * PT, 1) void ssd_primary2051(
    const __nv_bfloat16* __restrict__ gx, const __nv_bfloat16* __restrict__ gdt,
    const float* __restrict__ gA, const __nv_bfloat16* __restrict__ gB,
    const __nv_bfloat16* __restrict__ gC, const __nv_bfloat16* __restrict__ gD,
    const __nv_bfloat16* __restrict__ gdtb, __nv_bfloat16* __restrict__ gy, int T, int NH,
    int nchunks, const __grid_constant__ CUtensorMap tmaB, const __grid_constant__ CUtensorMap tmaC,
    int* routes) {
  constexpr int HEADS = (PT == 64) ? 128 : 64;  // PT=64 -> NH128, PT=32 -> NH64
  constexpr int HEADS_PER_GROUP = HEADS / NGR;
  constexpr int NSPL = HDP / PT;

  using namespace win1306;
  int repairMask = 0, nonmetricMask = 0;
  static_assert(PT == 64, "regx ownership path is NH128 only");
  if (blockIdx.x < unsigned(NH) && threadIdx.x < 2) routes[2 * blockIdx.x + threadIdx.x] = 0;
  unsigned long long id;
  asm volatile("mov.u64 %0, %%gridid;" : "=l"(id));
  const unsigned long long token = id + 1ull;
  const bool eligible = token != 0ull && token != ring_poison;
  if (eligible && blockIdx.x == 0 && threadIdx.x == 0) {
    unsigned long long old;
    asm volatile("atom.acq_rel.gpu.global.cas.b64 %0, [%1], %2, %3;"
                 : "=l"(old)
                 : "l"(&ring_owner), "l"(0ull), "l"(token)
                 : "memory");
    // Only possible for a stranded or diagnostic owner, not a live peer.
    if (old == token) gst_release(&ring_owner, ring_poison);
  }
  cooperative_groups::this_grid().sync();
  if (!eligible || gld_acquire(&ring_owner) != token) {
    if (blockIdx.x < (unsigned)NH) {
      extern __shared__ __align__(128) char smem_raw[];
      for (int k = 0; k < 2; ++k)
        ordered(gx, gdt, gA, gB, gC, gD, gdtb, gy, T, NH, blockIdx.x, 32 * k, smem_raw);
    }
    return;
  }
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

    Slot& slot = *reinterpret_cast<Slot*>(sScan);
    int bad = 0;
    Budget budget;
    float scorePeak = 0.f;
    const float Dh = from(bits(gD[h]));
    if (tid == 0) {
      prefetch_tma(&tmaB);
      prefetch_tma(&tmaC);
      tma_barrier_init(sTmaBar);
      // Every warp arrives, not just the score producers: with the group Gram
      // lifted out, all sixteen warps read the C tile and the state image for
      // exactly as long, so the four non-score warps are no longer guaranteed
      // to be past that phase when the tile is recycled below.
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
    // Highest published window of this group that this thread has observed.
    unsigned int gready = 0;
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

      int count = min(QT, T - c * QT);
      static_assert(NW == 16 && ((0x55154ee0u >> 4) & 15u) == 14u);
      setup<false>(slot, sX, gx, gdt, gB, gC, gdtb, gD, gA, c * QT, count, NH, h, 0, bad, 0, 0, 0,
                   g_bc1952 + (g * GWIN + (c & (GWIN - 1))) * QT);
      if (warp < (count + 15) / 16) scan1423<false>(budget, slot, sX, c * QT, count, Dh, bad);
      if (c + 1 < nchunks) {
        LOAD_X(c + 1)
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

      const bool hasNext = c + 1 < nchunks;
      const bool nextFull = c + 2 <= T / QT;
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

      const int ia0 = 16 * mi + (lane >> 2), khalf = ng / 2, mask = 1 << khalf;
      float accY[2][4] = {};
      if (c) {
        wproject(accY, accSt[0], cCur, sSt, mi, ng, nn, lane, 0, bad);
        __syncthreads();
        wpub<PT>(sSt, accSt[0], pm, nn, lane, 1, bad);
        __syncthreads();
        wproject(accY, accSt[0], cCur, sSt, mi, ng, nn, lane, 1, bad);
        for (int pt = 0; pt < 2; pt++)
          for (int q = 0; q < 4; q++)
            accY[pt][q] = checked(mul(checked(accY[pt][q], bad, mask), slot.row[ia0 + 8 * (q / 2)]),
                                  bad, mask);
      }
      if (hasG) {
        auto emit = [&](int jj, int half, float4 f) {
          int j = 16 * jj + 8 * half + 2 * (lane & 3);
          float v[4] = {f.x, f.y, f.z, f.w};
          for (int rb = 0; rb < 2; rb++) {
            int i = ia0 + 8 * rb;
            for (int q = 0; q < 2; q++) {
              float& a = v[2 * rb + q];
              a = i < count && j + q <= i
                      ? checked(mul(checked(a, bad, 3), weight(slot, i, j + q, bad, 3)), bad, 3)
                      : 0.f;
            }
            auto dst = sSc + score_index(i, j);
            for (int low = 0; low < 2; low++)
              *reinterpret_cast<unsigned*>(dst + 256 * low) =
                  word(v[2 * rb], v[2 * rb + 1], low, bad, 3);
          }
        };
        if (gFull) {
          emit(gj, 0, gfrag.lo);
          emit(gj, 1, gfrag.hi);
        } else
          emit(gj, gk == 3, ghalf);
      }

      if (hasNext) load_gram(c + 1);
      __syncthreads();  // retire low readers, old Y drain; publish scores
      if (hasNext) {
        for (int pair = tid; pair < QT * PT / 2; pair += NTHR) {
          int r = pair / (PT / 2), p = 2 * (pair % (PT / 2)), off = r * LDX + p, m = 1 << (p / 32);
          unsigned x = *reinterpret_cast<unsigned*>(sX + off);
          float a = checked(mul(from(x), slot.write[r]), bad, m);
          float b = checked(mul(from(x >> 16), slot.write[r]), bad, m);
          *reinterpret_cast<unsigned*>(sY + off) = word(a, b, 0, bad, m);
          *reinterpret_cast<unsigned*>(sSt + off) = word(a, b, 1, bad, m);
        }
        __syncthreads();
        wupdate<LDX>(accSt[0], sY, sSt, bCur, pm, nn, lane, slot.row[count - 1], bad);
        publish1438(slot, pm / 2, accSt[0]);
        __syncthreads();  // no weighted-X reader remains; both norms published
        if (tid < 2) tighten1438(budget, slot, tid);
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

      for (int i = warp; i < count; i += NW) {
        float r = 0.f;
        for (int j = lane; j <= i; j += 32)
          r = __fadd_ru(r, fabsf(from(bits(sSc[score_index(i, j)]))));
        for (int d = 16; d; d /= 2) r = __fadd_ru(r, __shfl_down_sync(~0u, r, d));
        if (!lane) scorePeak = fmaxf(scorePeak, r);
      }
      uint32_t sa = addr4<16>(sSc + score_tile(mi, 0), 0, 0, lane);
      uint32_t xb = addr4bt<LDX>(sX, 16 * ng, 0, lane);
      for (int low = 0; low < 2; low++) {
        for (int j = 0; j <= mi; j++) {
          uint32_t v[4], xx[4];
          ldm4(v, sa + 1024u * j + 512u * low);
          ldm4to<0>(xx, xb + 32u * j * LDX);
          mma_f32(accY[0], v, xx);
          mma_f32(accY[1], v, xx + 2);
        }
        for (int pt = 0; pt < 2; pt++)
          for (int q = 0; q < 4; q++) checked(accY[pt][q], bad, mask);
      }
      Peaks rp;
      uint32_t yy[4] = {};
      for (int pt = 0; pt < 2; pt++)
        for (int rb = 0; rb < 2; rb++) {
          int r = ia0 + 8 * rb, p = 16 * ng + 8 * pt + 2 * (lane & 3);
          if (r < count) {
            unsigned x = *reinterpret_cast<unsigned*>(sX + r * LDX + p);
            unsigned u = rp.finish(accY[pt][2 * rb], from(x), Dh, bad, mask);
            yy[2 * pt + rb] =
                u | (unsigned(rp.finish(accY[pt][2 * rb + 1], from(x >> 16), Dh, bad, mask)) << 16);
          }
        }
      peer_stm4(sptr(sY + (16 * mi + (lane & 15)) * LDX + 16 * ng + 8 * (lane >> 4)), yy[0], yy[1],
                yy[2], yy[3]);
      if (hasNext) wpub<PT>(sSt, accSt[0], pm, nn, lane, 0, bad);
      rp.reduce(slot, khalf);
      __syncthreads();
      if (tid < 2) collect(budget, slot, tid);
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
      __syncthreads();
    }
    if constexpr (FULL_T) {
      int tp = (nchunks - 1) * QT;
      for (int idx = tid; idx < XCH; idx += NTHR) {
        int r = idx / XRC, cc = idx % XRC;
        gst_cs128(gyh + (size_t)(tp + r) * XROW + cc * 8, *(const int4*)(sY + r * LDX + cc * 8));
      }
    }

    __syncthreads();  // all global output drains done
    double* peaks = reinterpret_cast<double*>(smem_raw);
    if (!lane) peaks[warp] = wide(scorePeak);
    __syncthreads();
    const int prior0 = __syncthreads_or(bad & 1);
    const int prior1 = __syncthreads_or(bad & 2);
    nonmetricMask = (prior0 != 0) | ((prior1 != 0) << 1);
    if (tid < 2) {
      double r = 0.;
      for (int w = 0; w < NW; w++) r = fmax(r, peaks[w]);
      if (refuses(budget, r)) bad |= 1 << tid;
    }
    int a = __syncthreads_or(bad & 1), b = __syncthreads_or(bad & 2);
    repairMask = (a != 0) | ((b != 0) << 1);
    if (tid == 0) {
      // The CTA votes precede this handoff. Each RMW acquires the
      // preceding completion and releases the accumulated chain.
      unsigned prior;
      asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"
                   : "=r"(prior)
                   : "l"(g_gdone + g)
                   : "memory");
    }
#undef LOAD_DT
#undef DT_SCAN
#undef ISSUE_ONE
#undef LOAD_X
#undef STORE_X
  }

  // Rendezvous after reset within THIS execution. Occupancy bounds are
  // not a stream dependency or a proof of isolation across graph replays.
  cooperative_groups::this_grid().sync();
  if (blockIdx.x == 0 && threadIdx.x == 0) gst_release(&ring_owner, 0ull);
  // No later ring access. Rewriting only the refused half does not need
  // producer residency, another launch or a decision carried across replays.
  if (repairMask) {
    extern __shared__ __align__(128) char smem_raw[];
    int metric = repairMask & ~nonmetricMask;
    if (threadIdx.x < 2 && (metric & (1 << threadIdx.x)))
      routes[2 * blockIdx.x + threadIdx.x] = metric == 3 ? 8 : 16;
    for (int k = 0; k < 2; k++)
      if (nonmetricMask & (1 << k)) {
        win1306::pipeline<16>(gx, gdt, gA, gB, gC, gD, gdtb, gy, T, NH, blockIdx.x, 32 * k,
                              smem_raw);
        __syncthreads();
      }
  }
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
  stream1306(x.device().device_id, stream);
  int T = (int)x.shape()[0];
  int NH = (int)x.shape()[1];
  int nchunks = (T + QT - 1) / QT;
  if (T <= 0) return;

  const __nv_bfloat16* px = (const __nv_bfloat16*)effective1306(x);
  const __nv_bfloat16* pdt = (const __nv_bfloat16*)effective1306(dt);
  const float* pA = (const float*)effective1306(A);
  const __nv_bfloat16* pB = (const __nv_bfloat16*)effective1306(Bm);
  const __nv_bfloat16* pC = (const __nv_bfloat16*)effective1306(Cm);
  const __nv_bfloat16* pD = (const __nv_bfloat16*)effective1306(D);
  const __nv_bfloat16* pdb = (const __nv_bfloat16*)effective1306(dtb);
  __nv_bfloat16* py = (__nv_bfloat16*)effective1306(y);

  alignas(64) CUtensorMap tmaB{}, tmaC{};
  TVM_FFI_ICHECK_EQ(make_bc_tma(&tmaB, pB, T), CUDA_SUCCESS) << "failed to encode B tensor map";
  TVM_FFI_ICHECK_EQ(make_bc_tma(&tmaC, pC, T), CUDA_SUCCESS) << "failed to encode C tensor map";

  const int smem64 = (int)((3 * QT * LDN + 64 * LDN + 2 * QT * (64 + 8) + SCORE_WORDS) * 2 +
                           8 * QT * (int)sizeof(float) + 2 * sizeof(uint64_t) + 2 * sizeof(float));
  static_assert(smem64 >= win1306::DualSH, "dual replay scratch must fit NH128 launch");
  const int smem32 = (int)((3 * QT * LDN + 32 * LDN + 2 * QT * (32 + 8) + SCORE_WORDS) * 2 +
                           8 * QT * (int)sizeof(float) + 2 * sizeof(uint64_t) + 2 * sizeof(float));

  // cudaLaunchCooperativeKernel is a stream operation and is captured as a
  // cooperative kernel node.  There is deliberately no host launch counter or
  // other replay-varying state: the owning grid resets its ring protocol before the
  // grid-wide barrier on every execution, including every graph replay.
  auto launch_cooperative = [&](const void* kernel, int grid, int smem,
                                CooperativeLaunchState& state, int threads) {
    const int device = x.device().device_id;
    auto submit = [&] {
      void* args[] = {&px, &pdt, &pA, &pB, &pC, &pD, &pdb, &py, &T, &NH, &nchunks, &tmaB, &tmaC};
      TVM_FFI_CHECK_CUDA_ERROR(cudaGetLastError());  // no submission on pending error
      const cudaError_t launch_status =
          cudaLaunchCooperativeKernel(kernel, dim3(grid), dim3(threads), args, smem, stream);
      TVM_FFI_CHECK_CUDA_ERROR(launch_status);  // acceptance, not execution completion
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

    submit();
  };

  // Eight group-Gram producers ride on the SMs left idle by the 128 consumer
  // CTAs.  Capacity admission is for cooperative launch, not ring isolation.

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
    TVM_FFI_ICHECK(false) << "internal regx dispatch";
  }
}
static void Run2051(ffi::TensorView x, ffi::TensorView dt, ffi::TensorView A, ffi::TensorView Bm,
                    ffi::TensorView Cm, ffi::TensorView D, ffi::TensorView dtb, ffi::TensorView y,
                    int* routes, cudaStream_t& accepted_stream) {
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, x.device().device_id));
  stream1306(x.device().device_id, stream);
  accepted_stream = stream;
  int T = (int)x.shape()[0];
  int NH = (int)x.shape()[1];
  int nchunks = (T + QT - 1) / QT;
  if (T <= 0) return;

  const __nv_bfloat16* px = (const __nv_bfloat16*)effective1306(x);
  const __nv_bfloat16* pdt = (const __nv_bfloat16*)effective1306(dt);
  const float* pA = (const float*)effective1306(A);
  const __nv_bfloat16* pB = (const __nv_bfloat16*)effective1306(Bm);
  const __nv_bfloat16* pC = (const __nv_bfloat16*)effective1306(Cm);
  const __nv_bfloat16* pD = (const __nv_bfloat16*)effective1306(D);
  const __nv_bfloat16* pdb = (const __nv_bfloat16*)effective1306(dtb);
  __nv_bfloat16* py = (__nv_bfloat16*)effective1306(y);

  alignas(64) CUtensorMap tmaB{}, tmaC{};
  TVM_FFI_ICHECK_EQ(make_bc_tma(&tmaB, pB, T), CUDA_SUCCESS) << "failed to encode B tensor map";
  TVM_FFI_ICHECK_EQ(make_bc_tma(&tmaC, pC, T), CUDA_SUCCESS) << "failed to encode C tensor map";

  const int smem64 = (int)((3 * QT * LDN + 64 * LDN + 2 * QT * (64 + 8) + SCORE_WORDS) * 2 +
                           8 * QT * (int)sizeof(float) + 2 * sizeof(uint64_t) + 2 * sizeof(float));
  static_assert(smem64 >= win1306::DualSH, "dual replay scratch must fit NH128 launch");
  const int smem32 = (int)((3 * QT * LDN + 32 * LDN + 2 * QT * (32 + 8) + SCORE_WORDS) * 2 +
                           8 * QT * (int)sizeof(float) + 2 * sizeof(uint64_t) + 2 * sizeof(float));

  // cudaLaunchCooperativeKernel is a stream operation and is captured as a
  // cooperative kernel node.  There is deliberately no host launch counter or
  // other replay-varying state: the owning grid resets its ring protocol before the
  // grid-wide barrier on every execution, including every graph replay.
  static CooperativeLaunchState g_coop64_2051, g_coop64_noprog_2051, g_coop64_full_2051,
      g_coop64_full_noprog_2051;
  auto launch_cooperative = [&](const void* kernel, int grid, int smem,
                                CooperativeLaunchState& state, int threads) {
    const int device = x.device().device_id;
    auto submit = [&] {
      void* args[] = {&px, &pdt, &pA, &pB,      &pC,   &pD,   &pdb,
                      &py, &T,   &NH, &nchunks, &tmaB, &tmaC, &routes};
      TVM_FFI_CHECK_CUDA_ERROR(cudaGetLastError());  // no submission on pending error
      const cudaError_t launch_status =
          cudaLaunchCooperativeKernel(kernel, dim3(grid), dim3(threads), args, smem, stream);
      TVM_FFI_CHECK_CUDA_ERROR(launch_status);  // acceptance, not execution completion
    };
    bool submitted = false;
    std::call_once(state.initialize, [&] {
      win1306::prepare2051<16>();
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

    submit();
  };

  // Eight group-Gram producers ride on the SMs left idle by the 128 consumer
  // CTAs.  Capacity admission is for cooperative launch, not ring isolation.

  if (NH >= 96) {
    if ((T & 63) == 0) {
      if (nchunks > GWIN)
        launch_cooperative((const void*)ssd_primary2051<64, true, 4, true, false>, NH + NGR, smem64,
                           g_coop64_full_2051, 512);
      else
        launch_cooperative((const void*)ssd_primary2051<64, true, 4, false, false>, NH + NGR,
                           smem64, g_coop64_full_noprog_2051, 512);
    } else {
      if (nchunks > GWIN)
        launch_cooperative((const void*)ssd_primary2051<64, false, 1, true, false>, NH + NGR,
                           smem64, g_coop64_2051, 512);
      else
        launch_cooperative((const void*)ssd_primary2051<64, false, 1, false, false>, NH + NGR,
                           smem64, g_coop64_noprog_2051, 512);
    }
  } else {
    TVM_FFI_ICHECK(false) << "internal regx dispatch";
  }
}
}  // namespace regx
#undef DI

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

#include <atomic>

#include "ssd_tile_recovery.cuh"

enum class WorkspaceContract2153 { Strict, FrameworkTensor };
static thread_local bool inWorkspaceFactory2350 = false;

// Called only for the effective current context. Skip a primary observed
// inactive. The subsequent balanced retain can race another thread's final
// release; it does not change this thread's current context or context stack.
static int isPrimary2153(CUdevice device, CUcontext cur) {
  unsigned flags = 0;
  int active = 0;
  TVM_FFI_ICHECK(cuDevicePrimaryCtxGetState(device, &flags, &active) == CUDA_SUCCESS)
      << "cannot query primary context state";
  if (!active) return 0;
  CUcontext primary = nullptr;
  TVM_FFI_ICHECK(cuDevicePrimaryCtxRetain(&primary, device) == CUDA_SUCCESS)
      << "cannot retain active primary context";
  const int match = (primary == cur);
  TVM_FFI_ICHECK(cuDevicePrimaryCtxRelease(device) == CUDA_SUCCESS)
      << "cannot release classification retain";
  return match;
}

static void common2051(ffi::TensorView x, ffi::TensorView dt, ffi::TensorView A,
                       ffi::TensorView Bmat, ffi::TensorView Cmat, ffi::TensorView D,
                       ffi::TensorView dt_bias, ffi::TensorView y, const ffi::TensorView* workspace,
                       WorkspaceContract2153 contract = WorkspaceContract2153::Strict,
                       const ffi::Function* factory = nullptr) {
  TVM_FFI_ICHECK(!inWorkspaceFactory2350) << "reentrant SSD workspace factory";
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
  TVM_FFI_ICHECK_EQ(x.ndim(), 3) << "x must have rank 3";
  const int64_t T = x.shape()[0], NH = x.shape()[1];
  // Both the chunk ceiling and the padded token indices use signed int arithmetic.
  TVM_FFI_ICHECK(T > 0 && T <= INT_MAX - 127) << "invalid SSD sequence length";
  TVM_FFI_ICHECK(NH == 64 || NH == 128) << "SSD requires 64 or 128 heads";
  constexpr DLDataType bf{kDLBfloat, 16, 1}, fp{kDLFloat, 32, 1};
  auto check_form = [](const ffi::TensorView& t, const char* name,
                       std::initializer_list<int64_t> shape, DLDataType expected) {
    TVM_FFI_ICHECK_EQ(t.ndim(), static_cast<int>(shape.size())) << name << " has wrong rank";
    int i = 0;
    for (int64_t extent : shape) {
      TVM_FFI_ICHECK_EQ(t.shape()[i], extent) << name << " has wrong shape";
      ++i;
    }
    auto dtype = t.dtype();
    TVM_FFI_ICHECK(dtype.code == expected.code && dtype.bits == expected.bits &&
                   dtype.lanes == expected.lanes)
        << name << " has wrong dtype";
    TVM_FFI_ICHECK(t.IsContiguous()) << name << " must be contiguous";
  };
  check_form(x, "x", {T, NH, 64}, bf);
  check_form(dt, "dt", {T, NH}, bf);
  check_form(A, "A", {NH}, fp);
  check_form(Bmat, "B", {T, 8, 128}, bf);
  check_form(Cmat, "C", {T, 8, 128}, bf);
  check_form(D, "D", {NH}, bf);
  check_form(dt_bias, "dt_bias", {NH}, bf);
  check_form(y, "y", {T, NH, 64}, bf);
  auto address = [](const ffi::TensorView& t) {
    return reinterpret_cast<uintptr_t>(effective1306(t));
  };
  TVM_FFI_ICHECK(((address(x) | address(Bmat) | address(Cmat) | address(y)) & 15) == 0)
      << "x, B, C and y must be 16-byte aligned";
  const size_t tokens = T, heads = NH;
  const size_t dt_bytes = tokens * heads * sizeof(bf16);
  const size_t y_bytes = dt_bytes * 64, bc_bytes = tokens * 8 * 128 * sizeof(bf16);

  TVM_FFI_ICHECK(!((address(dt) | address(D) | address(dt_bias)) & 1) && !(address(A) & 3))
      << "scalar alignment";
  const uintptr_t dst = address(y);
  TVM_FFI_ICHECK(y_bytes <= UINTPTR_MAX - dst) << "output span overflow";
  auto disjoint = [&](const ffi::TensorView& input, size_t bytes, const char* name) {
    const uintptr_t src = address(input);
    TVM_FFI_ICHECK(bytes <= UINTPTR_MAX - src) << "input span overflow";
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
  TVM_FFI_CHECK_CUDA_ERROR(cudaGetLastError());
  TVM_FFI_ICHECK(cuInit(0) == CUDA_SUCCESS) << "CUDA driver initialization failed";
  CUcontext cur = nullptr;
  CUdevice current = -1;
  TVM_FFI_ICHECK(cuCtxGetCurrent(&cur) == CUDA_SUCCESS) << "cannot query current context";
  if (cur) {
    TVM_FFI_ICHECK(cuCtxGetDevice(&current) == CUDA_SUCCESS) << "cannot query context device";
  }

  // Preserve a matching current context, primary OR non-primary. Only
  // select the requested device's primary when a device switch is needed
  // (including a caller with no current context). Restore the caller's stack.
  struct ContextGuard {
    CUdevice device;
    bool retained = false, pushed = false;
    ~ContextGuard() {
      CUcontext popped = nullptr;
      if (pushed) (void)cuCtxPopCurrent(&popped);
      if (retained) (void)cuDevicePrimaryCtxRelease(device);
    }
  } guard{dev.device_id};
  if (!cur || current != dev.device_id) {
    // Initialize the target through the runtime so it retains
    // its normal device initialization/lifetime, not a transient driver
    // retain whose release could tear down accepted asynchronous work.
    TVM_FFI_CHECK_CUDA_ERROR(cudaInitDevice(dev.device_id, 0, 0));
    TVM_FFI_ICHECK(cuDevicePrimaryCtxRetain(&cur, guard.device) == CUDA_SUCCESS)
        << "cannot select target device context";
    guard.retained = true;
    TVM_FFI_ICHECK(cuCtxPushCurrent(cur) == CUDA_SUCCESS) << "cannot push target context";
    guard.pushed = true;
  }

  // A context ID also distinguishes a destroyed context from a later one
  // whose opaque handle happens to have the same address.
  int* routes = nullptr;
  cudaStream_t stream = nullptr;
  // Tensor ownership and the allocator preserve stream/graph lifetime.
  ffi::Tensor owner;  // destroyed before guard, also on submission failure
  cudaStreamCaptureStatus expected_factory_capture = cudaStreamCaptureStatusNone;
  const size_t base_bytes = 2 * heads * (sizeof(int) + 16 * sizeof(win1306::Record2051));
  const size_t tile_offset = (base_bytes + 15) & ~size_t(15);
  size_t bytes = base_bytes;
  tile_recovery::Layout tile_layout{};
  tile_recovery::Arena* arena = nullptr;
  static_assert(offsetof(tile_recovery::Head, status) == 0);
  auto validate_workspace = [&](const ffi::TensorView& ws) {
    same_device(ws, "workspace");
    check_form(ws, "workspace", {int64_t(bytes)}, DLDataType{kDLUInt, 8, 1});
    uintptr_t p = address(ws), declared = reinterpret_cast<uintptr_t>(ws.data_ptr());
    TVM_FFI_ICHECK(!(p & 15) && bytes <= UINTPTR_MAX - p) << "workspace alignment/span";
    // A framework tensor supplies valid backing storage and asynchronous
    // lifetime. VMM and stream-ordered allocations need not have an owning
    // context; require local device placement and current-device accessibility.
    if (contract == WorkspaceContract2153::Strict) {
      CUdeviceptr base = 0;
      size_t len = 0;
      CUcontext owner = nullptr;
      unsigned kind = 0, managed = 1;
      int ordinal = -1;
      auto attr = [&](auto* out, CUpointer_attribute key) {
        TVM_FFI_ICHECK(cuPointerGetAttribute(out, key, CUdeviceptr(declared)) == CUDA_SUCCESS)
            << "cannot validate workspace allocation";
      };
      attr(&owner, CU_POINTER_ATTRIBUTE_CONTEXT);
      attr(&ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL);
      attr(&kind, CU_POINTER_ATTRIBUTE_MEMORY_TYPE);
      attr(&managed, CU_POINTER_ATTRIBUTE_IS_MANAGED);
      TVM_FFI_ICHECK(owner == cur && ordinal == dev.device_id && kind == CU_MEMORYTYPE_DEVICE &&
                     !managed)
          << "workspace must be local nonmanaged device memory in target context";
      TVM_FFI_ICHECK(cuMemGetAddressRange(&base, &len, CUdeviceptr(declared)) == CUDA_SUCCESS)
          << "workspace requires allocation with queryable device allocation bounds";
      TVM_FFI_ICHECK(base <= declared && declared - base < len);
      size_t remaining = len - (declared - base);
      TVM_FFI_ICHECK(ws.byte_offset() <= remaining && bytes <= remaining - ws.byte_offset())
          << "workspace escapes its declared allocation";
    } else {
      auto framework_stream = (cudaStream_t)TVMFFIEnvGetStream(dev.device_type, dev.device_id);
      stream1306(dev.device_id, framework_stream);
      cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
      TVM_FFI_CHECK_CUDA_ERROR(cudaStreamIsCapturing(framework_stream, &capture));
      TVM_FFI_ICHECK(capture != cudaStreamCaptureStatusInvalidated)
          << "framework workspace stream capture is invalidated";
      auto check_at = [&](CUdeviceptr address) {
        auto get = [&](auto* out, CUpointer_attribute key) {
          CUresult result = cuPointerGetAttribute(out, key, address);
          TVM_FFI_ICHECK(result == CUDA_SUCCESS)
              << "framework workspace attribute " << int(key) << " failed: " << int(result);
        };
        CUcontext owner = nullptr;
        unsigned kind = 0, managed = 1;
        int ordinal = -1;
        get(&owner, CU_POINTER_ATTRIBUTE_CONTEXT);
        get(&ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL);
        get(&kind, CU_POINTER_ATTRIBUTE_MEMORY_TYPE);
        get(&managed, CU_POINTER_ATTRIBUTE_IS_MANAGED);
        TVM_FFI_ICHECK(kind == CU_MEMORYTYPE_DEVICE && ordinal == dev.device_id && !managed)
            << "framework workspace must be local nonmanaged device memory";
        if (owner) {
          TVM_FFI_ICHECK(owner == cur) << "foreign framework workspace context";
        } else {
          unsigned mapped = 0, access = 0;
          get(&mapped, CU_POINTER_ATTRIBUTE_MAPPED);
          get(&access, CU_POINTER_ATTRIBUTE_ACCESS_FLAGS);
          TVM_FFI_ICHECK(mapped && access == CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READWRITE)
              << "framework workspace must be mapped read/write on the current device";
        }
      };
      // Endpoints check accessibility; the producer's valid-tensor contract
      // covers the backing and permissions throughout the complete span.
      // Captured stream-ordered allocations are graph allocation nodes:
      // their pointers need not be queryable/mapped until graph execution.
      // The framework producer guarantees storage before the first use and
      // through the last use, including each graph replay. This trusted
      // tensor contract does not authenticate arbitrary forged DLTensors.
      TVM_FFI_ICHECK(!factory || capture == expected_factory_capture)
          << "factory changed stream capture state";
      if (capture == cudaStreamCaptureStatusNone) {
        check_at(CUdeviceptr(p));
        check_at(CUdeviceptr(p + bytes - 1));
      }
    }
    auto apart = [&](const ffi::TensorView& t, size_t n) {
      uintptr_t a = address(t);  // original validation already checked a+n
      TVM_FFI_ICHECK(a + n <= p || p + bytes <= a) << "workspace overlaps an operand";
    };
    apart(x, y_bytes);
    apart(y, y_bytes);
    apart(dt, dt_bytes);
    apart(A, heads * sizeof(float));
    apart(Bmat, bc_bytes);
    apart(Cmat, bc_bytes);
    apart(D, heads * sizeof(bf16));
    apart(dt_bias, heads * sizeof(bf16));
    routes = reinterpret_cast<int*>(p);
  };
  if (workspace) validate_workspace(*workspace);
  unsigned long long ctx_id = 0;
  TVM_FFI_ICHECK(cuCtxGetId(cur, &ctx_id) == CUDA_SUCCESS) << "cannot identify target context";
  static std::once_flag bind;
  static int bound_device = -1;
  static unsigned long long bound_ctx = 0;
  static std::atomic<bool> binding_ready{false};
  auto check_binding = [&] {
    TVM_FFI_ICHECK_EQ(bound_device, dev.device_id) << "SSD bound to another CUDA device";
    TVM_FFI_ICHECK_EQ(bound_ctx, ctx_id) << "SSD bound to another CUDA context";
  };
  // Reject a known mismatch before invoking Python or allocating scratch.
  const bool already_bound = binding_ready.load(std::memory_order_acquire);
  if (already_bound) check_binding();
  // Scoped to this bind flag, not indexed by a reusable context handle.
  static std::atomic<int> primary_cache{-1};
  int probed = -1;
  // Python acquisition stays outside call_once: it may release the GIL or
  // invoke allocator hooks. Cold racing calls can each acquire scratch;
  // call_once and the post-wait identity check still admit only one context.
  {
    if (factory) {
      stream = (cudaStream_t)TVMFFIEnvGetStream(dev.device_type, dev.device_id);
      stream1306(dev.device_id, stream);
      cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
      TVM_FFI_CHECK_CUDA_ERROR(cudaStreamIsCapturing(stream, &status));
      TVM_FFI_ICHECK(status != cudaStreamCaptureStatusInvalidated)
          << "invalidated SSD stream capture";
      if (status == cudaStreamCaptureStatusNone || status == cudaStreamCaptureStatusActive) {
        int primary = already_bound ? primary_cache.load(std::memory_order_relaxed) : -1;
        if (primary < 0) primary = probed = isPrimary2153(dev.device_id, cur);
        if (primary) {
          const bool admitted = tile_recovery::try_layout(T, NH, tile_layout);
          if (admitted) bytes = tile_offset + size_t(tile_layout.used);
          expected_factory_capture = status;
          ffi::Any result;
          {
            struct FactoryScope {
              FactoryScope() { inWorkspaceFactory2350 = true; }
              ~FactoryScope() { inWorkspaceFactory2350 = false; }
            } factory_scope;
            result = (*factory)(int64_t(bytes), dev.device_id,
                                uint64_t(reinterpret_cast<uintptr_t>(stream)));
          }
          owner = std::move(result).cast<ffi::Tensor>();
          CUcontext after = nullptr;
          TVM_FFI_ICHECK(cuCtxGetCurrent(&after) == CUDA_SUCCESS && after == cur)
              << "workspace factory changed current context";
          TVM_FFI_ICHECK((cudaStream_t)TVMFFIEnvGetStream(dev.device_type, dev.device_id) == stream)
              << "workspace factory changed FFI stream";
          TVM_FFI_CHECK_CUDA_ERROR(cudaGetLastError());
          // Only the lazy factory may decline optional tile scratch by
          // returning the original small workspace. Strict and explicit
          // framework workspace contracts retain their exact original size.
          const ffi::TensorView owned_view(owner);
          const bool use_tile =
              admitted && !(owned_view.ndim() == 1 && owned_view.shape()[0] == int64_t(base_bytes));
          if (!use_tile) bytes = base_bytes;
          validate_workspace(owned_view);
          if (use_tile) {
            arena = reinterpret_cast<tile_recovery::Arena*>(routes) + tile_offset;
            tile_recovery::prepare();
          }
        }
      }
    }
  }
  auto launch = [&] {
    if (workspace || owner.defined()) {
      if (NH == 128)
        regx::Run2051(x, dt, A, Bmat, Cmat, D, dt_bias, y, routes, stream);
      else
        RunPeer2051(x, dt, A, Bmat, Cmat, D, dt_bias, y, routes, stream);
    } else if (NH == 128)
      regx::Run(x, dt, A, Bmat, Cmat, D, dt_bias, y);
    else
      RunPeer(x, dt, A, Bmat, Cmat, D, dt_bias, y);
    // Same acceptance point as the existing binding: primary submission,
    // not completion of finish. Nothing publishes on a throwing submission.
    if (probed >= 0) primary_cache.store(probed, std::memory_order_relaxed);
  };
  bool first = false;
  std::call_once(bind, [&] {
    launch();  // validation or submission exception leaves bind retriable
    bound_device = dev.device_id;
    bound_ctx = ctx_id;
    binding_ready.store(true, std::memory_order_release);
    first = true;
  });
  if (!first) {
    check_binding();
    launch();
  }
  if (workspace || owner.defined()) {
    if (arena) {
      tile_recovery::Args args{(const bf16*)address(x),
                               (const bf16*)address(dt),
                               (const float*)address(A),
                               (const bf16*)address(Bmat),
                               (const bf16*)address(Cmat),
                               (const bf16*)address(D),
                               (const bf16*)address(dt_bias),
                               (bf16*)address(y),
                               int(T),
                               int(NH)};
      tile_recovery::submit(args, routes, arena, tile_layout, stream);
    }
    auto finish = NH == 128 ? win1306::finish2051<16> : win1306::finish2051<8>;
    finish((const bf16*)address(x), (const bf16*)address(dt), (const float*)address(A),
           (const bf16*)address(Bmat), (const bf16*)address(Cmat), (const bf16*)address(D),
           (const bf16*)address(dt_bias), (bf16*)address(y), int(T), int(NH), routes, stream,
           reinterpret_cast<const int*>(arena));
  }
}

using View2051 = ffi::TensorView;
void Run(View2051 x, View2051 dt, View2051 A, View2051 B, View2051 C, View2051 D, View2051 bias,
         View2051 y) {
  common2051(x, dt, A, B, C, D, bias, y, nullptr);
}
void RunWorkspace2051(View2051 x, View2051 dt, View2051 A, View2051 B, View2051 C, View2051 D,
                      View2051 bias, View2051 y, View2051 workspace) {
  common2051(x, dt, A, B, C, D, bias, y, &workspace);
}
// Internal framework binding: workspace has valid, sufficiently backed device
// storage in this context and remains live until all accepted stream work ends.
void RunFrameworkWorkspace2153(View2051 x, View2051 dt, View2051 A, View2051 B, View2051 C,
                               View2051 D, View2051 bias, View2051 y, View2051 workspace) {
  common2051(x, dt, A, B, C, D, bias, y, &workspace, WorkspaceContract2153::FrameworkTensor);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(_run_framework_workspace, RunFrameworkWorkspace2153);

void RunLazy2153(View2051 x, View2051 dt, View2051 A, View2051 B, View2051 C, View2051 D,
                 View2051 bias, View2051 y, ffi::Function factory) {
  TVM_FFI_ICHECK(factory != nullptr) << "missing SSD workspace factory";
  common2051(x, dt, A, B, C, D, bias, y, nullptr, WorkspaceContract2153::FrameworkTensor, &factory);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(_run_framework_lazy, RunLazy2153);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, Run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_workspace, RunWorkspace2051);
