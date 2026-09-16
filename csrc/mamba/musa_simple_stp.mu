// Native MUSA Simple STP selective state update.
//
// This is the 4-warp SM90 Simple-STP dataflow specialized for the S5000
// Nemotron decode shape (B=1,H=64,D=64,N=128,G=8).  One CTA owns four
// consecutive D rows; each warp owns one row and each lane owns four N values.
// x/B/C/z are staged once in shared memory, while the state is moved as a
// packed uint2 (four FP16 values).  All recurrence arithmetic is FP32.

#include <ATen/Utils.h>
#include <torch/all.h>
#include <torch/extension.h>
#include <musa_runtime.h>
#include <musa_fp16.h>
#include <musa_bf16.h>
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSAStream.h"

#include <cstdint>
#include <cmath>

namespace {
constexpr int kThreads = 128;
constexpr int kWarp = 32;
constexpr int kRows = 4;
constexpr int kN = 128;
constexpr int kGroup = 4;
constexpr int kPhiloxKeyA = 0x9E3779B9u;
constexpr int kPhiloxKeyB = 0xBB67AE85u;
constexpr uint32_t kPhiloxMulA = 0xD2511F53u;
constexpr uint32_t kPhiloxMulB = 0xCD9E8D57u;

template <typename T> __device__ __forceinline__ float as_float(T v) {
  return static_cast<float>(v);
}
template <> __device__ __forceinline__ float as_float<__half>(__half v) {
  return __half2float(v);
}
template <> __device__ __forceinline__ float as_float<__mt_bfloat16>(__mt_bfloat16 v) {
  return static_cast<float>(v);
}

template <typename T> __device__ __forceinline__ T from_float(float v) {
  return static_cast<T>(v);
}
template <> __device__ __forceinline__ __half from_float<__half>(float v) {
  return __float2half(v);
}
template <> __device__ __forceinline__ __mt_bfloat16 from_float<__mt_bfloat16>(float v) {
  return __float2bfloat16(v);
}

__device__ __forceinline__ float fast_exp(float x) { return expf(x); }
__device__ __forceinline__ float softplus(float x) {
  return x > 20.f ? x : log1pf(expf(x));
}

// Philox-4x32, with the counter split into low/high words exactly as the
// Triton helper in musa_stochastic.py.  The caller supplies a group base
// offset, so each group consumes one four-word result.
template <int R>
__device__ __forceinline__ uint4 philox4x32(int64_t seed, int64_t offset) {
  uint64_t s = static_cast<uint64_t>(seed), o = static_cast<uint64_t>(offset);
  uint32_t c0 = static_cast<uint32_t>(o), c1 = static_cast<uint32_t>(o >> 32);
  uint32_t c2 = 0, c3 = 0;
  uint32_t k0 = static_cast<uint32_t>(s), k1 = static_cast<uint32_t>(s >> 32);
#pragma unroll
  for (int i = 0; i < R; ++i) {
    uint32_t a = c0, b = c2;
    c0 = __umulhi(kPhiloxMulB, b) ^ c1 ^ k0;
    c2 = __umulhi(kPhiloxMulA, a) ^ c3 ^ k1;
    c1 = kPhiloxMulB * b;
    c3 = kPhiloxMulA * a;
    k0 += kPhiloxKeyA; k1 += kPhiloxKeyB;
  }
  return make_uint4(c0, c1, c2, c3);
}

__device__ __forceinline__ uint16_t stochastic_f16(float x, uint32_t rnd) {
  uint32_t bits = __float_as_uint(x);
  uint32_t sign = bits & 0x80000000u;
  uint32_t mag = (bits & 0x7fffffffu) + (rnd & 0x1fffu);
  uint32_t e = (mag >> 23) & 0xffu;
  uint32_t m = mag & 0x7fffffu;
  uint16_t h;
  if (e == 255) h = m ? 0x7e00u : 0x7c00u;
  else if (e > 142) h = 0x7c00u;
  else if (e < 113) h = 0;
  else h = static_cast<uint16_t>(((e - 112) << 10) | (m >> 13));
  return static_cast<uint16_t>(sign >> 16) | h;
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int d = 16; d; d >>= 1) x += __shfl_down_sync(0xffffffffu, x, d);
  return x;
}

template <typename InT, typename DType, typename IndexT, int ROUNDS>
__global__ void simple_stp_kernel(
    __half* __restrict__ state, const InT* __restrict__ x, const float* __restrict__ dt,
    const float* __restrict__ A, const InT* __restrict__ B, const InT* __restrict__ C,
    const DType* __restrict__ Dv, const float* __restrict__ bias, const InT* __restrict__ z,
    const IndexT* __restrict__ src_slots, const IndexT* __restrict__ dst_slots,
    InT* __restrict__ out, const int64_t* __restrict__ seed,
    int64_t ss, int64_t sh, int64_t sd, int64_t sn,
    int64_t xb, int64_t xh, int64_t xd, int64_t dtb, int64_t dth, int64_t dtd,
    int64_t ah, int64_t ad, int64_t an, int64_t bb, int64_t bg, int64_t bn,
    int64_t cb, int64_t cg, int64_t cn, int64_t dh, int64_t dd,
    int64_t bh, int64_t bd, int64_t ob, int64_t oh, int64_t od,
    int64_t zb, int64_t zh, int64_t zd, int64_t pad, int slots,
    bool has_bias, bool bias_matrix, bool has_z, bool softplus_flag,
    bool d_vector, bool tie_hdim, bool use_sr) {
  int pid = blockIdx.x, batch = pid / 1024, tile = pid % 1024;
  int head = tile / 16, row = tile % 16;
  int d = row * kRows + (threadIdx.x >> 5);
  int lane = threadIdx.x & 31;
  if (batch >= 1 || head >= 64 || d >= 64) return;
  int group = head / 8;
  __shared__ InT sx[kRows], sb[kN], sc[kN], sz[kRows];
  __shared__ float sdt, sa, sbias;
  if (threadIdx.x < kRows) {
    sx[threadIdx.x] = x[batch * xb + head * xh + (row * kRows + threadIdx.x) * xd];
    sz[threadIdx.x] = has_z ? z[batch * zb + head * zh + (row * kRows + threadIdx.x) * zd] : from_float<InT>(0.f);
  }
  for (int n = threadIdx.x; n < kN; n += kThreads) {
    sb[n] = B[batch * bb + group * bg + n * bn];
    sc[n] = C[batch * cb + group * cg + n * cn];
  }
  if (threadIdx.x == 0) {
    sdt = dt[batch * dtb + head * dth + (tie_hdim ? 0 : d * dtd)];
    sa = A[head * ah + (tie_hdim ? 0 : d * ad) + (tie_hdim ? 0 : 0 * an)];
    sbias = has_bias ? (bias_matrix ? bias[head * bh + (tie_hdim ? 0 : d * bd)] : bias[head * bh]) : 0.f;
  }
  __syncthreads();
  float dtv = sdt + sbias, av = sa;
  if (softplus_flag) dtv = softplus(dtv);
  float da = fast_exp(av * dtv), accum = 0.f;
  IndexT src_i = src_slots[batch], dst_i = dst_slots[batch];
  bool src_ok = src_i >= 0 && src_i < slots && src_i != pad;
  bool dst_ok = dst_i >= 0 && dst_i < slots && dst_i != pad;
  int n0 = lane * kGroup;
  int64_t base = static_cast<int64_t>(src_i) * ss + head * sh + d * sd;
  int64_t dst_base = static_cast<int64_t>(dst_i) * ss + head * sh + d * sd;
  uint2 packed = make_uint2(0, 0);
  if (src_ok) packed = *reinterpret_cast<const uint2*>(state + base + n0 * sn);
  __half* vals = reinterpret_cast<__half*>(&packed);
  uint2 stored = make_uint2(0, 0);
  __half* sout = reinterpret_cast<__half*>(&stored);
  uint4 rnd = make_uint4(0, 0, 0, 0);
  if (use_sr && seed != nullptr) rnd = philox4x32<ROUNDS>(*seed, base + n0 * sn);
#pragma unroll
  for (int q = 0; q < kGroup; ++q) {
    int n = n0 + q;
    float oldv = as_float(vals[q]);
    float upd = oldv * da + dtv * as_float(sb[n]) * as_float(sx[threadIdx.x >> 5]);
    if (use_sr) sout[q] = __ushort_as_half(stochastic_f16(upd, q == 0 ? rnd.x : q == 1 ? rnd.y : q == 2 ? rnd.z : rnd.w));
    else sout[q] = __float2half(upd);
    accum += as_float(sc[n]) * upd;
  }
  if (dst_ok) *reinterpret_cast<uint2*>(state + dst_base + n0 * sn) = stored;
  float y = warp_sum(accum);
  if (lane == 0) {
    float xv = as_float(sx[threadIdx.x >> 5]);
    float dv = d_vector ? as_float(Dv[head * dh]) : as_float(Dv[head * dh + d * dd]);
    y += dv * xv;
    if (has_z) { float zv = as_float(sz[threadIdx.x >> 5]); y *= zv / (1.f + expf(-zv)); }
    out[batch * ob + head * oh + d * od] = from_float<InT>(y);
  }
}

#define LAUNCH(IN, DT, IDX, R, HB, BM, HZ, SP, DV, TH, SR) \
  simple_stp_kernel<IN, DT, IDX, R> \
    <<<1024, kThreads, 0, stream>>>(reinterpret_cast<__half*>(state.data_ptr()), reinterpret_cast<const IN*>(x.data_ptr()), reinterpret_cast<const float*>(dt.data_ptr()), reinterpret_cast<const float*>(A.data_ptr()), reinterpret_cast<const IN*>(B.data_ptr()), reinterpret_cast<const IN*>(C.data_ptr()), reinterpret_cast<const DT*>(Dv.data_ptr()), bias_ptr, reinterpret_cast<const IN*>(z_tensor.data_ptr()), reinterpret_cast<const IDX*>(src.data_ptr()), reinterpret_cast<const IDX*>(dst.data_ptr()), reinterpret_cast<IN*>(out.data_ptr()), seed_ptr, ss,sh,sd,sn,xb,xh,xd,dtb,dth,dtd,ah,ad,an,bb,bg,bn,cb,cg,cn,dh,dd,bh,bd,ob,oh,od,zb,zh,zd,pad_slot_id,state.size(0),HB,BM,HZ,SP,DV,TH,SR)

}  // namespace

at::Tensor musa_ssu_simple(
    at::Tensor state, at::Tensor x, at::Tensor dt, at::Tensor A,
    at::Tensor B, at::Tensor C, at::Tensor Dv, at::Tensor src, at::Tensor dst,
    c10::optional<at::Tensor> dt_bias, c10::optional<at::Tensor> z,
    bool dt_softplus, int64_t pad_slot_id, c10::optional<at::Tensor> out_opt,
    c10::optional<at::Tensor> rand_seed, int64_t philox_rounds) {
  TORCH_CHECK(state.device().is_privateuseone(), "state must be MUSA");
  TORCH_CHECK(state.dim() == 4 && state.size(1) == 64 && state.size(2) == 64 && state.size(3) == 128,
              "native Simple STP requires state [slots,64,64,128]");
  TORCH_CHECK(x.dim() == 3 && x.size(0) == 1 && x.size(1) == 64 && x.size(2) == 64, "x must be [1,64,64]");
  TORCH_CHECK(state.scalar_type() == at::kHalf && x.scalar_type() == at::kBFloat16,
              "native Simple STP requires fp16 state and bf16 x");
  TORCH_CHECK(state.is_contiguous(), "native Simple STP requires contiguous state for packed uint2 traffic");
  TORCH_CHECK(B.dim() == 3 && C.dim() == 3 && B.size(0) == 1 && B.size(1) == 8 && B.size(2) == 128 && C.sizes() == B.sizes(), "B/C must be [1,8,128]");
  TORCH_CHECK(B.scalar_type() == x.scalar_type() && C.scalar_type() == x.scalar_type(), "B/C must match x dtype");
  TORCH_CHECK(dt.dim() == 3 && dt.size(0) == 1 && dt.size(1) == 64 && dt.size(2) == 64, "dt must be [1,64,64]");
  TORCH_CHECK(A.dim() == 3 && A.size(0) == 64 && A.size(1) == 64 && A.size(2) == 128, "A must be [64,64,128]");
  TORCH_CHECK(dt.scalar_type() == at::kFloat && A.scalar_type() == at::kFloat, "dt and A must be fp32");
  TORCH_CHECK(A.stride(1) == 0 && A.stride(2) == 0 && dt.stride(2) == 0, "native Simple STP requires tied A/dt");
  TORCH_CHECK(state.device() == x.device() && state.device() == dt.device() && state.device() == A.device() && state.device() == B.device() && state.device() == C.device(), "native inputs must share a MUSA device");
  TORCH_CHECK(src.numel() >= 1 && dst.numel() >= 1 && (src.scalar_type() == at::kInt || src.scalar_type() == at::kLong) && src.scalar_type() == dst.scalar_type(), "slot indices must be int32/int64");
  TORCH_CHECK((Dv.scalar_type() == x.scalar_type() || Dv.scalar_type() == at::kFloat) && ((Dv.dim() == 1 && Dv.size(0) == 64) || (Dv.dim() == 2 && Dv.size(0) == 64 && Dv.size(1) == 64)), "D must be [64] or [64,64]");
  TORCH_CHECK(Dv.device() == state.device(), "D must be on the MUSA device");
  auto out = out_opt.has_value() ? *out_opt : at::empty_like(x);
  TORCH_CHECK(out.scalar_type() == x.scalar_type() && out.sizes() == x.sizes(), "out must match x");
  TORCH_CHECK(out.device() == state.device(), "out must be on the MUSA device");
  TORCH_CHECK(!rand_seed.has_value() || (rand_seed->scalar_type() == at::kLong && rand_seed->numel() == 1 && state.scalar_type() == at::kHalf && (philox_rounds == 5 || philox_rounds == 10)), "invalid stochastic-rounding arguments");
  TORCH_CHECK(!dt_bias.has_value() || dt_bias->scalar_type() == at::kFloat, "dt_bias must be fp32");
  TORCH_CHECK(!dt_bias.has_value() || dt_bias->device() == state.device(), "dt_bias must be on the MUSA device");
  TORCH_CHECK(!dt_bias.has_value() ||
                  (dt_bias->dim() == 1 && dt_bias->size(0) == 64) ||
                  (dt_bias->dim() == 2 && dt_bias->size(0) == 64 && dt_bias->size(1) == 64),
              "dt_bias must be [64] or [64,64]");
  TORCH_CHECK(!dt_bias.has_value() || dt_bias->dim() == 1 || dt_bias->stride(1) == 0, "dt_bias must be tied over D");
  TORCH_CHECK(!z.has_value() || z->scalar_type() == x.scalar_type(), "z dtype must match x");
  TORCH_CHECK(!z.has_value() || z->device() == state.device(), "z must be on the MUSA device");
  TORCH_CHECK(src.device() == state.device() && dst.device() == state.device(), "indices must be on the MUSA device");
  TORCH_CHECK(!rand_seed.has_value() || rand_seed->device() == state.device(), "rand_seed must be on the MUSA device");
  const c10::musa::OptionalMUSAGuard guard(device_of(state));
  musaStream_t stream = at::musa::getCurrentMUSAStream().stream();
  const int64_t ss=state.stride(0), sh=state.stride(1), sd=state.stride(2), sn=state.stride(3);
  const int64_t xb=x.stride(0),xh=x.stride(1),xd=x.stride(2),dtb=dt.stride(0),dth=dt.stride(1),dtd=dt.stride(2);
  const int64_t ah=A.stride(0),ad=A.stride(1),an=A.stride(2),bb=B.stride(0),bg=B.stride(1),bn=B.stride(2),cb=C.stride(0),cg=C.stride(1),cn=C.stride(2);
  const int64_t dh=Dv.stride(0),dd=Dv.dim()==2?Dv.stride(1):0,bh=dt_bias.has_value()?dt_bias->stride(0):0,bd=dt_bias.has_value()&&dt_bias->dim()==2?dt_bias->stride(1):0;
  const int64_t ob=out.stride(0),oh=out.stride(1),od=out.stride(2),zb=z.has_value()?z->stride(0):0,zh=z.has_value()?z->stride(1):0,zd=z.has_value()?z->stride(2):0;
  const float* bias_ptr = dt_bias.has_value() ? dt_bias->data_ptr<float>() : nullptr;
  const auto z_tensor = z.has_value() ? *z : x;
  const int64_t* seed_ptr = rand_seed.has_value() ? rand_seed->data_ptr<int64_t>() : nullptr;
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "native Simple STP currently requires BF16 x/output");
#define LAUNCH_ROUND(IDX, R, SR) do { \
  if (Dv.scalar_type() == at::kFloat) \
    LAUNCH(__mt_bfloat16, float, IDX, R, dt_bias.has_value(), dt_bias.has_value() && dt_bias->dim()==2, z.has_value(), dt_softplus, Dv.dim()==1, A.stride(1)==0 && A.stride(2)==0, SR); \
  else \
    LAUNCH(__mt_bfloat16, __mt_bfloat16, IDX, R, dt_bias.has_value(), dt_bias.has_value() && dt_bias->dim()==2, z.has_value(), dt_softplus, Dv.dim()==1, A.stride(1)==0 && A.stride(2)==0, SR); \
} while (0)
  if (src.scalar_type() == at::kInt) {
    if (rand_seed.has_value()) { if (philox_rounds == 5) LAUNCH_ROUND(int, 5, true); else LAUNCH_ROUND(int, 10, true); }
    else LAUNCH_ROUND(int, 10, false);
  } else {
    if (rand_seed.has_value()) { if (philox_rounds == 5) LAUNCH_ROUND(int64_t, 5, true); else LAUNCH_ROUND(int64_t, 10, true); }
    else LAUNCH_ROUND(int64_t, 10, false);
  }
#undef LAUNCH_ROUND
  TORCH_CHECK(musaGetLastError() == musaSuccess, "native Simple STP launch failed");
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("musa_ssu_simple", &musa_ssu_simple, "Native MUSA Simple STP"); }
