/*
 * SM120 FP8 specialization derived from FlashInfer's paged MLA algorithm.
 * Reuses its persistent work format, MLAParams, FP8 MMA, async copies,
 * online-softmax merge, and (parameterized) MLAPlan.
 * FlashInfer portions Copyright (c) 2023-2025 FlashInfer team, Apache-2.0.
 */
#include <algorithm>
#include <flashinfer/attention/mla.cuh>

#include "scheduler_fp8.cuh"

#ifndef TILE_Q
#define TILE_Q 32
#endif
#ifndef TILE_KV
#define TILE_KV 32
#endif
#ifndef STAGES
#define STAGES 1
#endif
#ifndef D_GROUPS
#define D_GROUPS 2
#endif
#ifndef SHARE_P
#define SHARE_P 0
#endif
#ifndef SHARD_QK
#define SHARD_QK 0
#endif

namespace flashinfer {
namespace mla_fp8 {
constexpr int BM = TILE_Q, BN = TILE_KV, NS = STAGES, DG = D_GROUPS;
constexpr bool SP = SHARE_P;
static_assert(!SHARD_QK || (SHARE_P && BN % (16 * DG) == 0));
constexpr int NT = (BM / 16) * DG * 32, OD = 512 / DG, NF = OD / 16;
using Base = MLAParams<__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16, int>;
struct Params {
  Base p;
  const float *qs, *ks;
};
struct MergeTraits {
  using IdType = int;
  using DTypeO = __nv_bfloat16;
  static constexpr int NUM_THREADS = NT, HEAD_DIM_CKV = 512;
};
struct alignas(16) Shared {
  uint8_t qc[BM * 512], qr[BM * 64];
  uint8_t kc[NS][BN * 512], kr[NS][BN * 64];
  float ks[NS][BN];
#if SHARD_QK
  uint8_t prob[BM * 64];
  float qk_max[DG][BM], p_max[DG][BM], p_sum[DG][BM];
#elif SHARE_P
  // One producer group computes QK/softmax and packs P once. All output
  // dimension groups consume the same register-layout fragments after a CTA barrier.
  uint4 pp[BN / 32][BM / 16][32];
  float factor[BM], scale[BM], row_m[BM], row_d[BM];
#endif
};

// Keep FlashInfer's merge metadata and stable state_t algebra, but parallelize
// the split dimension across warps. The original merge reads all splits in
// series for each output vector, which dominates short-query FP8 decode.
__device__ __forceinline__ void merge_parallel(const Base& p, float* scratch) {
  constexpr int NW = NT / 32;
  int cta = blockIdx.y * gridDim.x + blockIdx.x;
  int first = p.merge_packed_offset_start[cta], last = p.merge_packed_offset_end[cta];
  int ps = p.merge_partial_packed_offset_start[cta], pe = p.merge_partial_packed_offset_end[cta];
  int stride = p.merge_partial_stride[cta], warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  float* lse_s = scratch + NW * 512;
  for (int row = 0; row < last - first; ++row) {
    state_t<16> st;
    for (int s = ps + row + warp * stride; s < pe; s += NW * stride) {
      vec_t<float, 16> v;
      v.cast_load(p.partial_o + (int64_t)s * 512 + lane * 16);
      st.merge(v, p.partial_lse[s], 1);
    }
    st.normalize();
    st.o.store(scratch + warp * 512 + lane * 16);
    if (lane == 0) lse_s[warp] = st.get_lse();
    __syncthreads();
    float mx = -INFINITY;
#pragma unroll
    for (int w = 0; w < NW; ++w) mx = fmaxf(mx, lse_s[w]);
    float weights[NW], den = 0;
#pragma unroll
    for (int w = 0; w < NW; ++w) {
      weights[w] = exp2f(lse_s[w] - mx);
      den += weights[w];
    }
    for (int d = threadIdx.x; d < 512; d += NT) {
      float v = 0;
#pragma unroll
      for (int w = 0; w < NW; ++w) v += weights[w] * scratch[w * 512 + d];
      p.final_o[(int64_t)(first + row) * 512 + d] = __float2bfloat16(v / den);
    }
    if (threadIdx.x == 0 && p.final_lse) {
      float lse = mx + log2f(den);
      p.final_lse[first + row] = p.return_lse_base_on_e ? lse * math::loge2 : lse;
    }
    __syncthreads();
  }
}

template <int D>
__device__ __forceinline__ uint8_t* at(uint8_t* base, int row, int col) {
  constexpr auto mode = D == 64 ? SwizzleMode::k64B : SwizzleMode::k128B;
  return base + 16 * get_permuted_offset<mode, D / 16>(row, col / 16) + (col % 16);
}

__device__ __forceinline__ void copy16(uint8_t* dst, const uint8_t* src, bool valid) {
  // FlashInfer's zero-filling cp.async; predicates never dereference invalid rows.
  cp_async::pred_load<128, PrefetchMode::kNoPrefetch, SharedMemFillMode::kFillZero>(dst, src,
                                                                                    valid);
}

__device__ __forceinline__ void load_q(Shared& s, const Base& p, int qbegin, int packed, int qlen) {
  for (int i = threadIdx.x; i < BM * 36; i += NT) {
    int row = i / 36, c = i % 36, q, h;
    uint32_t uq, uh;
    p.num_heads.divmod(packed + row, uq, uh);
    q = uq;
    h = uh;
    bool valid = q < qlen;
    const uint8_t* src = reinterpret_cast<const uint8_t*>(c < 32 ? p.q_nope : p.q_pe);
    src += (valid ? (qbegin + q) * p.q_nope_stride_n + h * p.q_nope_stride_h : 0) +
           (c < 32 ? c : c - 32) * 16;
    copy16(c < 32 ? at<512>(s.qc, row, c * 16) : at<64>(s.qr, row, (c - 32) * 16), src, valid);
  }
}

__device__ __forceinline__ void load_kv(Shared& s, const Params& a, int slot, int begin, int end,
                                        int pagebegin) {
  const auto& p = a.p;
  for (int i = threadIdx.x; i < BN * 36; i += NT) {
    int row = i / 36, c = i % 36, n = begin + row;
    bool valid = n < end;
    uint32_t pg, off;
    p.block_size.divmod(n, pg, off);
    int phys = valid ? p.kv_indices[pagebegin + pg] : 0;
    const uint8_t* src = reinterpret_cast<const uint8_t*>(c < 32 ? p.ckv : p.kpe);
    src += (valid ? (int64_t)phys * p.ckv_stride_page + off * p.ckv_stride_n : 0) +
           (c < 32 ? c : c - 32) * 16;
    copy16(c < 32 ? at<512>(s.kc[slot], row, c * 16) : at<64>(s.kr[slot], row, (c - 32) * 16), src,
           valid);
    if (c == 0) s.ks[slot][row] = valid ? a.ks[(int64_t)phys * (uint32_t)p.block_size + off] : 1.f;
  }
}

template <int D, int N>
__device__ __forceinline__ void qk(uint8_t* q, uint8_t* k, int wq, float (&scores)[N][8], int first_n = 0) {
  int lane = threadIdx.x % 32;
#pragma unroll
  for (int d = 0; d < D; d += 32) {
    uint32_t qa[4];
    mma::ldmatrix_m8n8x4(qa, at<D>(q, wq * 16 + lane % 16, d + 16 * (lane / 16)));
#pragma unroll
    for (int n = 0; n < N; ++n) {
      uint32_t kb[4];
      mma::ldmatrix_m8n8x4(
          kb, at<D>(k, (n + first_n) * 16 + 8 * (lane / 16) + lane % 8, d + 16 * ((lane % 16) / 8)));
      mma::mma_sync_m16n16k32_row_col_f8f8f32<__nv_fp8_e4m3>(scores[n], qa, kb);
    }
  }
}

__device__ __forceinline__ float rowmax(float x) {
  x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, 1));
  return fmaxf(x, __shfl_xor_sync(0xffffffff, x, 2));
}
__device__ __forceinline__ float rowsum(float x) {
  x += __shfl_xor_sync(0xffffffff, x, 1);
  return x + __shfl_xor_sync(0xffffffff, x, 2);
}

// Repack two score fragments (16x16 each) into one FP8 16x32 A fragment.
// Scores distribute two adjacent columns/lane; FP8 A needs four/lane.
__device__ __forceinline__ void pack_p(const float* lo, const float* hi, const float (&scale)[2],
                                       uint32_t (&a)[4]) {
  int lane = threadIdx.x % 32, src = (lane & ~3) + 2 * (lane % 2);
#pragma unroll
  for (int block = 0; block < 2; ++block) {
    const float* p = block ? hi : lo;
#pragma unroll
    for (int row = 0; row < 2; ++row) {
      __nv_fp8x2_e4m3 l(make_float2(p[row * 2] / scale[row], p[row * 2 + 1] / scale[row]));
      __nv_fp8x2_e4m3 h(make_float2(p[row * 2 + 4] / scale[row], p[row * 2 + 5] / scale[row]));
      uint32_t ll = __shfl_sync(0xffffffff, (uint32_t)l.__x, src);
      uint32_t lh = __shfl_sync(0xffffffff, (uint32_t)l.__x, src + 1);
      uint32_t hl = __shfl_sync(0xffffffff, (uint32_t)h.__x, src);
      uint32_t hh = __shfl_sync(0xffffffff, (uint32_t)h.__x, src + 1);
      a[block * 2 + row] = (lane % 4 < 2) ? (ll | (lh << 16)) : (hl | (hh << 16));
    }
  }
}

template <int N>
__device__ __forceinline__ void pv(Shared& sm, int slot, int wq, int dg, float (&p)[N][8],
                                   const float (&scale)[2], float (&o)[NF][8]) {
  int lane = threadIdx.x % 32;
#pragma unroll
  for (int n = 0; n < BN; n += 32) {
    uint32_t a[4];
#if SHARD_QK
    mma::ldmatrix_m8n8x4(a, at<64>(sm.prob, wq * 16 + lane % 16, n + 16 * (lane / 16)));
#elif SHARE_P
    uint4 packed = sm.pp[n / 32][wq][lane];
    a[0] = packed.x;
    a[1] = packed.y;
    a[2] = packed.z;
    a[3] = packed.w;
#else
    pack_p(p[n / 16], p[n / 16 + 1], scale, a);
#endif
#pragma unroll
    for (int d = 0; d < NF; ++d) {
      uint32_t t[4], b[4];
      uint32_t addr = __cvta_generic_to_shared(at<512>(sm.kc[slot], n + lane, dg * OD + d * 16));
      // SM120 byte transpose. A b16 transpose would mix FP8 element pairs.
      asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared::cta.b8 {%0,%1,%2,%3},[%4];"
                   : "=r"(t[0]), "=r"(t[1]), "=r"(t[2]), "=r"(t[3])
                   : "r"(addr));
      b[0] = t[0];
      b[1] = t[2];
      b[2] = t[1];
      b[3] = t[3];
      mma::mma_sync_m16n16k32_row_col_f8f8f32<__nv_fp8_e4m3>(o[d], a, b);
    }
  }
}

template <bool CAUSAL, bool FUSED>
__global__ __launch_bounds__(NT) void BatchMLAPagedAttentionFP8SM120(
    const __grid_constant__ Params a) {
  extern __shared__ __align__(16) uint8_t storage[];
  Shared& sm = *reinterpret_cast<Shared*>(storage);
  const auto& p = a.p;
  int lane = threadIdx.x % 32, wq = (threadIdx.x / 32) % (BM / 16),
      dg = (threadIdx.x / 32) / (BM / 16);
  for (int work = p.work_indptr[blockIdx.y]; work < p.work_indptr[blockIdx.y + 1]; ++work) {
    int qb = p.q_indptr[work], ql = p.q_len[work], kl = p.kv_len[work];
    int packed = p.q_start[work] + blockIdx.x * BM;
    if (packed >= ql * (uint32_t)p.num_heads) continue;
    int begin = p.kv_start[work], end = p.kv_end[work], partial = p.partial_indptr[work];
    if (CAUSAL) end = min(end, kl - ql + int(uint32_t(packed + BM - 1) / p.num_heads) + 1);
    float out[NF][8] = {}, m[2] = {-INFINITY, -INFINITY}, den[2] = {}, oldscale[2] = {1, 1}, qs[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      int r = packed + wq * 16 + lane / 4 + 8 * j;
      qs[j] = r < ql * (uint32_t)p.num_heads ? a.qs[qb * (uint32_t)p.num_heads + r] : 1.f;
    }
    __syncthreads();
    load_q(sm, p, qb, packed, ql);
#pragma unroll
    for (int s = 0; s < NS; ++s) {
      load_kv(sm, a, s, begin + s * BN, end, p.kv_indptr[work]);
      cp_async::commit_group();
    }
    for (int base = begin, iter = 0; base < end; base += BN, ++iter) {
      cp_async::wait_group<NS - 1>();
      __syncthreads();
      int slot = iter % NS;
#if SHARD_QK
      constexpr int NC = BN / (16 * DG);
      float scores[NC][8] = {}, sp[2], factors[2];
      qk<64>(sm.qr, sm.kr[slot], wq, scores, dg * NC);
      qk<512>(sm.qc, sm.kc[slot], wq, scores, dg * NC);
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        int local = wq * 16 + lane / 4 + 8 * j;
        int qidx = uint32_t(packed + local) / p.num_heads;
        float mx = -INFINITY;
#pragma unroll
        for (int n = 0; n < NC; ++n) {
#pragma unroll
          for (int col = 0; col < 4; ++col) {
            int reg = j * 2 + (col / 2) * 4 + col % 2;
            int k = (dg * NC + n) * 16 + 2 * (lane % 4) + 8 * (col / 2) + col % 2;
            float x = scores[n][reg] * (qs[j] * sm.ks[slot][k] * p.sm_scale * math::log2e);
            bool valid = base + k < end && (!CAUSAL || base + k <= kl - ql + qidx);
            scores[n][reg] = valid ? x : -INFINITY;
            mx = fmaxf(mx, scores[n][reg]);
          }
        }
        mx = rowmax(mx);
        if (lane % 4 == 0) sm.qk_max[dg][local] = mx;
      }
      __syncthreads();
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        int local = wq * 16 + lane / 4 + 8 * j;
        float mx = m[j];
#pragma unroll
        for (int g = 0; g < DG; ++g) mx = fmaxf(mx, sm.qk_max[g][local]);
        factors[j] = isfinite(m[j]) ? exp2f(m[j] - mx) : 0.f;
        m[j] = mx;
        float sum = 0.f, pm = 0.f;
#pragma unroll
        for (int n = 0; n < NC; ++n) {
#pragma unroll
          for (int col = 0; col < 4; ++col) {
            int reg = j * 2 + (col / 2) * 4 + col % 2;
            int k = (dg * NC + n) * 16 + 2 * (lane % 4) + 8 * (col / 2) + col % 2;
            float v = isfinite(mx) ? exp2f(scores[n][reg] - mx) : 0.f;
            sum += v;
            v *= sm.ks[slot][k];
            scores[n][reg] = v;
            pm = fmaxf(pm, v);
          }
        }
        sum = rowsum(sum);
        pm = rowmax(pm);
        if (lane % 4 == 0) {
          sm.p_sum[dg][local] = sum;
          sm.p_max[dg][local] = pm;
        }
      }
      __syncthreads();
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        int local = wq * 16 + lane / 4 + 8 * j;
        float sum = 0.f, pm = 0.f;
#pragma unroll
        for (int g = 0; g < DG; ++g) {
          sum += sm.p_sum[g][local];
          pm = fmaxf(pm, sm.p_max[g][local]);
        }
        den[j] = den[j] * factors[j] + sum;
        sp[j] = fmaxf(pm, 1e-12f) / 448.f;
        factors[j] *= oldscale[j] / sp[j];
        oldscale[j] = sp[j];
#pragma unroll
        for (int n = 0; n < NC; ++n) {
#pragma unroll
          for (int col = 0; col < 2; ++col) {
            int reg = j * 2 + col * 4;
            int k = (dg * NC + n) * 16 + 2 * (lane % 4) + 8 * col;
            __nv_fp8x2_e4m3 pair(make_float2(scores[n][reg] / sp[j], scores[n][reg + 1] / sp[j]));
            *reinterpret_cast<uint16_t*>(at<64>(sm.prob, local, k)) = pair.__x;
          }
        }
      }
      __syncthreads();
#pragma unroll
      for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int d = 0; d < NF; ++d) {
          out[d][j * 2] *= factors[j];
          out[d][j * 2 + 1] *= factors[j];
          out[d][j * 2 + 4] *= factors[j];
          out[d][j * 2 + 5] *= factors[j];
        }
      }
#else
      float scores[BN / 16][8] = {};
      float sp[2], factors[2];
      if (!SP || dg == 0) {
        qk<64>(sm.qr, sm.kr[slot], wq, scores);
        qk<512>(sm.qc, sm.kc[slot], wq, scores);

#pragma unroll
        for (int j = 0; j < 2; ++j) {
          int row = packed + wq * 16 + lane / 4 + 8 * j;
          int qidx = uint32_t(row) / p.num_heads;
          float mx = -INFINITY;
#pragma unroll
          for (int n = 0; n < BN / 16; ++n) {
#pragma unroll
            for (int col = 0; col < 4; ++col) {
              int reg = j * 2 + (col / 2) * 4 + col % 2;
              int k = n * 16 + 2 * (lane % 4) + 8 * (col / 2) + col % 2;
              float x = scores[n][reg] * (qs[j] * sm.ks[slot][k] * p.sm_scale * math::log2e);
              bool valid = base + k < end && (!CAUSAL || base + k <= kl - ql + qidx);
              scores[n][reg] = valid ? x : -INFINITY;
              mx = fmaxf(mx, scores[n][reg]);
            }
          }
          mx = fmaxf(m[j], rowmax(mx));
          float alpha = isfinite(m[j]) ? exp2f(m[j] - mx) : 0.f;
          float sum = 0.f, pm = 0.f;
#pragma unroll
          for (int n = 0; n < BN / 16; ++n) {
#pragma unroll
            for (int col = 0; col < 4; ++col) {
              int reg = j * 2 + (col / 2) * 4 + col % 2;
              int k = n * 16 + 2 * (lane % 4) + 8 * (col / 2) + col % 2;
              float v = isfinite(mx) ? exp2f(scores[n][reg] - mx) : 0.f;
              sum += v;
              v *= sm.ks[slot][k];
              scores[n][reg] = v;
              pm = fmaxf(pm, v);
            }
          }
          den[j] = den[j] * alpha + rowsum(sum);
          sp[j] = fmaxf(rowmax(pm), 1e-12f) / 448.f;
          float rescale = alpha * (oldscale[j] / sp[j]);
#if SHARE_P
          factors[j] = rescale;
#else
#pragma unroll
          for (int d = 0; d < NF; ++d) {
            out[d][j * 2] *= rescale;
            out[d][j * 2 + 1] *= rescale;
            out[d][j * 2 + 4] *= rescale;
            out[d][j * 2 + 5] *= rescale;
          }
#endif
          oldscale[j] = sp[j];
          m[j] = mx;
#if SHARE_P
          if (lane % 4 == 0) {
            int local = wq * 16 + lane / 4 + 8 * j;
            sm.factor[local] = factors[j];
            sm.scale[local] = sp[j];
            sm.row_m[local] = m[j];
            sm.row_d[local] = den[j];
          }
#endif
        }
#if SHARE_P
#pragma unroll
        for (int n = 0; n < BN; n += 32) {
          uint32_t packed_p[4];
          pack_p(scores[n / 16], scores[n / 16 + 1], sp, packed_p);
          sm.pp[n / 32][wq][lane] = make_uint4(packed_p[0], packed_p[1], packed_p[2], packed_p[3]);
        }
#endif
      }
#if SHARE_P
      __syncthreads();
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        int local = wq * 16 + lane / 4 + 8 * j;
        factors[j] = sm.factor[local];
        sp[j] = sm.scale[local];
        m[j] = sm.row_m[local];
        den[j] = sm.row_d[local];
        oldscale[j] = sp[j];
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int d = 0; d < NF; ++d) {
          out[d][j * 2] *= factors[j];
          out[d][j * 2 + 1] *= factors[j];
          out[d][j * 2 + 4] *= factors[j];
          out[d][j * 2 + 5] *= factors[j];
        }
      }
#endif
#endif
      pv(sm, slot, wq, dg, scores, sp, out);
      __syncthreads();
      load_kv(sm, a, slot, base + NS * BN, end, p.kv_indptr[work]);
      cp_async::commit_group();
    }
    cp_async::wait_group<0>();
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      int local = wq * 16 + lane / 4 + 8 * j, row = packed + local;
      if (row < ql * (uint32_t)p.num_heads) {
        int64_t dest =
            partial < 0 ? qb * (uint32_t)p.num_heads + row : partial + blockIdx.x * BM + local;
        auto* ptr = partial < 0 ? p.final_o : p.partial_o;
        float norm = den[j] > 0 ? oldscale[j] / den[j] : 0.f;
#pragma unroll
        for (int d = 0; d < NF; ++d) {
#pragma unroll
          for (int c = 0; c < 2; ++c) {
            int reg = j * 2 + c * 4, dim = dg * OD + d * 16 + 2 * (lane % 4) + 8 * c;
            __nv_bfloat162 v = __floats2bfloat162_rn(out[d][reg] * norm, out[d][reg + 1] * norm);
            *reinterpret_cast<__nv_bfloat162*>(ptr + dest * 512 + dim) = v;
          }
        }
        if (dg == 0 && lane % 4 == 0) {
          float lse = den[j] > 0 ? m[j] + log2f(den[j]) : -INFINITY;
          if (partial >= 0)
            p.partial_lse[dest] = lse;
          else if (p.final_lse)
            p.final_lse[dest] = p.return_lse_base_on_e ? lse * math::loge2 : lse;
        }
      }
    }
  }
  if constexpr (FUSED) {
    cooperative_groups::this_grid().sync();
    merge_parallel(p, reinterpret_cast<float*>(storage));
  }
}

__global__ void merge_kernel(Base p) {
  extern __shared__ float scratch[];
  merge_parallel(p, scratch);
}
}  // namespace mla_fp8
}  // namespace flashinfer

using namespace flashinfer;
using namespace flashinfer::mla_fp8;

extern "C" int fp8_plan(void* fw, size_t fbytes, void* iw, void* hostiw, size_t ibytes, int* qo,
                        int* ki, int* kl, int batch, int heads, int causal, int workers,
                        int64_t* info, cudaStream_t stream) {
  try {
    MLAPlanInfo plan;
    auto err = MLAPlanSM120Fp8(fw, fbytes, iw, hostiw, ibytes, plan, qo, ki, kl, batch, heads, 512,
                               causal, BM, BN, workers, stream);
    if (err != cudaSuccess) return err;
    auto v = plan.ToVector();
    std::copy(v.begin(), v.end(), info);
    return 0;
  } catch (const std::exception&) {
    return cudaErrorInvalidValue;
  }
}

extern "C" int fp8_run(void* fw, void* iw, int64_t* info, void* q, void* kv, float* qs, float* ks,
                       int* indices, void* out, float* lse, int heads, int page, int causal,
                       float scale, int fused, cudaStream_t stream) {
  MLAPlanInfo plan;
  plan.FromVector(std::vector<int64_t>(info, info + 18));
  Params a{};
  auto& p = a.p;
  a.qs = qs;
  a.ks = ks;
  p.q_nope = (__nv_fp8_e4m3*)q;
  p.q_pe = p.q_nope + 512;
  p.ckv = (__nv_fp8_e4m3*)kv;
  p.kpe = p.ckv + 512;
  p.final_o = (__nv_bfloat16*)out;
  p.final_lse = lse;
  p.partial_o = GetPtrFromBaseOffset<__nv_bfloat16>(fw, plan.partial_o_offset);
  p.partial_lse = GetPtrFromBaseOffset<float>(fw, plan.partial_lse_offset);
#define PTR(name) p.name = GetPtrFromBaseOffset<int>(iw, plan.name##_offset)
  PTR(q_indptr);
  PTR(kv_indptr);
  PTR(partial_indptr);
  PTR(q_len);
  PTR(kv_len);
  PTR(q_start);
  PTR(kv_start);
  PTR(kv_end);
  PTR(work_indptr);
  PTR(merge_packed_offset_start);
  PTR(merge_packed_offset_end);
  PTR(merge_partial_packed_offset_start);
  PTR(merge_partial_packed_offset_end);
  PTR(merge_partial_stride);
#undef PTR
  p.kv_indices = indices;
  p.num_heads = uint_fastdiv(heads);
  p.block_size = uint_fastdiv(page);
  p.q_nope_stride_n = p.q_pe_stride_n = heads * 576;
  p.q_nope_stride_h = p.q_pe_stride_h = 576;
  p.ckv_stride_n = p.kpe_stride_n = 576;
  p.ckv_stride_page = p.kpe_stride_page = page * 576;
  p.o_stride_n = heads * 512;
  p.o_stride_h = 512;
  p.sm_scale = scale;
  p.return_lse_base_on_e = true;
  auto kernel = causal ? (fused ? BatchMLAPagedAttentionFP8SM120<true, true>
                                : BatchMLAPagedAttentionFP8SM120<true, false>)
                       : (fused ? BatchMLAPagedAttentionFP8SM120<false, true>
                                : BatchMLAPagedAttentionFP8SM120<false, false>);
  cudaError_t err =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Shared));
  if (err != cudaSuccess) return err;
  dim3 grid(plan.num_blks_x, plan.num_blks_y);
  if (fused) {
    void* args[] = {&a};
    err = cudaLaunchCooperativeKernel((void*)kernel, grid, NT, args, sizeof(Shared), stream);
  } else {
    kernel<<<grid, NT, sizeof(Shared), stream>>>(a);
    err = cudaGetLastError();
    if (err == cudaSuccess)
      merge_kernel<<<grid, NT, (NT / 32) * (512 + 1) * sizeof(float), stream>>>(p);
  }
  return err == cudaSuccess ? cudaGetLastError() : err;
}

extern "C" int fp8_attributes(int causal, int fused, int* values) {
  auto kernel = causal ? (fused ? BatchMLAPagedAttentionFP8SM120<true, true>
                                : BatchMLAPagedAttentionFP8SM120<true, false>)
                       : (fused ? BatchMLAPagedAttentionFP8SM120<false, true>
                                : BatchMLAPagedAttentionFP8SM120<false, false>);
  cudaFuncAttributes a{};
  int active = 0;
  auto err =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Shared));
  if (err != cudaSuccess) return err;
  err = cudaFuncGetAttributes(&a, kernel);
  if (err != cudaSuccess) return err;
  err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, NT, sizeof(Shared));
  values[0] = a.numRegs;
  values[1] = sizeof(Shared);
  values[2] = active;
  values[3] = NT;
  values[4] = a.localSizeBytes;
  return err;
}
