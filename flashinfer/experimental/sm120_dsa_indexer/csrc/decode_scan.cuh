/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Decode logits scan: every row's scores into logits[num_rows][logits_stride]
// for the selection kernel that follows (decode_select_long.cuh).
//
// A unit is kNextN rows of one request, scored together per key tile. The
// grid is one wave; each CTA streams its pieces (a unit's tile range) back to
// back, and the producer warp loads the next piece's queries into the other
// query buffer while the math warps score the current one.
#pragma once

#include "decode_common.cuh"

namespace flashinfer::sm120_dsa_indexer {

constexpr int kScanMaxStages = 6;

struct ScanParams {
  const int32_t* row_len;      // [num_rows]: row r sees keys [0, row_len[r])
  const int32_t* block_table;  // [requests, bt_stride]
  int bt_stride;
  int page_rows;
  int num_rows;
  int units_per_req;
  int max_tiles;  // the capture's bound on a row's key tiles
  int k_stages;
  float* logits;
  int64_t logits_stride;
  uint32_t* long_ws;  // kLong: [num_rows][kLongWords]
};

template <int kNumHeads, bool kFp4, int kNextN, bool kLong = false>
struct ScanSmem {
  static constexpr int kRow = Fmt<kFp4>::kRow;
  static constexpr int kQBytes = kNextN * kNumHeads * kRow;
  static constexpr int kKBytes = kBlockKV * kRow;
  static constexpr int kWBytes = kNextN * kNumHeads * 4;
  __host__ __device__ static int q_off(int b) { return b * kQBytes; }
  __host__ __device__ static int k_off(int st) { return 2 * kQBytes + st * kKBytes; }
  __host__ __device__ static int scale_off(int stages, int st) {
    return k_off(stages) + st * kBlockKV * 4;
  }
  __host__ __device__ static int w_off(int stages, int b) {
    return scale_off(stages, stages) + b * kWBytes;
  }
  __host__ __device__ static int qsf_off(int stages, int b) {
    return w_off(stages, 2) + (kFp4 ? b * kWBytes : 0);
  }
  __host__ __device__ static int bars_off(int stages) { return (qsf_off(stages, 2) + 127) & ~127; }
  // kLong: the unit's rows' histograms (two 16-bit counts a word), then the
  // rows' epoch parities.
  __host__ __device__ static int hist_off(int stages) {
    return (bars_off(stages) + 8 * (4 + 2 * stages) + 15) & ~15;
  }
  __host__ __device__ static int bytes(int stages) {
    return hist_off(stages) + (kLong ? kNextN * 2048 * 4 + 16 : 0);
  }
};

// One producer warp (TMA) and 256 math threads. setmaxnreg moves the
// producer warpgroup's registers to the math warps (40 / 232 around the
// launch's 168), so both rows' query fragments stay in registers at H = 64.
//
// kLong: the long rows' histograms are counted here, in shared memory, and
// added to long_ws at the end of each piece; this saves the selection a pass
// over the logits. kKeep: the logits are stored with an L2 evict_last policy
// and dropped by the selection after reading, so they are neither evicted by
// the streamed keys nor written back.
template <int kNumHeads, bool kFp4, int kNextN, bool kLong, bool kKeep = false>
__global__ void __launch_bounds__(kNumWsThreads, 1)
    logits_scan_kernel(const __grid_constant__ ScanParams p,
                       const __grid_constant__ CUtensorMap tmap_q,
                       const __grid_constant__ CUtensorMap tmap_w,
                       const __grid_constant__ CUtensorMap tmap_k,
                       const __grid_constant__ CUtensorMap tmap_ks,
                       const __grid_constant__ CUtensorMap tmap_qsf) {
  using L = ScanSmem<kNumHeads, kFp4, kNextN, kLong>;
  using F = Fmt<kFp4>;
  constexpr int kKSteps = F::kKSteps;
  constexpr int kNTiles = kNumHeads / 8;
  const int stages = p.k_stages;

  extern __shared__ __align__(1024) uint8_t smem[];
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + L::bars_off(stages));
  uint64_t* full_q = bars;
  uint64_t* empty_q = bars + 2;
  uint64_t* full_k = bars + 4;
  uint64_t* empty_k = bars + 4 + stages;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const bool producer = warp == kNumMathThreads / 32;
  if (producer && lane == 0) {
    prefetch_tmap(&tmap_q);
    prefetch_tmap(&tmap_w);
    prefetch_tmap(&tmap_k);
    prefetch_tmap(&tmap_ks);
    if constexpr (kFp4) prefetch_tmap(&tmap_qsf);
    for (int b = 0; b < 2; ++b) {
      mbar_init(full_q + b, 1);
      mbar_init(empty_q + b, kNumMathThreads);
    }
    for (int i = 0; i < stages; ++i) {
      mbar_init(full_k + i, 1);
      mbar_init(empty_k + i, kNumMathThreads);
    }
    fence_barrier_init();
  }
  __syncthreads();
  asm volatile(".pragma \"global knob CtaReconfigOptInSeparateComp=1\";\n");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;");

  // The split: every request spans M virtual tiles (M from the capture's
  // bound, so the split needs no length read), spread evenly over slots of
  // upr CTAs, one CTA per unit of the request. A request's units thus stream
  // the same key tiles at the same time: the keys come from DRAM once and
  // from L2 for the other units.
  const int cta = static_cast<int>(blockIdx.x), G = static_cast<int>(gridDim.x);
  const int M = p.max_tiles;
  const int units = p.num_rows / kNextN;
  const int upr = p.units_per_req > 1 && G >= p.units_per_req ? p.units_per_req : 1;
  const int slots = G / upr, slot = cta / upr, uk = cta - slot * upr;
  const uint32_t V = static_cast<uint32_t>(units / upr) * M, uS = slots;
  const int v_lo = slot < slots ? static_cast<int>(V * slot / uS) : 0;
  const int v_hi = slot < slots ? static_cast<int>(V * (slot + 1) / uS) : 0;
  // A piece: request rq's unit uk, tiles [t0, t1). Both roles walk the same
  // pieces.
  auto piece = [&](int rq, int& t0, int& t1, int(&len)[2]) {
    const int u = rq * upr + uk;
    int kend = 0;
#pragma unroll
    for (int qi = 0; qi < kNextN; ++qi) {
      len[qi] = __ldg(p.row_len + u * kNextN + qi);
      kend = max(kend, len[qi]);
    }
    const int ntiles = (kend + kBlockKV - 1) / kBlockKV;
    const int vs = max(rq * M, v_lo), ve = min((rq + 1) * M, v_hi);
    t0 = static_cast<int>(static_cast<uint32_t>(vs - rq * M) * ntiles / static_cast<uint32_t>(M));
    t1 = static_cast<int>(static_cast<uint32_t>(ve - rq * M) * ntiles / static_cast<uint32_t>(M));
    return kend;
  };
  const int rq_lo = v_lo / M, rq_hi = v_hi > v_lo ? (v_hi - 1) / M + 1 : rq_lo;

  if (tid >= kNumMathThreads) {
    setmaxnreg_dec<40>();
    if (!producer) return;
    const int box = p.page_rows < kBlockKV ? p.page_rows : kBlockKV;
    const int box_shift = __ffs(box) - 1, page_shift = __ffs(p.page_rows) - 1,
              page_mask = p.page_rows - 1;
    const int boxes_per_tile = kBlockKV >> box_shift;
    int st = 0, n = 0;
    uint32_t parity = 0u;
    for (int rq = rq_lo; rq < rq_hi; ++rq, ++n) {
      const int u = rq * upr + uk;
      int t0, t1, len[2];
      const int kend = piece(rq, t0, t1, len);
      const int qb = n & 1;
      if (lane == 0) {
        if (n >= 2) mbar_wait(empty_q + qb, static_cast<uint32_t>(((n - 2) >> 1) & 1));
        mbar_expect_tx(full_q + qb, L::kQBytes + L::kWBytes + (kFp4 ? L::kWBytes : 0));
        const int row0 = u * kNextN;
        tma_load_2d(smem + L::q_off(qb), &tmap_q, full_q + qb, 0, row0 * kNumHeads);
        tma_load_2d(smem + L::w_off(stages, qb), &tmap_w, full_q + qb, 0, row0);
        if constexpr (kFp4)
          tma_load_2d(smem + L::qsf_off(stages, qb), &tmap_qsf, full_q + qb, 0, row0);
      }
      const int bt_req = p.units_per_req == 1 ? u : u / p.units_per_req;
      const int64_t bt_row = static_cast<int64_t>(bt_req) * p.bt_stride;
      const int box0 = t0 * boxes_per_tile, box1 = t1 * boxes_per_tile;
      // The lanes hold the next 64 boxes' pages (two block-table loads in
      // flight); lane 0 issues the TMA loads.
      auto page_of = [&](int b) -> int {
        const int key = b << box_shift;
        return b < box1 && key < kend ? __ldg(p.block_table + bt_row + (key >> page_shift)) : 0;
      };
      int win = box0;
      int cur = page_of(win + lane), nxt = page_of(win + 32 + lane);
      for (int i = 0; i < t1 - t0; ++i) {
        if (lane == 0) {
          mbar_wait(empty_k + st, parity ^ 1u);
          mbar_expect_tx(full_k + st, L::kKBytes + kBlockKV * 4);
        }
        const int key0 = (t0 + i) * kBlockKV;
        for (int r = 0; r < kBlockKV; r += box) {
          const int b = box0 + i * boxes_per_tile + (r >> box_shift);
          if (b - win == 32) {
            win += 32;
            cur = nxt;
            nxt = page_of(win + 32 + lane);
          }
          const int page = __shfl_sync(0xffffffffu, cur, b - win);
          const int key = key0 + r;
          if (lane == 0) {
            tma_load_3d(smem + L::k_off(st) + r * F::kRow, &tmap_k, full_k + st, 0, key & page_mask,
                        page);
            tma_load_2d(smem + L::scale_off(stages, st) + r * 4, &tmap_ks, full_k + st,
                        key & page_mask, page);
          }
        }
        __syncwarp();
        if (++st == stages) {
          st = 0;
          parity ^= 1u;
        }
      }
    }
    return;
  }

  setmaxnreg_inc<232>();
  const int g = lane >> 2, t4 = lane & 3;
  const int a_row = warp * 16 + (lane & 15);
  const int a_chunk_hi = lane >> 4;
  const int b_row = lane & 7;
  const int b_chunk_hi = (lane >> 3) & 1;
  uint32_t* shh = reinterpret_cast<uint32_t*>(smem + L::hist_off(stages));
  uint32_t* shpar = shh + kNextN * 2048;
  if constexpr (kLong) {
    for (int w = tid; w < kNextN * 2048; w += kNumMathThreads) shh[w] = 0u;
    math_barrier();
  }
  // Adds the unit's counts to its long rows' histograms (fp32 counts: one
  // v4 RED per four bins) and clears them.
  auto flush = [&](int u) {
    math_barrier();
    for (int w = tid * 2; w < kNextN * 2048; w += kNumMathThreads * 2) {
      const uint2 v = *reinterpret_cast<const uint2*>(shh + w);
      if (v.x | v.y) {
        const int qi = w >> 11, b = (w & 2047) * 2;
        uint32_t* ws = p.long_ws + static_cast<int64_t>(u * kNextN + qi) * kLongWords;
        red_add_v4_f32_gpu(reinterpret_cast<float*>(ws + kLongHist0 + shpar[qi] * 4096 + b),
                           static_cast<float>(v.x & 0xffffu), static_cast<float>(v.x >> 16),
                           static_cast<float>(v.y & 0xffffu), static_cast<float>(v.y >> 16));
        *reinterpret_cast<uint2*>(shh + w) = make_uint2(0u, 0u);
      }
    }
    math_barrier();
  };
  int st = 0, n = 0;
  uint32_t parity = 0u;
  for (int rq = rq_lo; rq < rq_hi; ++rq, ++n) {
    const int u = rq * upr + uk;
    int t0, t1, len[2];
    piece(rq, t0, t1, len);
    bool lrow[2] = {false, false};
    if constexpr (kLong) {
#pragma unroll
      for (int qi = 0; qi < kNextN; ++qi) {
        lrow[qi] = len[qi] > kRowSelKeys;
        if (tid == qi && lrow[qi])
          shpar[qi] = __ldcg(p.long_ws + static_cast<int64_t>(u * kNextN + qi) * kLongWords +
                             kLongCtr + kCtrEpoch) &
                      1u;
      }
    }
    const int qb = n & 1;
    mbar_wait(full_q + qb, static_cast<uint32_t>((n >> 1) & 1));
    const uint8_t* sq = smem + L::q_off(qb);
    const float* sw = reinterpret_cast<const float*>(smem + L::w_off(stages, qb));
    const uint32_t* sqsf = reinterpret_cast<const uint32_t*>(smem + L::qsf_off(stages, qb));
    uint32_t qf[kNextN][kKSteps][kNTiles][2];
    float wf[kNextN][kNTiles][2];
    uint32_t qsf_r[kNextN][kNTiles];
#pragma unroll
    for (int qi = 0; qi < kNextN; ++qi) {
#pragma unroll
      for (int ks = 0; ks < kKSteps; ++ks)
#pragma unroll
        for (int nt = 0; nt < kNTiles; ++nt)
          ldmatrix_x2(qf[qi][ks][nt],
                      sq + F::swz(qi * kNumHeads + nt * 8 + b_row, ks * 2 + b_chunk_hi));
#pragma unroll
      for (int nt = 0; nt < kNTiles; ++nt) {
        wf[qi][nt][0] = sw[qi * kNumHeads + nt * 8 + t4 * 2];
        wf[qi][nt][1] = sw[qi * kNumHeads + nt * 8 + t4 * 2 + 1];
        qsf_r[qi][nt] = kFp4 ? sqsf[qi * kNumHeads + nt * 8 + g] : 0u;
      }
    }
    mbar_arrive(empty_q + qb);
    float* lg = p.logits + static_cast<int64_t>(u * kNextN) * p.logits_stride;
    uint64_t lg_pol = 0;
    if constexpr (kKeep)
      asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(lg_pol));
    // Scalars: the arrays, captured by reference, would be in local memory.
    const int len0 = len[0], len1 = kNextN > 1 ? len[1] : 0;
    const bool lrow0 = lrow[0], lrow1 = kNextN > 1 && lrow[1];
    // kH: the variant that counts histograms, only for units with a long row,
    // so the plain loop keeps its code and registers.
    auto tiles = [&](auto hist_tag) {
      constexpr bool kH = decltype(hist_tag)::value;
      for (int i = 0; i < t1 - t0; ++i) {
        mbar_wait(full_k + st, parity);
        const uint8_t* kt = smem + L::k_off(st);
        const float* sks = reinterpret_cast<const float*>(smem + L::scale_off(stages, st));
        float scale_0 = 1.0f, scale_1 = 1.0f;
        uint32_t ksf = 0;
        if constexpr (kFp4) {
          ksf = reinterpret_cast<const uint32_t*>(sks)[warp * 16 + g + (t4 & 1) * 8];
        } else {
          scale_0 = sks[warp * 16 + g];
          scale_1 = sks[warp * 16 + g + 8];
        }
        uint32_t kf[kKSteps][4];
#pragma unroll
        for (int ks = 0; ks < kKSteps; ++ks)
          ldmatrix_x4(kf[ks], kt + F::swz(a_row, ks * 2 + a_chunk_hi));
        mbar_arrive(empty_k + st);
        const int key_0 = (t0 + i) * kBlockKV + warp * 16 + g, key_1 = key_0 + 8;
        // DeepGEMM's arithmetic, so the scores are bit-identical to its logits.
        float part[kNextN][2];
#pragma unroll
        for (int qi = 0; qi < kNextN; ++qi) part[qi][0] = part[qi][1] = 0.0f;
#pragma unroll
        for (int nt = 0; nt < kNTiles; ++nt) {
          float acc[kNextN][4];
#pragma unroll
          for (int qi = 0; qi < kNextN; ++qi)
            acc[qi][0] = acc[qi][1] = acc[qi][2] = acc[qi][3] = 0.0f;
#pragma unroll
          for (int ks = 0; ks < kKSteps; ++ks) {
#pragma unroll
            for (int qi = 0; qi < kNextN; ++qi) {
              if constexpr (kFp4) {
                mma_fp4<0>(acc[qi], kf[ks], qf[qi][ks][nt], (ksf >> (16 * ks)) & 0xffffu,
                           (qsf_r[qi][nt] >> (16 * ks)) & 0xffffu);
              } else {
                mma_fp8(acc[qi], kf[ks], qf[qi][ks][nt]);
              }
            }
          }
#pragma unroll
          for (int qi = 0; qi < kNextN; ++qi) {
            const float* d = acc[qi];
            const float w0 = wf[qi][nt][0], w1 = wf[qi][nt][1];
            part[qi][0] = __fadd_rn(
                part[qi][0], __fmaf_rn(fmaxf(d[0], 0.0f), w0, __fmul_rn(fmaxf(d[1], 0.0f), w1)));
            part[qi][1] = __fadd_rn(
                part[qi][1], __fmaf_rn(fmaxf(d[2], 0.0f), w0, __fmul_rn(fmaxf(d[3], 0.0f), w1)));
          }
        }
        float vk[kNextN][2];
#pragma unroll
        for (int qi = 0; qi < kNextN; ++qi) {
          float v_0 = part[qi][0], v_1 = part[qi][1];
          if constexpr (!kFp4) {
            v_0 = __fmul_rn(v_0, scale_0);
            v_1 = __fmul_rn(v_1, scale_1);
          }
          v_0 = __fadd_rn(v_0, __shfl_xor_sync(0xffffffffu, v_0, 1));
          v_1 = __fadd_rn(v_1, __shfl_xor_sync(0xffffffffu, v_1, 1));
          v_0 = __fadd_rn(v_0, __shfl_xor_sync(0xffffffffu, v_0, 2));
          v_1 = __fadd_rn(v_1, __shfl_xor_sync(0xffffffffu, v_1, 2));
          vk[qi][0] = v_0;
          vk[qi][1] = v_1;
        }
        // Every lane of a quad holds its two keys' sums for each row: lane
        // t4 stores row t4 >> 1, key t4 & 1, so one store and one count serve
        // both rows.
        if (kNextN == 2 || t4 < 2) {
          const int qi = kNextN == 2 ? t4 >> 1 : 0;
          const bool k1 = t4 & 1;
          const int key = k1 ? key_1 : key_0;
          float v;
          if constexpr (kNextN == 2)
            v = qi ? (k1 ? vk[1][1] : vk[1][0]) : (k1 ? vk[0][1] : vk[0][0]);
          else
            v = k1 ? vk[0][1] : vk[0][0];
          if (key < (qi == 0 ? len0 : len1)) {
            if constexpr (kKeep)
              asm volatile("st.global.L2::cache_hint.f32 [%0], %1, %2;" ::"l"(
                               lg + qi * p.logits_stride + key),
                           "f"(v), "l"(lg_pol)
                           : "memory");
            else
              lg[qi * p.logits_stride + key] = v;
            if constexpr (kH) {
              if (qi == 0 ? lrow0 : lrow1) {
                const uint32_t b = rowsel_bin(v);
                atomicAdd(shh + qi * 2048 + (b >> 1), 1u << ((b & 1u) * 16));
              }
            }
          }
        }
        if (++st == stages) {
          st = 0;
          parity ^= 1u;
        }
        // A bin gains at most kBlockKV a tile: flushed every 511 tiles, the
        // 16-bit counts cannot overflow.
        if constexpr (kH) {
          if ((i + 1) % 511 == 0) flush(u);
        }
      }
    };
    if constexpr (kLong) {
      if (lrow[0] || lrow[1]) {
        tiles(std::true_type{});
        flush(u);
      } else {
        tiles(std::false_type{});
      }
    } else {
      tiles(std::false_type{});
    }
  }
}

}  // namespace flashinfer::sm120_dsa_indexer
