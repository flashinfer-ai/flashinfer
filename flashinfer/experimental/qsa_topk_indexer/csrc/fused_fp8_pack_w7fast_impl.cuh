// Copyright (c) 2026 by FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
// Shared-score-only top-k.  The caller launches this for tiles capped by
// VCAP; keeping the global recomputation fallback out of this kernel lowers
// register pressure enough to target four resident CTAs per Rubin SM.
extern "C" __global__ void __launch_bounds__(NTK, 4)
    qsa_topk_fast(const u8* __restrict__ qi, const u8* __restrict__ ck, const int* __restrict__ cu,
                  const int* __restrict__ cpb, int* __restrict__ packed, int B) {
  // The extra uint4 per row pads the FP8 query stride to 528 bytes, distributing
  // the four MMA A-fragment loads across all shared-memory banks.
  __shared__ __align__(16) uint4 qtile[TM][33];
  __shared__ int hist[TM * 256];

  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;

  // Fixed-range CTA mapping: CTA i owns packed rows [i*TM, i*TM + TM).  Every
  // scored row sits at request-local pos >= FASTPOS > TM, so one range never
  // holds scored rows of two requests: they all belong to the request of the
  // range's first row, found with one warp binary search instead of a
  // full-batch scan.  Rows below FASTPOS stay with the store kernel, and the
  // fast and slow scorers share this mapping and complementary visMax gates,
  // so each scored row keeps exactly one writer.
  // Reverse launch order (last range first) keeps the heaviest tiles — the
  // large-vis tails of each request — scheduled at the front of the wave,
  // matching the retired batch-scan mapping's schedule.
  const int t_begin = ((int)gridDim.x - 1 - (int)blockIdx.x) * TM;
  if (t_begin >= cu[B]) return;
  const int r = findReqWarp(cu, B, t_begin);
  const int cu0 = cu[r];
  int tile_end = t_begin + TM;
  if (tile_end > cu[r + 1]) tile_end = cu[r + 1];
  int t0 = cu0 + FASTPOS;
  if (t0 < t_begin) t0 = t_begin;
  const int nTok = tile_end - t0;
  if (nTok <= 0) return;
  const int pos0 = t0 - cu0;
  const int visMax = (pos0 + nTok) >> 2;
  // Row-level ownership: this kernel scores the rows whose own vis fits the
  // shared-score budget (vis <= VCAP) and the slow scorer owns the rest, so a
  // boundary tile can hold both kinds.  The tile exits only when even its
  // first (smallest-vis) row exceeds the budget.
  if (((pos0 + 1) >> 2) > VCAP) return;
  const int visEnd = visMax > VCAP ? VCAP : visMax;
  const u8* kbase = ck + (size_t)cpb[r] * CPAGE * IDXD;
  float* sc = reinterpret_cast<float*>(packed) + (size_t)t0 * SCSTR;

  for (int i = tid; i < TM * 256; i += NTK) hist[i] = 0;
  __syncthreads();
  for (int v = tid; v < TM * 32; v += NTK) {
    const int m = v >> 5, vec = v & 31;
    uint4 x = make_uint4(0u, 0u, 0u, 0u);
    if (m < nTok)
      x = *reinterpret_cast<const uint4*>(qi + (size_t)(t0 + m) * IDXH * IDXD + vec * 16);
    qtile[m][vec] = x;
  }
  __syncthreads();

  const int group = lane >> 2, quad = lane & 3;
  const int row0 = group, row1 = group + 8;
  const u8* q0base = reinterpret_cast<const u8*>(qtile[row0]);
  const u8* q1base = reinterpret_cast<const u8*>(qtile[row1]);
  for (int blockBase = warp * 8; blockBase < visEnd; blockBase += 64) {
    const int keyCol = blockBase + group;
    const bool keyValid = keyCol < visEnd;
    const u8* key = keyValid ? kbase + (size_t)keyCol * IDXD : kbase;
    float sum0 = 0.f, sum1 = 0.f, sum2 = 0.f, sum3 = 0.f;
#pragma unroll
    for (int h = 0; h < IDXH; ++h) {
      float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
#pragma unroll
      for (int kt = 0; kt < 4; ++kt) {
        const int k0 = kt * 32 + quad * 4;
        const u8* q0 = q0base + h * IDXD;
        const u8* q1 = q1base + h * IDXD;
        unsigned a0 = *reinterpret_cast<const unsigned*>(q0 + k0);
        unsigned a1 = *reinterpret_cast<const unsigned*>(q1 + k0);
        unsigned a2 = *reinterpret_cast<const unsigned*>(q0 + k0 + 16);
        unsigned a3 = *reinterpret_cast<const unsigned*>(q1 + k0 + 16);
        unsigned b0 = keyValid ? *reinterpret_cast<const unsigned*>(key + k0) : 0u;
        unsigned b1 = keyValid ? *reinterpret_cast<const unsigned*>(key + k0 + 16) : 0u;
        mma_fp8(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
      }
      sum0 += qsa_relu_exact(d0);
      sum1 += qsa_relu_exact(d1);
      sum2 += qsa_relu_exact(d2);
      sum3 += qsa_relu_exact(d3);
    }
    const int col = blockBase + quad * 2;
    // A row past the budget belongs to the slow scorer: never touch its
    // packed row (sc aliases it) or its histogram.
    const int v0 = (pos0 + row0 + 1) >> 2;
    const int v1 = (pos0 + row1 + 1) >> 2;
    const int vis0 = v0 <= VCAP ? v0 : 0;
    const int vis1 = v1 <= VCAP ? v1 : 0;
    if (row0 < nTok && col < vis0) {
      sc[(size_t)row0 * SCSTR + col] = sum0;
      atomicAdd(&hist[row0 * 256 + (qsa_score_key_exact(sum0) >> 24)], 1);
    }
    if (row0 < nTok && col + 1 < vis0) {
      sc[(size_t)row0 * SCSTR + col + 1] = sum1;
      atomicAdd(&hist[row0 * 256 + (qsa_score_key_exact(sum1) >> 24)], 1);
    }
    if (row1 < nTok && col < vis1) {
      sc[(size_t)row1 * SCSTR + col] = sum2;
      atomicAdd(&hist[row1 * 256 + (qsa_score_key_exact(sum2) >> 24)], 1);
    }
    if (row1 < nTok && col + 1 < vis1) {
      sc[(size_t)row1 * SCSTR + col + 1] = sum3;
      atomicAdd(&hist[row1 * 256 + (qsa_score_key_exact(sum3) >> 24)], 1);
    }
  }
  __syncthreads();

  // ---- per-warp exact top-512: 2 tokens per warp (TM=16, 8 warps) ----
  const unsigned FULL = 0xFFFFFFFFu;
  const unsigned ltmask = (1u << lane) - 1u;
  for (int m = warp; m < nTok; m += 8) {
    const int pos = pos0 + m;
    const int vis = (pos + 1) >> 2;
    if (vis > VCAP) continue;  // slow-scorer row (uniform across the warp)
    const float* scm = sc + (size_t)m * SCSTR;
    int* hw = hist + m * 256;

    unsigned tau = 0u;
    int krem = KTOP;
    for (int shift = 24; shift >= 0; shift -= 8) {
      unsigned hmask = (shift == 24) ? 0u : (0xFFFFFFFFu << (shift + 8));
      // The top byte histogram was accumulated while scores were produced.
      // Lower radix passes rebuild only from the shared-score row.
      if (shift != 24) {
        for (int d = lane; d < 256; d += 32) hw[d] = 0;
        __syncwarp();
        const int nv4 = vis >> 2;
        const int ni = (nv4 + 31) >> 5;  // uniform across the warp
        const float4* s4 = reinterpret_cast<const float4*>(scm);
        for (int it = 0; it < ni; ++it) {
          const int i = (it << 5) + lane;
          float4 f = make_float4(0.f, 0.f, 0.f, 0.f);
          if (i < nv4) f = s4[i];
          const float fv[4] = {f.x, f.y, f.z, f.w};
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            unsigned bin = 0x100u;
            if (i < nv4) {
              unsigned b = qsa_score_key_exact(fv[u]);
              if ((b & hmask) == tau) bin = (b >> shift) & 0xFFu;
            }
            unsigned peers = __match_any_sync(FULL, bin);
            if (bin != 0x100u && (__ffs(peers) - 1) == lane) atomicAdd(&hw[bin], __popc(peers));
          }
        }
        for (int v = (nv4 << 2) + lane; v < (nv4 << 2) + 32; v += 32) {
          unsigned bin = 0x100u;
          if (v < vis) {
            unsigned b = qsa_score_key_exact(scm[v]);
            if ((b & hmask) == tau) bin = (b >> shift) & 0xFFu;
          }
          unsigned peers = __match_any_sync(FULL, bin);
          if (bin != 0x100u && (__ffs(peers) - 1) == lane) atomicAdd(&hw[bin], __popc(peers));
        }
        __syncwarp();
      }
      int g = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) g += hw[lane * 8 + i];
      int suf = g;
#pragma unroll
      for (int off = 1; off < 32; off <<= 1) {
        int y = __shfl_down_sync(FULL, suf, off);
        if (lane + off < 32) suf += y;
      }
      const int above = suf - g;
      const bool owner = (above < krem) && (above + g >= krem);
      const int src = __ffs(__ballot_sync(FULL, owner)) - 1;
      int dsel = 0, acc = above;
      if (owner) {
        for (dsel = lane * 8 + 7; dsel >= lane * 8; --dsel) {
          if (acc + hw[dsel] >= krem) break;
          acc += hw[dsel];
        }
      }
      int bucket = owner ? hw[dsel] : 0;
      dsel = __shfl_sync(FULL, dsel, src);
      acc = __shfl_sync(FULL, acc, src);
      bucket = __shfl_sync(FULL, bucket, src);
      tau |= ((unsigned)dsel) << shift;
      krem -= acc;
      __syncwarp();
      // If every score in the threshold prefix bucket is required, the
      // unresolved low bits cannot affect membership.  Keeping tau at the
      // prefix's lower bound makes larger values `gt`; values exactly at the
      // bound are all admitted by need==bucket, preserving lower-index ties.
      if (bucket == krem) break;
    }
    const int need = krem;

    int4* out = reinterpret_cast<int4*>(packed + (size_t)(t0 + m) * PWIDTH);
    // Scores occupy this destination row, so first materialize the exact
    // selection flags into the no-longer-needed radix histogram.  Only after
    // every score has been consumed do we compact flags into the row.
    int runEq = 0;
    for (int base = 0; base < vis; base += 32) {
      const int v = base + lane;
      int gt = 0, eq = 0;
      if (v < vis) {
        float s = scm[v];
        unsigned b = qsa_score_key_exact(s);
        gt = (b > tau);
        eq = (b == tau);
      }
      unsigned mg = __ballot_sync(FULL, gt), me = __ballot_sync(FULL, eq);
      const int peq = __popc(me & ltmask);
      const bool take = gt || (eq && runEq + peq < need);
      const unsigned mt = __ballot_sync(FULL, take);
      if (lane == 0) hw[base >> 5] = static_cast<int>(mt);
      runEq += __popc(me);
    }
    __syncwarp();

    int rank = 0;
    for (int base = 0; base < vis; base += 32) {
      const unsigned mt = static_cast<unsigned>(hw[base >> 5]);
      if (mt & (1u << lane)) {
        const int p = rank + __popc(mt & ltmask);
        const int b0 = (base + lane) << 2;
        st_cs(out + p, make_int4(b0, b0 + 1, b0 + 2, b0 + 3));
      }
      rank += __popc(mt);
    }
    if (lane == 0) {
      const int tailStart = vis << 2;
      const int tailCnt = (pos + 1) & 3;
      int4 v;
      v.x = (0 < tailCnt) ? (tailStart + 0) : -1;
      v.y = (1 < tailCnt) ? (tailStart + 1) : -1;
      v.z = (2 < tailCnt) ? (tailStart + 2) : -1;
      v.w = KTOP * CR + tailCnt;
      st_cs(out + ROW_V4 - 1, v);
    }
    __syncwarp();
  }
}
