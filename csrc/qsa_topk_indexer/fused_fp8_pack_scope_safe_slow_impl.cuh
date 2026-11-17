// Copyright (c) 2026 by FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
// Scope-safe recomputation fallback for qsa_topk_slow.
//
// The shared histogram remains exactly 256 radix bins per token.  Unlike the
// legacy fallback, this kernel never reuses that histogram as an unbounded
// ceil(vis / 32)-word selection bitmap.  Once the exact radix threshold is
// known, it recomputes scores one final time and compacts each warp ballot
// directly to the destination row in ascending block-index order.

extern "C" __global__ void __launch_bounds__(NTK, 2)
    qsa_topk_slow_scope_safe(const u8* __restrict__ qi, const u8* __restrict__ ck,
                             const int* __restrict__ cu, const int* __restrict__ cpb,
                             int* __restrict__ packed, int B) {
  __shared__ int hist[TM * 256];

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;

  // Preserve the legacy reverse tile mapping so the fast and slow kernels
  // continue to partition the exact same rows.
  int total = 0;
  for (int rr = 0; rr < B; ++rr) {
    const int n = cu[rr + 1] - cu[rr] - FASTPOS;
    if (n > 0) total += (n + TM - 1) / TM;
  }
  if (static_cast<int>(blockIdx.x) >= total) return;

  int rem = total - 1 - static_cast<int>(blockIdx.x);
  int r = -1;
  int tile = 0;
  for (int rr = 0; rr < B; ++rr) {
    const int n = cu[rr + 1] - cu[rr] - FASTPOS;
    if (n <= 0) continue;
    const int ntl = (n + TM - 1) / TM;
    if (rem < ntl) {
      r = rr;
      tile = rem;
      break;
    }
    rem -= ntl;
  }
  if (r < 0) return;

  const int cu0 = cu[r];
  const int len = cu[r + 1] - cu0;
  const int pos0 = FASTPOS + tile * TM;
  const int t0 = cu0 + pos0;
  int nTok = len - pos0;
  if (nTok > TM) nTok = TM;
  const int visMax = (pos0 + nTok) >> 2;
  if (visMax <= VCAP) return;

  const u8* kbase = ck + static_cast<size_t>(cpb[r]) * CPAGE * IDXD;

  for (int i = tid; i < TM * 256; i += NTK) hist[i] = 0;
  __syncthreads();

  const unsigned full = 0xffffffffu;
  const unsigned ltmask = (1u << lane) - 1u;

  // One warp owns each token; with TM=16 every warp processes at most two.
  for (int m = warp; m < nTok; m += 8) {
    const int pos = pos0 + m;
    const int vis = (pos + 1) >> 2;
    const u8* qg = qi + static_cast<size_t>(t0 + m) * (IDXH * IDXD);
    int* hw = hist + m * 256;

    unsigned tau = 0u;
    int krem = KTOP;
    for (int shift = 24; shift >= 0; shift -= 8) {
      if (shift != 24) {
        for (int d = lane; d < 256; d += 32) hw[d] = 0;
        __syncwarp();
      }

      const unsigned hmask = (shift == 24) ? 0u : (0xffffffffu << (shift + 8));
      for (int base = 0; base < vis; base += 32) {
        const int v = base + lane;
        unsigned bin = 0x100u;
        if (v < vis) {
          const unsigned bits =
              qsa_score_key_exact(scoreG(qg, kbase + static_cast<size_t>(v) * IDXD));
          if ((bits & hmask) == tau) bin = (bits >> shift) & 0xffu;
        }
        const unsigned peers = __match_any_sync(full, bin);
        if (bin != 0x100u && (__ffs(peers) - 1) == lane) atomicAdd(&hw[bin], __popc(peers));
      }
      __syncwarp();

      int group_count = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) group_count += hw[lane * 8 + i];

      int suffix = group_count;
#pragma unroll
      for (int off = 1; off < 32; off <<= 1) {
        const int value = __shfl_down_sync(full, suffix, off);
        if (lane + off < 32) suffix += value;
      }

      const int above = suffix - group_count;
      const bool owner = above < krem && above + group_count >= krem;
      const int src = __ffs(__ballot_sync(full, owner)) - 1;
      int selected_digit = 0;
      int accumulated = above;
      if (owner) {
        for (selected_digit = lane * 8 + 7; selected_digit >= lane * 8; --selected_digit) {
          if (accumulated + hw[selected_digit] >= krem) break;
          accumulated += hw[selected_digit];
        }
      }

      int bucket = owner ? hw[selected_digit] : 0;
      selected_digit = __shfl_sync(full, selected_digit, src);
      accumulated = __shfl_sync(full, accumulated, src);
      bucket = __shfl_sync(full, bucket, src);
      tau |= static_cast<unsigned>(selected_digit) << shift;
      krem -= accumulated;
      __syncwarp();
      if (bucket == krem) break;
    }

    const int need = krem;
    int4* out = reinterpret_cast<int4*>(packed + static_cast<size_t>(t0 + m) * PWIDTH);

    // Direct stable compaction.  Blocks are visited in increasing index order,
    // and the prefix popcount assigns consecutive output slots.  Thus equal
    // scores retain the required lower-block-index tie break and output order.
    int runEq = 0;
    int rank = 0;
    for (int base = 0; base < vis; base += 32) {
      const int v = base + lane;
      int gt = 0;
      int eq = 0;
      if (v < vis) {
        const unsigned bits =
            qsa_score_key_exact(scoreG(qg, kbase + static_cast<size_t>(v) * IDXD));
        gt = bits > tau;
        eq = bits == tau;
      }
      const unsigned equal = __ballot_sync(full, eq);
      const int equal_prefix = __popc(equal & ltmask);
      const bool take = gt || (eq && runEq + equal_prefix < need);
      const unsigned selected = __ballot_sync(full, take);
      if (selected & (1u << lane)) {
        const int p = rank + __popc(selected & ltmask);
        const int b0 = v << 2;
        st_cs(out + p, make_int4(b0, b0 + 1, b0 + 2, b0 + 3));
      }
      rank += __popc(selected);
      runEq += __popc(equal);
    }

    if (lane == 0) {
      const int tailStart = vis << 2;
      const int tailCnt = (pos + 1) & 3;
      st_cs(out + ROW_V4 - 1,
            make_int4(tailCnt > 0 ? tailStart : -1, tailCnt > 1 ? tailStart + 1 : -1,
                      tailCnt > 2 ? tailStart + 2 : -1, KTOP * CR + tailCnt));
    }
    __syncwarp();
  }
}
