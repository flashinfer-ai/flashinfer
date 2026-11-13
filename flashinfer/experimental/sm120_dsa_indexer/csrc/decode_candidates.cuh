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

// Decode candidate rows (DeepSeek-V4.1's consumer layers): each row ranks only
// the keys of the blocks a caller lists for it. The lists come in any order,
// with duplicates, -1 padding and blocks past the row; this kernel turns each
// into its distinct visible blocks in ascending order, -1 padded, and the
// virtual row's length. The scan then scores the listed blocks back to back as
// one row of at most num_cands blocks (decode_scan.cuh), and the selection maps
// its positions back to keys (decode_select_row.cuh). Ascending blocks keep the
// positions in key order, so the selection's tie-break (larger key) holds; only
// the newest block can be partial, and it is last.
#pragma once

#include <cstdint>

#include "decode_common.cuh"

namespace flashinfer::sm120_dsa_indexer {

constexpr int kCandThreads = 512;

struct CandParams {
  const int32_t* row_len;  // [num_rows] the rows' visible keys
  const int32_t* blocks;   // [num_rows, blocks_stride] the caller's lists
  int64_t blocks_stride;
  int num_cands;    // entries a list
  int shift;        // log2 of the block size
  int max_blocks;   // the bitmap's bits: blocks of max_context_len
  int32_t* sorted;  // [num_rows, num_cands]
  int32_t* vlen;    // [num_rows]
};

// One CTA a row. Dynamic shared memory: a bitmap of the bound's blocks, one
// bit a block (sized by the host from max_context_len).
__global__ void __launch_bounds__(kCandThreads)
    candidates_kernel(const __grid_constant__ CandParams p) {
  extern __shared__ uint32_t bits[];
  __shared__ int warp_sum[kCandThreads / 32];
  __shared__ int last_block;
  const int row = static_cast<int>(blockIdx.x), tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const int len = max(__ldg(p.row_len + row), 0);
  const int nb = min((len + (1 << p.shift) - 1) >> p.shift, p.max_blocks);
  const int nw = (nb + 31) >> 5;
  for (int w = tid; w < nw; w += kCandThreads) bits[w] = 0u;
  if (tid == 0) last_block = -1;
  __syncthreads();
  const int32_t* in = p.blocks + static_cast<int64_t>(row) * p.blocks_stride;
  for (int i = tid; i < p.num_cands; i += kCandThreads) {
    const int b = __ldg(in + i);
    if (b >= 0 && b < nb) {
      atomicOr(bits + (b >> 5), 1u << (b & 31));
      atomicMax(&last_block, b);
    }
  }
  __syncthreads();
  // Each thread compacts a contiguous run of words; an exclusive scan of the
  // runs' counts gives each run its place.
  const int per = (nw + kCandThreads - 1) / kCandThreads;
  const int w0 = min(tid * per, nw), w1 = min(w0 + per, nw);
  int count = 0;
  for (int w = w0; w < w1; ++w) count += __popc(bits[w]);
  int incl = count;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int v = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += v;
  }
  if (lane == 31) warp_sum[warp] = incl;
  __syncthreads();
  int base = 0, total = 0;
#pragma unroll
  for (int i = 0; i < kCandThreads / 32; ++i) {
    base += i < warp ? warp_sum[i] : 0;
    total += warp_sum[i];
  }
  int at = base + incl - count;
  int32_t* out = p.sorted + static_cast<int64_t>(row) * p.num_cands;
  for (int w = w0; w < w1; ++w) {
    uint32_t m = bits[w];
    while (m) {
      const int bit = __ffs(m) - 1;
      m &= m - 1;
      out[at++] = (w << 5) + bit;
    }
  }
  for (int i = total + tid; i < p.num_cands; i += kCandThreads) out[i] = -1;
  if (tid == 0) {
    const int last = last_block;
    p.vlen[row] =
        last < 0 ? 0 : ((total - 1) << p.shift) + min(1 << p.shift, len - (last << p.shift));
  }
  asm volatile("griddepcontrol.launch_dependents;");
}

}  // namespace flashinfer::sm120_dsa_indexer
