/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file alphamoe_router.cuh
 * \brief Fused AlphaMoE gating router ("vibecuda" backend, plain CUDA).
 *
 * Consumes per-token expert logits and emits the complete block-sparse MoE
 * routing metadata bundle in a small, shape-adaptive kernel pipeline:
 *   - stable descending top-k selection over the routed experts (equal logits
 *     keep the lower expert index first; stable sort order), with the
 *     optional shared expert appended as the last selected column,
 *   - max-subtracted fp32 softmax over the selected logits,
 *   - expert histogram, block_m-aligned padded expert offsets, the
 *     per-expert scatter offsets (identical to the expert histogram), and
 *     the padded extent,
 *   - deterministic expert-grouped sorted route ids (flat token*top_k+slot
 *     route indices, increasing token order) and per-block expert ids.
 *
 * Dispatch (all thresholds measured on Blackwell B300):
 *   small tiles (experts <= 512, tokens <= 16, pairs <= 256) run as a single
 *   fused_small_kernel: selection + histogram + scan + scatter + fills in one
 *   block behind block barriers, so a forward costs one launch instead of
 *   three. Medium small inputs (num_experts <= 1024) run as:
 *     select_kernel<CPL> : warp-per-token stable top-k selection with
 *       register-cached ordering keys and fp32 max-subtracted softmax.
 *     finish_kernel (tokens <= 256) : every block redundantly builds the
 *       expert histogram, (expert x token) route bitmap and padded exclusive
 *       scan in smem, then all blocks grid-stride the bitmap-rank scatter and
 *       the sentinel/zero fills; block 0 stores the histogram/scan outputs.
 *     reduce_kernel + tail_kernel (tokens > 256) : single-block
 *       histogram/scan, then a per-expert-warp ballot rescan of the route
 *       stream.
 *   Dependent kernels overlap predecessor tails via Programmatic Dependent
 *   Launch on SM90+ (pdl_sync gates every global read; a plain launch is used
 *   on older architectures). Large inputs (num_experts > 1024) take a generic
 *   grid-parallel path (select with global histogram atomics + single-block
 *   scan + grid scatter) with identical semantics; it requires one scratch
 *   element per route (num_tokens * ceil(num_experts/32) ints).
 *
 * The file is framework-agnostic: no torch headers. Callers allocate all
 * outputs (and the generic-path scratch) and pass raw pointers plus the
 * current CUDA stream. All outputs are fully written on every call, shapes
 * depend only on (num_tokens, num_experts, top_k, block_m), and there is no
 * host synchronization, so the interface is CUDA-graph capture safe.
 */

#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace flashinfer::fused_moe {


constexpr int kSelectThreads = 128;
constexpr int kReduceThreads = 512;
constexpr int kScatterThreads = 128;
constexpr int kScanThreads = 1024;
constexpr int kReduceMaxExperts = 1024;
constexpr int kSbMaxTwords = 8;  // route-bitmap scatter covers tokens <= 256
constexpr int kSbBitmapInts = kReduceMaxExperts * kSbMaxTwords;  // 32KB smem
constexpr int kSBlkCap = 1000;  // finish_kernel block->expert flat-map capacity
// finish_kernel's scan phase (which already touches every (word, expert)
// pair with conflict-free column accesses) stamps each bitmap word's base
// slot (s_off[e] + column prefix) into a small dynamic-smem table. The
// route-parallel scatter then needs ONE table load + ONE bitmask popcount
// per route instead of TW random-column popcounts, keeping full grid
// coverage and zero extra global traffic. Dynamic smem keeps the static
// footprint under the 48KB compiler limit.
// Single-block fully-fused path (small tiles): the whole 5-stage pipeline in
// one launch, removing the launch-chain + PDL floor that dominates tiny
// forwards. Guards: num_experts <= kFusedMaxExperts, pairs <= kFusedPairsCap,
// tokens <= 16 (one token per select warp).
constexpr int kFusedThreads = 512;
constexpr int kFusedMaxExperts = 512;
constexpr int kFusedMaxTwords = 8;
constexpr int kFusedPairsCap = 256;

// Programmatic Dependent Launch helpers (SM90+). Successor kernels sync
// against their stream predecessors before touching any global memory, so
// launch scaffolding and shared-memory setup overlap the predecessor's tail.
__device__ __forceinline__ void pdl_sync() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize();
#endif
}

__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// Launch with the programmatic-stream-serialization attribute so the kernel
// may begin its (sync-guarded) prologue while the predecessor finishes.
template <typename Kernel, typename... Args>
void launch_pdl(Kernel kernel, int blocks, int threads, size_t smem_bytes,
                cudaStream_t stream, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)blocks);
    cfg.blockDim = dim3((unsigned)threads);
    cfg.dynamicSmemBytes = smem_bytes;
    cfg.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    const cudaError_t err = cudaLaunchKernelEx(&cfg, kernel, args...);
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("alphamoe_router: PDL launch "
                                 "failed: ") +
                               cudaGetErrorString(err));
    }
}

// Monotonic map float -> uint32 so that descending float order matches
// descending integer order (handles the full IEEE ordering incl. negatives).
__device__ __forceinline__ unsigned int float_sortable_key(float f) {
    unsigned int u = __float_as_uint(f);
    return u ^ ((u & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u);
}

// Inverse of float_sortable_key: recover the original float from its key.
__device__ __forceinline__ float float_from_sortable_key(unsigned int u) {
    unsigned int v = (u & 0x80000000u) ? (u ^ 0x80000000u) : (u ^ 0xFFFFFFFFu);
    return __uint_as_float(v);
}

// Exact branch-free unsigned division by a kernel-constant divisor
// (Granlund-Montgomery round-up method, same family as libdivide's
// branchfree u32 path). top_k and block_m are Model-constructor constants,
// so the launcher builds the magic pair once per forward on the host and
// passes it by value; on device each division is one mulhi + 3 cheap ops
// instead of the compiler's ~15-instruction runtime s32 division sequence.
// Exact for every 32-bit dividend and divisor >= 1.
struct FastDivU32 {
    unsigned int magic;  // 0 marks the power-of-two (shift) path
    unsigned int shift;
};

static inline FastDivU32 make_fast_div_u32(unsigned int d) {
    FastDivU32 fd;
    if ((d & (d - 1u)) == 0u) {  // includes d == 1 (shift 0)
        fd.magic = 0u;
        fd.shift = (unsigned int)__builtin_ctz(d);
    } else {
        const unsigned int l = 31u - (unsigned int)__builtin_clz(d);
        const unsigned long long two_pow = 1ull << (32 + l);
        unsigned long long m = two_pow / d;
        unsigned long long rem = two_pow - m * (unsigned long long)d;
        m += m;
        rem += rem;
        if (rem >= (unsigned long long)d) {
            m += 1;
        }
        m += 1;
        fd.magic = (unsigned int)m;  // stores the low 32 bits (33rd implied)
        fd.shift = l;
    }
    return fd;
}

__device__ __forceinline__ unsigned int fdiv_u32(unsigned int n,
                                                 FastDivU32 fd) {
    if (fd.magic == 0u) {
        return n >> fd.shift;
    }
    const unsigned int t = __umulhi(n, fd.magic);
    return (t + ((n - t) >> 1)) >> fd.shift;
}

// 64-bit ordering key: descending logit, ties keep the lower expert index
// first. Keys are unique per expert so iterating "argmax key below previous
// winner" reproduces exactly the stable descending index order.
__device__ __forceinline__ unsigned long long route_key(float f, int expert) {
    unsigned long long hi = (unsigned long long)float_sortable_key(f);
    unsigned long long lo = (unsigned long long)(0xFFFFFFFFu - (unsigned int)expert);
    return (hi << 32) | lo;
}

// Warp-wide stable top-k selection for one token. CPL is a compile-time cap
// on ceil(routed_experts / 32) so candidate keys live in registers. Lane
// (j & 31) owns selection round j: records id/logit. The per-round warp max
// uses two hardware redux ops instead of a 10-instruction shuffle. SHARED is
// compile-time so the shared-expert round stays out of the selection loop.
// EMIT additionally bumps the smem expert histogram and (token, expert) route
// bitmap from the owning lane's registers, removing the separate phase that
// re-reads the just-written topk ids from global memory (fused small path).
// Round j's selected expert is also returned through own_expert for lanes
// where lane == (j & 31) (only tracked while j < 32; -1 otherwise).
template <int CPL, bool SHARED, int EMIT = 0>
__device__ __forceinline__ void token_select(
    const float* __restrict__ row,
    float* __restrict__ w_row,
    int* __restrict__ id_row,
    int lane,
    int num_experts,
    int top_k,
    int routed_top_k,
    int* __restrict__ own_expert = nullptr,
    int* __restrict__ s_hist = nullptr,
    unsigned int* __restrict__ s_bitmap = nullptr,
    int token = 0) {
    const int routed_experts = num_experts - (SHARED ? 1 : 0);

    unsigned long long keys[CPL];
#pragma unroll
    for (int c = 0; c < CPL; ++c) {
        const int e = lane + c * 32;
        keys[c] = (e < routed_experts) ? route_key(row[e], e) : 0ull;
    }

    int own = -1;
    unsigned long long prev = 0xFFFFFFFFFFFFFFFFull;
    // Register-distributed softmax (top_k <= 32): each lane keeps the value
    // of its own selection round, avoiding a global round-trip through w_row.
    const bool reg_softmax = top_k <= 32;
    float v_acc = -INFINITY;
    for (int j = 0; j < routed_top_k; ++j) {
        // Mask out keys at/above the previous winner, then take the max
        // as a pairwise tree (log2(CPL) compare depth instead of a
        // CPL-long dependent chain -- this is the per-round critical path).
        unsigned long long cand[CPL];
#pragma unroll
        for (int c = 0; c < CPL; ++c) {
            cand[c] = (keys[c] < prev) ? keys[c] : 0ull;
        }
#pragma unroll
        for (int step = CPL / 2; step > 0; step >>= 1) {
#pragma unroll
            for (int c = 0; c < step; ++c) {
                cand[c] = cand[c] > cand[c + step] ? cand[c] : cand[c + step];
            }
        }
        const unsigned long long best = cand[0];
        const unsigned int lane_hi = (unsigned int)(best >> 32);
        const unsigned int best_hi = __reduce_max_sync(0xFFFFFFFFu, lane_hi);
        const unsigned int tag =
            (lane_hi == best_hi) ? (unsigned int)best : 0u;
        const unsigned int best_lo = __reduce_max_sync(0xFFFFFFFFu, tag);
        prev = ((unsigned long long)best_hi << 32) | best_lo;
        const int expert = (int)(0xFFFFFFFFu - best_lo);
        const float value = float_from_sortable_key(best_hi);  // logit in key
        if (reg_softmax) {
            // All lanes track their own round's value via a cheap select.
            v_acc = (lane == (j & 31)) ? value : v_acc;
        }
        if (lane == (j & 31)) {
            id_row[j] = expert;
            if (!reg_softmax) {
                w_row[j] = value;  // scratch: selected logits, softmax below
            }
            if (j < 32) {
                own = expert;
            }
            if (EMIT == 1) {
                atomicAdd(s_hist + expert, 1);
                atomicOr(s_bitmap + expert, 1u << token);
            } else if (EMIT == 3) {
                // Bitmap only: counts derivable as bitmap popcounts when
                // num_tokens <= block_m (POPC fused path).
                atomicOr(s_bitmap + expert, 1u << token);
            } else if (EMIT == 2) {
                // Single-token path: expert bitmap over words + plain route
                // record (selected experts are distinct, so no contention).
                atomicOr(s_bitmap + (expert >> 5), 1u << (expert & 31));
                s_hist[expert] = j;
            }
        }
    }
    // Shared-expert round (compile-time presence): last logit column, id
    // num_experts-1, appended as the final topk column.
    if (SHARED) {
        const int j = routed_top_k;
        const float value = row[num_experts - 1];
        if (reg_softmax) {
            v_acc = (lane == (j & 31)) ? value : v_acc;
        }
        if (lane == (j & 31)) {
            id_row[j] = num_experts - 1;
            if (!reg_softmax) {
                w_row[j] = value;
            }
            if (j < 32) {
                own = num_experts - 1;
            }
            if (EMIT == 1) {
                atomicAdd(s_hist + (num_experts - 1), 1);
                atomicOr(s_bitmap + (num_experts - 1), 1u << token);
            } else if (EMIT == 3) {
                atomicOr(s_bitmap + (num_experts - 1), 1u << token);
            } else if (EMIT == 2) {
                atomicOr(s_bitmap + ((num_experts - 1) >> 5),
                         1u << ((num_experts - 1) & 31));
                s_hist[num_experts - 1] = j;
            }
        }
    }
    if (own_expert != nullptr) {
        *own_expert = own;
    }
    __syncwarp(0xFFFFFFFFu);

    const int rounds = routed_top_k + (SHARED ? 1 : 0);
    if (reg_softmax) {
        float m = v_acc;
#pragma unroll
        for (int ofs = 16; ofs > 0; ofs >>= 1) {
            const float other = __shfl_xor_sync(0xFFFFFFFFu, m, ofs);
            m = fmaxf(m, other);
        }
        const float e = (lane < rounds) ? expf(v_acc - m) : 0.0f;
        float sum = e;
#pragma unroll
        for (int ofs = 16; ofs > 0; ofs >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFFu, sum, ofs);
        }
        if (lane < rounds) {
            w_row[lane] = e / sum;
        }
        return;
    }

    float sm = -INFINITY;
    for (int j = lane; j < top_k; j += 32) {
        sm = fmaxf(sm, w_row[j]);
    }
    for (int ofs = 16; ofs > 0; ofs >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFFu, sm, ofs);
        sm = fmaxf(sm, other);
    }
    float ssum = 0.0f;
    for (int j = lane; j < top_k; j += 32) {
        ssum += expf(w_row[j] - sm);
    }
    for (int ofs = 16; ofs > 0; ofs >>= 1) {
        ssum += __shfl_xor_sync(0xFFFFFFFFu, ssum, ofs);
    }
    const float inv = 1.0f / ssum;
    for (int j = lane; j < top_k; j += 32) {
        w_row[j] = expf(w_row[j] - sm) * inv;
    }
}


// CPL=16 expert band with register-softmax top_k (top_k <= 32): four
// warps share one token's tournament (select_quad_kernel, 128-thread block,
// CPL4 keys per lane per warp): warp w owns experts [w*32*CPL4,
// (w+1)*32*CPL4) and each round's four local winners combine through shared
// memory behind one double-buffered block barrier (round j touches slot j&1
// only, so one barrier suffices). Selection math is identical to the
// single-warp tournament -- the round max is the max of the four local
// maxima of keys below prev -- so the stable-tie output matches bit for
// bit. Narrower bands and the fused paths keep the single-warp tournament.
// Fully-fused single-block path for small tiles. One launch runs selection,
// histogram, padded scan and bitmap-rank scatter with block barriers in place
// of the multi-kernel PDL chain, so the whole forward costs one launch
// instead of three. One warp per token (up to 16); num_tokens <= 16 keeps
// the (expert x token) smem bitmap to one word per expert and warp 0's
// serial padded scan to per-lane chunks of at most 16 experts.
// POPC (num_tokens <= block_m): every expert count is bounded by num_tokens,
// so counts are recovered as bitmap popcounts and the smem histogram zero
// loop plus the per-round atomicAdds disappear entirely.
template <int CPL, bool SHARED, bool POPC = false>
__global__ void __launch_bounds__(kFusedThreads)
fused_small_kernel(const float* __restrict__ logits,
                   float* __restrict__ topk_weights,
                   int* __restrict__ topk_ids,
                   int* __restrict__ expert_counts,
                   int* __restrict__ expert_offsets,
                   int* __restrict__ num_tokens_post_padded,
                   int* __restrict__ expert_scatter_offsets,
                   int* __restrict__ block_expert_ids,
                   int* __restrict__ sorted_token_ids,
                   int num_tokens,
                   int num_experts,
                   int top_k,
                   int block_m,
                   int routed_top_k,
                   int pairs,
                   int max_blocks,
                   int slots,
                   int twords,
                   FastDivU32 fd_topk,
                   FastDivU32 fd_bm) {
    __shared__ int s_hist[kFusedMaxExperts];
    __shared__ unsigned int s_bitmap[kFusedMaxExperts * kFusedMaxTwords];
    __shared__ int s_off[kFusedMaxExperts + 1];
    __shared__ int s_wsum[kFusedThreads / 32];

    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    constexpr int kWarps = kFusedThreads / 32;

    // num_tokens <= 16 in this path, so the route bitmap is one word per
    // expert (the twords param is always 1; see launch_fused).
    for (int e = tid; e < num_experts; e += kFusedThreads) {
        if (!POPC) {
            s_hist[e] = 0;
        }
        s_bitmap[e] = 0u;
    }
    __syncthreads();

    // Warp-per-token stable selection (num_tokens <= warps in this path: one
    // token per warp). Emitting warps bump the smem histogram/bitmap from
    // registers as rounds resolve, removing the idle reload phase over the
    // just-written global ids. Warps without a token instead zero the two
    // global outputs *concurrently* with selection, so the post-scan leg
    // only writes real route ids, in-segment padding sentinels, and real
    // per-block expert ids; the barrier below orders these plain stores
    // before the later overlapping stores by other threads.
    int own_expert = -1;
    const int token = warp;
    if (token < num_tokens) {
        token_select<CPL, SHARED, POPC ? 3 : 1>(
            logits + (long long)token * num_experts,
            topk_weights + (long long)token * top_k,
            topk_ids + (long long)token * top_k, lane, num_experts, top_k,
            routed_top_k, &own_expert, s_hist, s_bitmap, token);
    } else {
        const int ptid = tid - num_tokens * 32;
        const int pthreads = kFusedThreads - num_tokens * 32;
        for (int s = ptid; s < slots; s += pthreads) {
            sorted_token_ids[s] = 0;
        }
        for (int b = ptid; b < max_blocks; b += pthreads) {
            block_expert_ids[b] = 0;
        }
    }
    __syncthreads();

    // Block_m-padded exclusive scan (single chunk: num_experts <= threads).
    // Round-31 latency design: the zero tails were already filled by the
    // non-emitter warps during selection, and the tail below collapses
    // scatter + padding + ownership into one barrier-free final leg, so the
    // old 6-barrier phase machine (scan / s_off / ownership+rendezvous /
    // three strided fill passes) needs only 3 barriers total. Every warp
    // redundantly scans the 16 warp sums in registers and shuffles out its
    // own base (the finish_kernel round-11 pattern), so no warp-0-only
    // serial section lengthens the scan barrier's arrival leg -- the
    // r16-r18 single-block rule never violated here. (A round-31A
    // single-warp serial scan over per-lane expert chunks was REVERTED on
    // measurement: production 5.54us vs this hybrid's target ~5.1us at
    // (8,257,9,8); the warp-0-only leg between the two remaining barriers
    // simply moved the rendezvous cost onto the scan barrier.)
    int count = 0;
    if (tid < num_experts) {
        count = __popc(s_bitmap[tid]);  // one bitmap word per expert here
        s_hist[tid] = count;  // normalize for the segment fills
        expert_counts[tid] = count;
        expert_scatter_offsets[tid] = count;
    }
    const int value =
        (int)(fdiv_u32((unsigned int)(count + block_m - 1), fd_bm) *
              (unsigned int)block_m);
    int scan = value;
    for (int ofs = 1; ofs < 32; ofs <<= 1) {
        const int other = __shfl_up_sync(0xFFFFFFFFu, scan, ofs);
        if (lane >= ofs) {
            scan += other;
        }
    }
    if (lane == 31) {
        s_wsum[warp] = scan;
    }
    __syncthreads();
    int wsum = (lane < kWarps) ? s_wsum[lane] : 0;
    for (int ofs = 1; ofs < kWarps; ofs <<= 1) {
        const int other = __shfl_up_sync(0xFFFFFFFFu, wsum, ofs);
        if (lane >= ofs) {
            wsum += other;
        }
    }
    const int warp_base =
        (warp > 0) ? __shfl_sync(0xFFFFFFFFu, wsum, warp - 1) : 0;
    const int inclusive = warp_base + scan;
    if (tid < num_experts) {
        expert_offsets[tid + 1] = inclusive;
        s_off[tid + 1] = inclusive;
    }
    const int extent = __shfl_sync(0xFFFFFFFFu, wsum, kWarps - 1);
    if (tid == 0) {
        expert_offsets[0] = 0;
        s_off[0] = 0;
        num_tokens_post_padded[0] = extent;
    }
    __syncthreads();

    // Final leg (no barrier after): emitters scatter their kept route from
    // the register-cached expert (rank = number of earlier tokens selecting
    // the same expert), and every thread tid < num_experts writes its
    // expert's in-segment padding sentinels and per-block expert ids.
    if (top_k <= 32 && token < num_tokens) {
        if (own_expert >= 0) {
            const int e = own_expert;
            const int rank =
                __popc(s_bitmap[e] & ((1u << token) - 1u));
            sorted_token_ids[s_off[e] + rank] = token * top_k + lane;
        }
    } else if (top_k > 32) {
        for (int r = tid; r < pairs; r += kFusedThreads) {
            const int token = (int)fdiv_u32((unsigned int)r, fd_topk);
            const int e = topk_ids[r];
            const int rank = __popc(s_bitmap[e] & ((1u << token) - 1u));
            sorted_token_ids[s_off[e] + rank] = r;
        }
    }
    if (tid < num_experts) {
        const int off = s_off[tid];
        const int end = s_off[tid + 1];
        for (int s = off + s_hist[tid]; s < end; ++s) {
            sorted_token_ids[s] = pairs;  // padding sentinel
        }
        for (int b = (int)fdiv_u32((unsigned int)off, fd_bm);
             b < (int)fdiv_u32((unsigned int)end, fd_bm); ++b) {
            block_expert_ids[b] = tid;
        }
    }
    // num_tokens == kFusedThreads/32: no warp was free for the selection-leg
    // prefill, so the zero tails are written here instead (segment interiors
    // are fully covered by the scatter + sentinel stores above).
    if (num_tokens * 32 == kFusedThreads) {
        const int extent = s_off[num_experts];
        const int used_blocks = (int)fdiv_u32((unsigned int)extent, fd_bm);
        for (int s = extent + tid; s < slots; s += kFusedThreads) {
            sorted_token_ids[s] = 0;
        }
        for (int b = used_blocks + tid; b < max_blocks; b += kFusedThreads) {
            block_expert_ids[b] = 0;
        }
    }
}

// Single-warp fully-fused path for exactly one token. The 512-thread fused
// path pays phase-separation __syncthreads barriers for phases a single warp
// can run serially with __syncwarp only: histogram/scan/fills here collapse
// to one smem expert-bitmap pass plus one 32-expert-per-lane chunk sweep.
// Valid for num_experts <= kFusedMaxExperts and top_k <= 32 (one selection
// round per lane); count per expert is 1 (selection rounds choose distinct
// experts), so each padded segment is exactly block_m slots.
template <int CPL, bool SHARED>
__global__ void fused_single_kernel(const float* __restrict__ logits,
                                    float* __restrict__ topk_weights,
                                    int* __restrict__ topk_ids,
                                    int* __restrict__ expert_counts,
                                    int* __restrict__ expert_offsets,
                                    int* __restrict__ num_tokens_post_padded,
                                    int* __restrict__ expert_scatter_offsets,
                                    int* __restrict__ block_expert_ids,
                                    int* __restrict__ sorted_token_ids,
                                    int num_experts,
                                    int top_k,
                                    int block_m,
                                    int routed_top_k,
                                    int max_blocks,
                                    int slots,
                                    FastDivU32 fd_bm) {
    __shared__ int s_seg[32];  // segment index -> expert id
    __shared__ int s_route[kFusedMaxExperts];
    __shared__ unsigned int s_bits[kFusedMaxExperts / 32];

    const int lane = (int)threadIdx.x;
    const int words = (num_experts + 31) / 32;
    for (int w = lane; w < words; w += 32) {
        s_bits[w] = 0u;
    }
    __syncwarp(0xFFFFFFFFu);

    int own_expert = -1;
    token_select<CPL, SHARED, 2>(logits, topk_weights, topk_ids, lane,
                                 num_experts, top_k, routed_top_k, &own_expert,
                                 s_route, s_bits, 0);

    // Per-lane chunk of 32 experts: prefix over chunk bitmaps, then a
    // popcount-parallel sweep (no serial carry chain). counts, scatter and
    // offsets all go out as aligned int4 vectors: writing the UNSHIFTED
    // offsets window (offsets[e] = selected-count prefix below e times
    // block_m) keeps 16B alignment, with the offsets[num_experts] tail
    // stored by the last-chunk lane. The earlier per-element store loop
    // with a serial carry measured ~1us slower here: it serialized 100+
    // dependent stores per lane.
    const int rounds = routed_top_k + (SHARED ? 1 : 0);
    const int extent = rounds * block_m;
    // Round-26 re-chunk experiment, retained on measured latency: 16-expert
    // chunks. The original sweep maps one 32-expert chunk per lane, so at
    // num_experts <= kFusedMaxExperts (512) only ceil(num_experts/32) lanes
    // work (16 of 32 at 512 experts) while the other half of the warp sits
    // idle through the 8-quad serial store chain. 16-expert chunks activate
    // all 32 lanes for the full fused-single band (exactly 32 chunks at 512
    // experts, matching the guard) and halve the per-lane quad chain to 4 at
    // that endpoint. Bitwise-identical outputs: chunk c covers experts
    // [16c, 16c+16), i.e. half of bitmap word c/2 (bit position preserved);
    // int4 stores stay 16B aligned since chunk bases are multiples of 16
    // experts (64B). Revert = set kSingleChunkExperts back to 32.
    constexpr int kSingleChunkExperts = 16;
    constexpr int kSingleQuads = kSingleChunkExperts / 4;
    static_assert(kSingleChunkExperts == 16 || kSingleChunkExperts == 32);
    const int nchunks =
        (num_experts + kSingleChunkExperts - 1) / kSingleChunkExperts;
    const int chunk = lane;  // per-lane expert chunk
    const int e0 = chunk * kSingleChunkExperts;
    const unsigned int w = (chunk < nchunks) ? s_bits[e0 >> 5] : 0u;
    const unsigned int mine = (kSingleChunkExperts == 16)
                                  ? ((w >> (e0 & 31)) & 0xFFFFu)
                                  : w;
    const int cnt = __popc(mine);
    int pre = cnt;
    for (int ofs = 1; ofs < 32; ofs <<= 1) {
        const int other = __shfl_up_sync(0xFFFFFFFFu, pre, ofs);
        if (lane >= ofs) {
            pre += other;
        }
    }
    // pre is the inclusive prefix of selected-expert counts over chunks.
    const int base_cnt = pre - cnt;  // selected experts before this chunk
    if (chunk < nchunks) {
        const int avail = num_experts - e0;  // >= 1
        const int vec_quads =
            (avail >> 2) < kSingleQuads ? (avail >> 2) : kSingleQuads;
        // qb tracks the selected-count prefix at each quad start (running
        // add replaces the per-element popcount).
        int qb = base_cnt;
#pragma unroll
        for (int q = 0; q < kSingleQuads; ++q) {
            const int i = q * 4;
            if (i >= avail) {
                break;
            }
            const int b0 = (int)((mine >> i) & 1u);
            const int b1 = (int)((mine >> (i + 1)) & 1u);
            const int b2 = (int)((mine >> (i + 2)) & 1u);
            const int b3 = (int)((mine >> (i + 3)) & 1u);
            const int p1 = b0;
            const int p2 = b0 + b1;
            const int p3 = p2 + b2;
            if (q < vec_quads) {
                const int4 sel4 = {b0, b1, b2, b3};
                *reinterpret_cast<int4*>(expert_counts + e0 + i) = sel4;
                *reinterpret_cast<int4*>(expert_scatter_offsets + e0 + i) =
                    sel4;
                const int4 off4 = {qb * block_m, (qb + p1) * block_m,
                                   (qb + p2) * block_m, (qb + p3) * block_m};
                *reinterpret_cast<int4*>(expert_offsets + e0 + i) = off4;
            } else {
                const int bits[4] = {b0, b1, b2, b3};
                const int pfx[4] = {0, p1, p2, p3};
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    if (i + t >= avail) {
                        break;
                    }
                    expert_counts[e0 + i + t] = bits[t];
                    expert_scatter_offsets[e0 + i + t] = bits[t];
                    expert_offsets[e0 + i + t] = (qb + pfx[t]) * block_m;
                }
            }
            if (b0) {
                s_seg[qb] = e0 + i;
            }
            if (b1) {
                s_seg[qb + p1] = e0 + i + 1;
            }
            if (b2) {
                s_seg[qb + p2] = e0 + i + 2;
            }
            if (b3) {
                s_seg[qb + p3] = e0 + i + 3;
            }
            qb += p3 + b3;
        }
        // Tail of the unshifted window: offsets[num_experts] = extent = pre
        // (inclusive prefix over all chunks) only on the last chunk lane.
        if (chunk == nchunks - 1) {
            expert_offsets[num_experts] = extent;
        }
    }
    if (lane == 0) {
        num_tokens_post_padded[0] = extent;
    }
    __syncwarp(0xFFFFFFFFu);

    // Sentinel padding / real route / zero tail, slot-parallel over the warp.
    // Every selected segment is exactly block_m wide and holds one route at
    // slot 0, so segment/block ids come straight from the segment table.
    for (int s = lane; s < slots; s += 32) {
        if (s >= extent) {
            sorted_token_ids[s] = 0;
            continue;
        }
        const int seg = (int)fdiv_u32((unsigned int)s, fd_bm);
        // num_tokens == 1: the padding sentinel value is pairs == top_k.
        sorted_token_ids[s] =
            (s == seg * block_m) ? s_route[s_seg[seg]] : top_k;
    }

    // Per-block expert ids: each nonempty segment spans exactly one block.
    for (int b = lane; b < max_blocks; b += 32) {
        block_expert_ids[b] = (b < rounds) ? s_seg[b] : 0;
    }
}
// NOTE (round 4 experiment, reverted): a radix-histogram threshold selection
// (8-bit prefix refinement levels over the 64-bit route keys + gather + an
// in-register bitonic ordering sort across lanes) was measured ~2x SLOWER
// than the tournament above at the staged shapes (select 6.5 vs 3.66us at
// shape 3, 7.4 vs 3.9us at shape 4): each histogram level serializes 64-bit
// variable shifts, __match_any_sync aggregation, smem atomics and a per-lane
// 8-bin suffix scan (~1us/level on this latency-bound one-warp-per-block
// regime), and randn logits need ~3 levels, versus ~90cy x k serial rounds
// for the tournament. Correctness subtleties solved before reverting: the
// termination test must subtract the current level's above-count from k_rem
// BEFORE comparing against the bucket size, and the universal membership
// predicate is keys[c] >= (prefix << (64 - pl)).

__device__ __forceinline__ void token_softmax_finish(
    float* __restrict__ w_row, int lane, int top_k) {
    float sm = -INFINITY;
    for (int j = lane; j < top_k; j += 32) {
        sm = fmaxf(sm, w_row[j]);
    }
    for (int ofs = 16; ofs > 0; ofs >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFFu, sm, ofs);
        sm = fmaxf(sm, other);
    }
    float ssum = 0.0f;
    for (int j = lane; j < top_k; j += 32) {
        ssum += expf(w_row[j] - sm);
    }
    for (int ofs = 16; ofs > 0; ofs >>= 1) {
        ssum += __shfl_xor_sync(0xFFFFFFFFu, ssum, ofs);
    }
    const float inv = 1.0f / ssum;
    for (int j = lane; j < top_k; j += 32) {
        w_row[j] = expf(w_row[j] - sm) * inv;
    }
}

// Exclusive-scan helper: binary search for the expert whose block-aligned
// segment contains sorted-slot v over offs[1..num_experts].
__device__ __forceinline__ int segment_owner(const int* offs, int num_experts,
                                             int v) {
    int lo = 1, hi = num_experts;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (offs[mid] <= v) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo - 1;
}

// ----------------------- small/medium two-kernel path ----------------------

template <int CPL, bool SHARED>
__global__ void select_kernel(const float* __restrict__ logits,
                              float* __restrict__ topk_weights,
                              int* __restrict__ topk_ids,
                              int num_tokens,
                              int num_experts,
                              int top_k,
                              int routed_top_k) {
    const int token = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = (int)(threadIdx.x & 31);
    if (token < num_tokens) {
        token_select<CPL, SHARED>(logits + (long long)token * num_experts,
                                  topk_weights + (long long)token * top_k,
                                  topk_ids + (long long)token * top_k, lane,
                                  num_experts, top_k, routed_top_k);
    }
    pdl_trigger();
}

// Four-warps-per-token selection. Warp w owns experts [w*32*CPL4,
// (w+1)*32*CPL4): 4 warps cover the full CPL=16 band with CPL4=4 keys per
// lane, one warp per SM sub-partition scheduler. Each round's four local
// winners meet in shared memory behind a single double-buffered block
// barrier (slot j&1): a write to slot p at round j+2 is separated from
// every read of slot p in round j by barrier j+1. Selection math is
// IDENTICAL to the single-warp tournament -- the round max is the max of
// the four local maxima of keys below prev -- so the stable-tie output
// matches bit for bit. Register-softmax regime only (top_k <= 32, rounds
// owned by warp-0 threads).
template <int CPL4, bool SHARED>
__global__ void __launch_bounds__(128)
select_quad_kernel(const float* __restrict__ logits,
                   float* __restrict__ topk_weights,
                   int* __restrict__ topk_ids,
                   int num_tokens,
                   int num_experts,
                   int top_k,
                   int routed_top_k) {
    __shared__ unsigned long long s_win[2][4];  // [parity][warp]

    const int token = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;  // 0..127
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const float* row = logits + (long long)token * num_experts;
    float* w_row = topk_weights + (long long)token * top_k;
    int* id_row = topk_ids + (long long)token * top_k;
    const int routed_experts = num_experts - (SHARED ? 1 : 0);

    unsigned long long keys[CPL4];
#pragma unroll
    for (int c = 0; c < CPL4; ++c) {
        const int e = warp * (32 * CPL4) + lane + c * 32;
        keys[c] = (e < routed_experts) ? route_key(row[e], e) : 0ull;
    }

    unsigned long long prev = 0xFFFFFFFFFFFFFFFFull;
    float v_acc = -INFINITY;
    for (int j = 0; j < routed_top_k; ++j) {
        unsigned long long cand[CPL4];
#pragma unroll
        for (int c = 0; c < CPL4; ++c) {
            cand[c] = (keys[c] < prev) ? keys[c] : 0ull;
        }
#pragma unroll
        for (int step = CPL4 / 2; step > 0; step >>= 1) {
#pragma unroll
            for (int c = 0; c < step; ++c) {
                cand[c] = cand[c] > cand[c + step] ? cand[c] : cand[c + step];
            }
        }
        const unsigned long long best = cand[0];
        const unsigned int lane_hi = (unsigned int)(best >> 32);
        const unsigned int best_hi = __reduce_max_sync(0xFFFFFFFFu, lane_hi);
        const unsigned int tag =
            (lane_hi == best_hi) ? (unsigned int)best : 0u;
        const unsigned int best_lo = __reduce_max_sync(0xFFFFFFFFu, tag);
        if (lane == 0) {
            s_win[j & 1][warp] =
                ((unsigned long long)best_hi << 32) | best_lo;
        }
        __syncthreads();
        const unsigned long long win0 = s_win[j & 1][0];
        const unsigned long long win1 = s_win[j & 1][1];
        const unsigned long long win2 = s_win[j & 1][2];
        const unsigned long long win3 = s_win[j & 1][3];
        const unsigned long long win01 = win0 > win1 ? win0 : win1;
        const unsigned long long win23 = win2 > win3 ? win2 : win3;
        const unsigned long long win = win01 > win23 ? win01 : win23;
        prev = win;
        const unsigned int win_hi = (unsigned int)(win >> 32);
        const unsigned int win_lo = (unsigned int)(win & 0xFFFFFFFFu);
        // round owner lives in warp 0 (rounds <= 32 in this regime)
        if (tid == (j & 31)) {
            id_row[j] = (int)(0xFFFFFFFFu - win_lo);
            v_acc = float_from_sortable_key(win_hi);
        }
    }
    if (SHARED) {
        // rounds <= 32 here, so routed_top_k <= 31 and the owner lands in
        // warp 0 with no aliasing of an earlier round.
        const int j = routed_top_k;
        if (tid == (j & 31)) {
            id_row[j] = num_experts - 1;
            v_acc = row[num_experts - 1];
        }
    }

    // Register-distributed softmax: one round value per warp-0 thread.
    if (warp == 0) {
        const int rounds = routed_top_k + (SHARED ? 1 : 0);
        float m = v_acc;
#pragma unroll
        for (int ofs = 16; ofs > 0; ofs >>= 1) {
            const float other = __shfl_xor_sync(0xFFFFFFFFu, m, ofs);
            m = fmaxf(m, other);
        }
        const float e = (tid < rounds) ? expf(v_acc - m) : 0.0f;
        float sum = e;
#pragma unroll
        for (int ofs = 16; ofs > 0; ofs >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFFu, sum, ofs);
        }
        if (tid < rounds) {
            w_row[tid] = e / sum;
        }
    }
    pdl_trigger();
}

// Single block: expert histogram, then block_m-padded exclusive prefix
// scan (the >256-token companion of finish_kernel).
__global__ void __launch_bounds__(kReduceThreads)
reduce_kernel(const int* __restrict__ topk_ids,
              int* __restrict__ expert_counts,
              int* __restrict__ expert_offsets,
              int* __restrict__ num_tokens_post_padded,
              int* __restrict__ expert_scatter_offsets,
              int num_experts,
              int block_m,
              int pairs) {
    __shared__ int s_hist[kReduceMaxExperts];
    __shared__ int s_wsum[kReduceThreads / 32];

    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    for (int e = tid; e < num_experts; e += kReduceThreads) {
        s_hist[e] = 0;
    }
    // Global reads of topk_ids must wait for the predecessor; the smem-zero
    // prologue above runs overlapped with it under PDL.
    pdl_sync();
    __syncthreads();

    // One pass over the routes builds the expert histogram.
    for (int r = tid; r < pairs; r += kReduceThreads) {
        const int e = topk_ids[r];
        atomicAdd(s_hist + e, 1);
    }
    __syncthreads();

    // block_m-padded exclusive prefix scan over the histogram. Each thread
    // owns one expert index per 512-wide chunk; warp-shuffle scans plus a
    // one-warp scan of the warp sums keep it to three barriers per chunk.
    int carry = 0;
    for (int base = 0; base < num_experts; base += kReduceThreads) {
        const int e = base + tid;
        int count = 0;
        if (e < num_experts) {
            count = s_hist[e];
            expert_counts[e] = count;
            expert_scatter_offsets[e] = count;
        }
        int value = ((count + block_m - 1) / block_m) * block_m;
        int scan = value;
        for (int ofs = 1; ofs < 32; ofs <<= 1) {
            const int other = __shfl_up_sync(0xFFFFFFFFu, scan, ofs);
            if (lane >= ofs) {
                scan += other;
            }
        }
        if (lane == 31) {
            s_wsum[warp] = scan;
        }
        __syncthreads();
        if (warp == 0) {
            // All 32 lanes must execute the sync-shuffle; only the first
            // kReduceThreads/32 lanes carry meaningful warp sums.
            int w = (lane < kReduceThreads / 32) ? s_wsum[lane] : 0;
            for (int ofs = 1; ofs < kReduceThreads / 32; ofs <<= 1) {
                const int other = __shfl_up_sync(0xFFFFFFFFu, w, ofs);
                if (lane >= ofs) {
                    w += other;
                }
            }
            if (lane < kReduceThreads / 32) {
                s_wsum[lane] = w;
            }
        }
        __syncthreads();
        const int warp_base = (warp > 0) ? s_wsum[warp - 1] : 0;
        const int inclusive = carry + warp_base + scan;
        if (e < num_experts) {
            expert_offsets[e + 1] = inclusive;
        }
        carry += s_wsum[kReduceThreads / 32 - 1];
        // Same last-chunk skip as finish_kernel: no later chunk writes
        // s_wsum, and only thread-local carry feeds the loop-exit stores.
        if (base + kReduceThreads < num_experts) {
            __syncthreads();
        }
    }
    // carry is uniform across the block after the final chunk and equals the
    // total padded extent (works for multi-chunk num_experts > kReduceThreads).
    if (tid == 0) {
        expert_offsets[0] = 0;
        num_tokens_post_padded[0] = carry;
    }
    pdl_trigger();
}

// Parallel tail: one warp per expert. Each warp replays the route stream with
// ballot-scan to scatter its expert's routes in ascending route order, fills
// its sentinel padding, stamps its block ids, and (with the trailing loop)
// writes zeros/zeros for everything at/after the extent. Spreads ~9K output
// stores over the whole GPU instead of one SM.
__global__ void tail_kernel(const int* __restrict__ topk_ids,
                            const int* __restrict__ expert_counts,
                            const int* __restrict__ expert_offsets,
                            int* __restrict__ block_expert_ids,
                            int* __restrict__ sorted_token_ids,
                            int num_experts,
                            int block_m,
                            int pairs,
                            int max_blocks,
                            int slots) {
    const int e = (int)blockIdx.x;
    const int lane = (int)(threadIdx.x & 31);
    pdl_sync();
    const int start_off = expert_offsets[e];
    const int end_off = expert_offsets[e + 1];
    const int count = expert_counts[e];

    if (count > 0) {
        // Scatter: with r scanned in ascending order, the i-th route matching
        // this expert lands at start_off + i.
        int base = 0;
        for (int c0 = 0; c0 < pairs; c0 += 32) {
            const int r = c0 + lane;
            const bool match = r < pairs && topk_ids[r] == e;
            const unsigned int ball = __ballot_sync(0xFFFFFFFFu, match);
            if (match) {
                const int rank = base + __popc(ball & ((1u << lane) - 1u));
                sorted_token_ids[start_off + rank] = r;
            }
            base += __popc(ball);
        }
        for (int j = start_off + count + lane; j < end_off; j += 32) {
            sorted_token_ids[j] = pairs;  // padding sentinel
        }
        const int first_block = start_off / block_m;
        const int last_block = end_off / block_m;
        for (int b = first_block + lane; b < last_block; b += 32) {
            block_expert_ids[b] = e;
        }
    }

    // Zero everything at/after the extent: slots and per-block expert ids.
    const int extent = expert_offsets[num_experts];
    const int stride = num_experts * 32;
    for (int j = extent + e * 32 + lane; j < slots; j += stride) {
        sorted_token_ids[j] = 0;
    }
    const int used_blocks = extent / block_m;
    for (int b = used_blocks + e * 32 + lane; b < max_blocks; b += stride) {
        block_expert_ids[b] = 0;
    }
}

// -------------------- two-kernel finish (small/medium path) ----------------
//
// Replaces the reduce_kernel + scatter_bitmap_kernel pair: every block
// rebuilds the histogram, (expert x token) route bitmap and padded exclusive
// scan redundantly in its own shared memory (a ~1.5us prologue that runs in
// parallel across all blocks), then all blocks grid-stride the scatter and
// the sentinel/zero fills. Block 0 alone stores the histogram/scan outputs.
// This trades a small amount of redundant smem compute for one less kernel
// launch plus one less PDL dependency in the chain, which is worth ~3us on
// the 17..256-token shapes.
template <int TW>
__global__ void __launch_bounds__(kReduceThreads)
finish_kernel(const int* __restrict__ topk_ids,
              int* __restrict__ expert_counts,
              int* __restrict__ expert_offsets,
              int* __restrict__ num_tokens_post_padded,
              int* __restrict__ expert_scatter_offsets,
              int* __restrict__ block_expert_ids,
              int* __restrict__ sorted_token_ids,
              int num_experts,
              int top_k,
              int block_m,
              int pairs,
              int max_blocks,
              int slots,
              int twords,
              FastDivU32 fd_topk,
              FastDivU32 fd_bm) {
    __shared__ int s_hist[kReduceMaxExperts];
    __shared__ __align__(16) unsigned int s_bitmap[kSbBitmapInts];
    __shared__ int s_off[kReduceMaxExperts + 1];
    __shared__ int s_wsum[kReduceThreads / 32];
    __shared__ int s_blk[kSBlkCap];  // block -> expert map (guarded)
    // Scan-stamped per-word base slots (launched with
    // twords*num_experts*4 dynamic bytes, <= 32KB worst case).
    extern __shared__ int s_wslot[];

    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gtid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = (int)(gridDim.x * blockDim.x);
    const bool store_outputs = blockIdx.x == 0;
    // Word-major bitmap layout: token-word i, expert e lives at
    // i * ne_words + e. The scan phase's count popcounts (thread e reads its
    // whole column) are then bank-conflict-free for any num_experts; the
    // build atomics and scatter rank reads keep their original random-column
    // pattern.
    const int ne_words = num_experts;

    const int words = num_experts * TW;
    // Only the route bitmap needs zeroing: expert counts are recovered in
    // the scan phase as bitmap column popcounts (each token selects an
    // expert at most once, so the bit count equals the route count), which
    // removes one smem atomic per route from the redundant prologue.
    // Vectorized when the (16B-aligned) word count is a multiple of 4.
    if ((words & 3) == 0) {
        uint4* z = reinterpret_cast<uint4*>(s_bitmap);
        for (int w = tid; w < (words >> 2); w += kReduceThreads) {
            z[w] = make_uint4(0u, 0u, 0u, 0u);
        }
    } else {
        for (int w = tid; w < words; w += kReduceThreads) {
            s_bitmap[w] = 0u;
        }
    }
    // Global reads of topk_ids must wait for the predecessor; the smem-zero
    // prologue above runs overlapped with it under PDL.
    pdl_sync();
    __syncthreads();
    {
        // Peel the first two strided routes so both topk_ids loads are
        // issued back-to-back and their latencies overlap: pairs <=
        // 2*kReduceThreads covers every staged shape, and this prologue is
        // latency-bound, not bandwidth-bound. The generic loop below keeps
        // every larger input.
        const int r0 = tid;
        if (r0 < pairs) {
            const int r1 = r0 + kReduceThreads;
            const bool two = r1 < pairs;
            const int e0 = topk_ids[r0];
            const int e1 = two ? topk_ids[r1] : 0;
            const int t0 = (int)fdiv_u32((unsigned int)r0, fd_topk);
            atomicOr(s_bitmap + (t0 >> 5) * ne_words + e0, 1u << (t0 & 31));
            if (two) {
                const int t1 = (int)fdiv_u32((unsigned int)r1, fd_topk);
                atomicOr(s_bitmap + (t1 >> 5) * ne_words + e1,
                         1u << (t1 & 31));
            }
        }
    }
    for (int r = 2 * kReduceThreads + tid; r < pairs; r += kReduceThreads) {
        const int token = (int)fdiv_u32((unsigned int)r, fd_topk);
        const int e = topk_ids[r];
        atomicOr(s_bitmap + (token >> 5) * ne_words + e, 1u << (token & 31));
    }
    __syncthreads();

    // block_m-padded exclusive scan (loop over chunks for num_experts >
    // kReduceThreads). Offsets stay in smem; only block 0 touches the global
    // histogram/scan outputs. Every padded offset is a multiple of block_m,
    // so when max_blocks fits kSBlkCap thread e also stamps its own
    // [exclusive, inclusive) block span into s_blk (spans are disjoint and
    // contiguous), giving the fill phase a 1-load expert lookup instead of a
    // 9-level binary search.
    const bool use_blkmap = max_blocks <= kSBlkCap;
    int carry = 0;
    for (int base = 0; base < num_experts; base += kReduceThreads) {
        const int e = base + tid;
        int count = 0;
        int wcnt[TW];  // per-word column popcounts, reused by the slot stamp
        if (e < num_experts) {
#pragma unroll
            for (int i = 0; i < TW; ++i) {
                wcnt[i] = __popc(s_bitmap[i * ne_words + e]);
                count += wcnt[i];
            }
            s_hist[e] = count;
            if (store_outputs) {
                expert_counts[e] = count;
                expert_scatter_offsets[e] = count;
            }
        }
        const int value =
            (int)(fdiv_u32((unsigned int)(count + block_m - 1), fd_bm) *
                  (unsigned int)block_m);
        int scan = value;
        for (int ofs = 1; ofs < 32; ofs <<= 1) {
            const int other = __shfl_up_sync(0xFFFFFFFFu, scan, ofs);
            if (lane >= ofs) {
                scan += other;
            }
        }
        if (lane == 31) {
            s_wsum[warp] = scan;
        }
        __syncthreads();
        // Every warp redundantly scans the 16 warp sums in registers and
        // shuffles out its own base, removing the round-11 warp-0-only
        // serial section and its second barrier (single-chunk shapes then
        // need only the one barrier above; the conditional barrier below
        // still guards s_wsum against the next chunk's writers).
        int wsum = (lane < kReduceThreads / 32) ? s_wsum[lane] : 0;
        for (int ofs = 1; ofs < kReduceThreads / 32; ofs <<= 1) {
            const int other = __shfl_up_sync(0xFFFFFFFFu, wsum, ofs);
            if (lane >= ofs) {
                wsum += other;
            }
        }
        const int warp_base =
            (warp > 0) ? __shfl_sync(0xFFFFFFFFu, wsum, warp - 1) : 0;
        const int inclusive = carry + warp_base + scan;
        if (e < num_experts) {
            s_off[e + 1] = inclusive;
            // Stamp each bitmap word's base slot
            // s_off[e] + (# set bits in lower words of this column).
            int run = inclusive - value;
#pragma unroll
            for (int i = 0; i < TW; ++i) {
                s_wslot[i * ne_words + e] = run;
                run += wcnt[i];
            }
            if (use_blkmap) {
                // NOTE: a hoisted endpoint-divide (b0, b1) form measured
                // 0.1-0.18us SLOWER than this per-block divide-and-store
                // loop on both 512-expert cases; keep the o-loop.
                for (int o = inclusive - value; o < inclusive;
                     o += block_m) {
                    s_blk[o / block_m] = e;
                }
            }
            if (store_outputs) {
                expert_offsets[e + 1] = inclusive;
            }
        }
        carry += __shfl_sync(0xFFFFFFFFu, wsum, kReduceThreads / 32 - 1);
        // Only another chunk can clobber s_wsum underneath this chunk's
        // readers; the trailing __syncthreads after the loop covers the rest,
        // so the single-chunk case (num_experts <= kReduceThreads) skips it.
        if (base + kReduceThreads < num_experts) {
            __syncthreads();
        }
    }
    const int extent = carry;  // uniform across the block
    if (tid == 0) {
        s_off[0] = 0;
        if (store_outputs) {
            expert_offsets[0] = 0;
            num_tokens_post_padded[0] = extent;
        }
    }
    __syncthreads();

    // Scatter: rank inside the expert segment = number of earlier tokens
    // selecting the same expert -> ascending flattened route order.
    // Route-parallel: the per-route rank is a single table load (stamped by
    // the scan phase) plus one bitmask popcount of the route's own word.
    for (int r = gtid; r < pairs; r += total) {
        const int token = (int)fdiv_u32((unsigned int)r, fd_topk);
        const int e = topk_ids[r];
        const int w = (token >> 5) * ne_words + e;
        const int rank =
            s_wslot[w] + __popc(s_bitmap[w] & ((1u << (token & 31)) - 1u));
        sorted_token_ids[rank] = r;
    }

    // Sentinel padding + zero tail + per-block expert ids in one slot-parallel
    // pass; the scatter phase owns the in-extent non-pad slots. Every expert
    // offset is block_m-aligned, so the enclosing expert for a slot is exactly
    // s_blk[s/block_m] (stamped during the scan, ~1 store per expert) when the
    // map fits kSBlkCap; larger block counts keep the binary search. The
    // slot-parallel form keeps stores fully coalesced; expert-segment-
    // parallel variants (thread-per-expert and warp-per-expert) were measured
    // 0.6-3us SLOWER here (scattered half-full store transactions). Round 9's
    // flat-map attempt built its map with a separate redundant search pass
    // and measured ~0.1us slower at 544 blocks; stamping the same map from
    // the scan loop itself removes that build cost.
    for (int s = gtid; s < slots; s += total) {
        const int b = (int)fdiv_u32((unsigned int)s, fd_bm);
        const bool block_start = s == b * block_m;
        if (s >= extent) {
            sorted_token_ids[s] = 0;
            if (block_start) {
                block_expert_ids[b] = 0;
            }
            continue;
        }
        int e;
        if (use_blkmap) {
            e = s_blk[b];
        } else {
            // General path (max_blocks > kSBlkCap): 9-level binary search
            // for the enclosing expert segment.
            int lo = 1, hi = num_experts;
            while (lo < hi) {
                const int mid = (lo + hi) >> 1;
                if (s_off[mid] <= s) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            e = lo - 1;
        }
        if (s - s_off[e] >= s_hist[e]) {
            sorted_token_ids[s] = pairs;
        }
        if (block_start) {
            block_expert_ids[b] = e;
        }
    }
}

// ------------------------- generic large-input path ------------------------

__global__ void select_generic_kernel(const float* __restrict__ logits,
                                      float* __restrict__ topk_weights,
                                      int* __restrict__ topk_ids,
                                      int* __restrict__ expert_counts,
                                      unsigned int* __restrict__ route_bits,
                                      int num_tokens,
                                      int num_experts,
                                      int top_k,
                                      int routed_top_k,
                                      int has_shared,
                                      int words_per_token) {
    const int token = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = (int)(threadIdx.x & 31);
    if (token >= num_tokens) {
        return;
    }

    const float* row = logits + (long long)token * num_experts;
    unsigned int* bits_row = route_bits + (long long)token * words_per_token;
    for (int i = lane; i < words_per_token; i += 32) {
        bits_row[i] = 0u;
    }
    __syncwarp();

    const int routed_experts = num_experts - has_shared;
    float* w_row = topk_weights + (long long)token * top_k;
    int* id_row = topk_ids + (long long)token * top_k;

    unsigned long long prev = 0xFFFFFFFFFFFFFFFFull;
    for (int j = 0; j < routed_top_k; ++j) {
        unsigned long long best = 0ull;
        for (int e = lane; e < routed_experts; e += 32) {
            unsigned long long key = route_key(row[e], e);
            if (key < prev && key > best) {
                best = key;
            }
        }
        for (int ofs = 16; ofs > 0; ofs >>= 1) {
            unsigned long long other = __shfl_xor_sync(0xFFFFFFFFu, best, ofs);
            best = other > best ? other : best;
        }
        prev = best;
        const int expert =
            (int)(0xFFFFFFFFu - (unsigned int)(best & 0xFFFFFFFFull));
        if (lane == (j & 31)) {
            id_row[j] = expert;
            w_row[j] = row[expert];
            atomicAdd(expert_counts + expert, 1);
            atomicOr(bits_row + (expert >> 5), 1u << (expert & 31));
        }
    }
    if (has_shared) {
        const int expert = num_experts - 1;
        const int j = routed_top_k;
        if (lane == (j & 31)) {
            id_row[j] = expert;
            w_row[j] = row[expert];
            atomicAdd(expert_counts + expert, 1);
            atomicOr(bits_row + (expert >> 5), 1u << (expert & 31));
        }
    }

    token_softmax_finish(w_row, lane, top_k);
}

__global__ void scan_kernel(const int* __restrict__ expert_counts,
                            int* __restrict__ expert_offsets,
                            int* __restrict__ num_tokens_post_padded,
                            int* __restrict__ expert_scatter_offsets,
                            int* __restrict__ block_expert_ids,
                            int num_experts,
                            int block_m,
                            int max_blocks) {
    __shared__ int tile[kScanThreads];
    const int tid = (int)threadIdx.x;

    int carry = 0;
    for (int base = 0; base < num_experts; base += kScanThreads) {
        const int e = base + tid;
        int count = 0;
        if (e < num_experts) {
            count = expert_counts[e];
            expert_scatter_offsets[e] = count;
        }
        const int padded = ((count + block_m - 1) / block_m) * block_m;
        tile[tid] = padded;
        __syncthreads();
        for (int ofs = 1; ofs < kScanThreads; ofs <<= 1) {
            int v = (tid >= ofs) ? tile[tid - ofs] : 0;
            __syncthreads();
            tile[tid] += v;
            __syncthreads();
        }
        if (e < num_experts) {
            expert_offsets[e + 1] = carry + tile[tid];
        }
        carry += tile[kScanThreads - 1];
        __syncthreads();
    }
    if (tid == 0) {
        expert_offsets[0] = 0;
        num_tokens_post_padded[0] = carry;
    }

    const int extent = carry;  // uniform across the block
    for (int b = tid; b < max_blocks; b += kScanThreads) {
        const int v = b * block_m;
        block_expert_ids[b] =
            (v < extent) ? segment_owner(expert_offsets, num_experts, v) : 0;
    }
}

__global__ void scatter_kernel(const int* __restrict__ topk_ids,
                               const int* __restrict__ expert_counts,
                               const int* __restrict__ expert_offsets,
                               const unsigned int* __restrict__ route_bits,
                               int* __restrict__ sorted_token_ids,
                               int num_tokens,
                               int top_k,
                               int block_m,
                               int num_experts,
                               int words_per_token,
                               int pairs,
                               int slots) {
    const int r = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    if (r < pairs) {
        const int token = r / top_k;
        const int expert = topk_ids[r];
        const unsigned int mask = 1u << (expert & 31);
        const unsigned int* col = route_bits + (expert >> 5);
        int rank = 0;
        for (int t = 0; t < token; ++t) {
            rank += (col[(long long)t * words_per_token] & mask) ? 1 : 0;
        }
        sorted_token_ids[expert_offsets[expert] + rank] = r;
    }

    if (r < slots) {
        const int extent = expert_offsets[num_experts];
        if (r >= extent) {
            sorted_token_ids[r] = 0;
        } else {
            const int expert = segment_owner(expert_offsets, num_experts, r);
            if (r - expert_offsets[expert] >= expert_counts[expert]) {
                sorted_token_ids[r] = pairs;  // padding sentinel
            }
        }
    }
}

template <int CPL>
void launch_fused(const float* logits_ptr,
                  float* weights_ptr,
                  int* ids_ptr,
                  int* counts_ptr,
                  int* offsets_ptr,
                  int* extent_ptr,
                  int* scatter_ptr,
                  int* blocks_ptr,
                  int* sorted_ptr,
                  int num_tokens,
                  int num_experts,
                  int top_k,
                  int block_m,
                  int routed_top_k,
                  int has_shared,
                  int pairs,
                  int max_blocks,
                  int slots,
                  int twords,
                  cudaStream_t stream) {
    // Exact magic-division pairs for the two constructor-constant divisors
    // (built on the host once per forward; pure CPU math, no input
    // dependence).
    const FastDivU32 fd_topk = make_fast_div_u32((unsigned int)top_k);
    const FastDivU32 fd_bm = make_fast_div_u32((unsigned int)block_m);
    // One token: every other warp in the block would be idle and the phase
    // barriers dominate; a single warp runs the whole pipeline with
    // __syncwarp only. top_k <= 32 keeps one selection round per lane.
    if (num_tokens == 1 && top_k <= 32) {
        if (has_shared) {
            fused_single_kernel<CPL, true><<<1, 32, 0, stream>>>(
                logits_ptr, weights_ptr, ids_ptr, counts_ptr,
                offsets_ptr, extent_ptr, scatter_ptr, blocks_ptr, sorted_ptr,
                num_experts, top_k, block_m, routed_top_k, max_blocks, slots,
                fd_bm);
        } else {
            fused_single_kernel<CPL, false><<<1, 32, 0, stream>>>(
                logits_ptr, weights_ptr, ids_ptr, counts_ptr,
                offsets_ptr, extent_ptr, scatter_ptr, blocks_ptr, sorted_ptr,
                num_experts, top_k, block_m, routed_top_k, max_blocks, slots,
                fd_bm);
        }
        return;
    }
    const bool popc_ok = num_tokens <= block_m;
    // (Round-31 experiment -- REVERTED on measurement: widening the fused
    // path to 17..32 tokens with a 1024-thread instantiation measured 15.5us
    // on (32,512,8,16) vs the 7.0us select+finish pair; launch_bounds(1024)
    // caps the selector at ~55 regs/thread and the per-lane key/id
    // tournament spills. The multi-kernel dispatch for >16 tokens stays.)
    if (has_shared) {
        if (popc_ok) {
            fused_small_kernel<CPL, true, true><<<1, kFusedThreads, 0, stream>>>(
                logits_ptr, weights_ptr, ids_ptr, counts_ptr,
                offsets_ptr, extent_ptr, scatter_ptr, blocks_ptr, sorted_ptr,
                num_tokens, num_experts, top_k, block_m, routed_top_k, pairs,
                max_blocks, slots, twords, fd_topk, fd_bm);
        } else {
            fused_small_kernel<CPL, true, false><<<1, kFusedThreads, 0,
                                                   stream>>>(
                logits_ptr, weights_ptr, ids_ptr, counts_ptr,
                offsets_ptr, extent_ptr, scatter_ptr, blocks_ptr, sorted_ptr,
                num_tokens, num_experts, top_k, block_m, routed_top_k, pairs,
                max_blocks, slots, twords, fd_topk, fd_bm);
        }
    } else {
        if (popc_ok) {
            fused_small_kernel<CPL, false, true><<<1, kFusedThreads, 0,
                                                   stream>>>(
                logits_ptr, weights_ptr, ids_ptr, counts_ptr,
                offsets_ptr, extent_ptr, scatter_ptr, blocks_ptr, sorted_ptr,
                num_tokens, num_experts, top_k, block_m, routed_top_k, pairs,
                max_blocks, slots, twords, fd_topk, fd_bm);
        } else {
            fused_small_kernel<CPL, false, false><<<1, kFusedThreads, 0,
                                                    stream>>>(
                logits_ptr, weights_ptr, ids_ptr, counts_ptr,
                offsets_ptr, extent_ptr, scatter_ptr, blocks_ptr, sorted_ptr,
                num_tokens, num_experts, top_k, block_m, routed_top_k, pairs,
                max_blocks, slots, twords, fd_topk, fd_bm);
        }
    }
}

template <int CPL>
void launch_small(const float* logits_ptr,
                  float* weights_ptr,
                  int* ids_ptr,
                  int* counts_ptr,
                  int* offsets_ptr,
                  int* extent_ptr,
                  int* scatter_ptr,
                  int* blocks_ptr,
                  int* sorted_ptr,
                  bool use_finish,
                  int num_tokens,
                  int num_experts,
                  int top_k,
                  int block_m,
                  int routed_top_k,
                  int has_shared,
                  int words_per_token,
                  int pairs,
                  int max_blocks,
                  int slots,
                  int twords,
                  cudaStream_t stream) {
    const int select_blocks =
        (int)((num_tokens * 32 + kSelectThreads - 1) / kSelectThreads);
    // CPL=16 expert band with register-softmax top_k: four warps share
    // one token's tournament (select_quad_kernel). Everything else keeps
    // warp-per-token selection.
    const bool split_ok = CPL == 16 && top_k <= 32 && num_tokens > 0;
    if (split_ok) {
        if (has_shared) {
            select_quad_kernel<4, true><<<num_tokens, 128, 0, stream>>>(
                logits_ptr, weights_ptr, ids_ptr, num_tokens,
                num_experts, top_k, routed_top_k);
        } else {
            select_quad_kernel<4, false><<<num_tokens, 128, 0, stream>>>(
                logits_ptr, weights_ptr, ids_ptr, num_tokens,
                num_experts, top_k, routed_top_k);
        }
    } else if (has_shared) {
        select_kernel<CPL, true><<<select_blocks, kSelectThreads, 0, stream>>>(
            logits_ptr, weights_ptr, ids_ptr, num_tokens, num_experts,
            top_k, routed_top_k);
    } else {
        select_kernel<CPL, false><<<select_blocks, kSelectThreads, 0, stream>>>(
            logits_ptr, weights_ptr, ids_ptr, num_tokens, num_experts,
            top_k, routed_top_k);
    }
    // NOTE: a single-block "finish everything" kernel was measured strictly
    // slower than the reduce+tail/scatter pair at every staged shape across
    // three independent implementations (rounds 2-3, up to 17us at shape 3):
    // serial warp sections dominate once grid parallelism is removed. The
    // finish_kernel below keeps grid parallelism (every block redundantly
    // rebuilds the smem histogram/bitmap/scan, then grid-strides the fills)
    // and replaces reduce+scatter with a single launch chain link.
    if (use_finish) {
        // finish_kernel's static smem is ~45KB, so the word-slot dynamic
        // table needs the >48KB opt-in exactly once per process.
        static const bool smem_opt_in = [] {
            constexpr int kDynMax = kSbMaxTwords * kReduceMaxExperts * 4;
            for (auto kern :
                 {finish_kernel<1>, finish_kernel<2>, finish_kernel<4>,
                  finish_kernel<8>}) {
                const cudaError_t err = cudaFuncSetAttribute(
                    kern, cudaFuncAttributeMaxDynamicSharedMemorySize, kDynMax);
                if (err != cudaSuccess) {
                    throw std::runtime_error(
                        std::string("alphamoe_router: smem opt-in failed: ") +
                        cudaGetErrorString(err));
                }
            }
            return true;
        }();
        (void)smem_opt_in;
        // Measured grid sweep: finish_kernel plateaus at ~16 blocks (the
        // redundant prologue runs in parallel and the fills are strip-thin);
        // keep a floor of 16 and cap at 64 to bound redundant traffic.
        const int work = pairs > slots ? pairs : slots;
        int blocks = (work + kReduceThreads - 1) / kReduceThreads;
        if (blocks < 16) {
            blocks = 16;
        }
        if (blocks > 64) {
            blocks = 64;
        }
        // finish is templated on the route-bitmap word count so the popcount
        // rank/count loops unroll and drop their dynamic bounds checks.
        // Exact magic-division pairs for the two constructor-constant
        // divisors; built on the host once per forward (pure CPU math, no
        // input dependence).
        const FastDivU32 fd_topk = make_fast_div_u32((unsigned int)top_k);
        const FastDivU32 fd_bm = make_fast_div_u32((unsigned int)block_m);
        switch (twords) {
        case 1:
            launch_pdl(finish_kernel<1>, blocks, kReduceThreads,
                       (size_t)1 * num_experts * 4, stream,
                       ids_ptr, counts_ptr, offsets_ptr, extent_ptr,
                       scatter_ptr, blocks_ptr, sorted_ptr, num_experts, top_k,
                       block_m, pairs, max_blocks, slots, twords, fd_topk,
                       fd_bm);
            break;
        case 2:
            launch_pdl(finish_kernel<2>, blocks, kReduceThreads,
                       (size_t)2 * num_experts * 4, stream,
                       ids_ptr, counts_ptr, offsets_ptr, extent_ptr,
                       scatter_ptr, blocks_ptr, sorted_ptr, num_experts, top_k,
                       block_m, pairs, max_blocks, slots, twords, fd_topk,
                       fd_bm);
            break;
        case 3:
        case 4:
            launch_pdl(finish_kernel<4>, blocks, kReduceThreads,
                       (size_t)4 * num_experts * 4, stream,
                       ids_ptr, counts_ptr, offsets_ptr, extent_ptr,
                       scatter_ptr, blocks_ptr, sorted_ptr, num_experts, top_k,
                       block_m, pairs, max_blocks, slots, twords, fd_topk,
                       fd_bm);
            break;
        default:
            launch_pdl(finish_kernel<8>, blocks, kReduceThreads,
                       (size_t)8 * num_experts * 4, stream,
                       ids_ptr, counts_ptr, offsets_ptr, extent_ptr,
                       scatter_ptr, blocks_ptr, sorted_ptr, num_experts, top_k,
                       block_m, pairs, max_blocks, slots, twords, fd_topk,
                       fd_bm);
            break;
        }
    } else {
        launch_pdl(reduce_kernel, 1, kReduceThreads, 0, stream, ids_ptr,
                   counts_ptr, offsets_ptr, extent_ptr, scatter_ptr,
                   num_experts, block_m, pairs);
        launch_pdl(tail_kernel, num_experts, 32, 0, stream, ids_ptr,
                   counts_ptr, offsets_ptr, blocks_ptr, sorted_ptr,
                   num_experts, block_m, pairs, max_blocks, slots);
    }
}

}  // namespace flashinfer::fused_moe

namespace flashinfer::fused_moe {

// ---------------------------------------------------------------------------
// Host-side geometry + launch interface (raw pointers, caller-allocated).
// ---------------------------------------------------------------------------

struct AlphaMoeRouterParams {
  int num_tokens = 0;
  int num_experts = 0;
  int top_k = 0;
  int block_m = 0;
  int has_shared = 0;
  int routed_top_k = 0;
  int routed_experts = 0;
  int64_t pairs = 0;
  int64_t max_blocks = 0;
  int64_t slots = 0;
  int64_t words_per_token = 0;
  int64_t twords = 0;
};

// Pure arithmetic; every field depends only on the shape configuration, so
// the geometry is fixed under CUDA graph capture for a fixed captured shape.
inline AlphaMoeRouterParams make_alphamoe_router_params(int num_tokens,
                                                        int num_experts,
                                                        int top_k,
                                                        int block_m,
                                                        int has_shared) {
  if (num_experts < 2) {
    throw std::invalid_argument("alphamoe_router: num_experts must be >= 2");
  }
  if (top_k < 1) {
    throw std::invalid_argument("alphamoe_router: top_k must be >= 1");
  }
  if (block_m < 1) {
    throw std::invalid_argument("alphamoe_router: block_m must be >= 1");
  }
  if (num_tokens < 1) {
    throw std::invalid_argument("alphamoe_router: num_tokens must be >= 1");
  }
  AlphaMoeRouterParams p;
  p.num_tokens = num_tokens;
  p.num_experts = num_experts;
  p.top_k = top_k;
  p.block_m = block_m;
  p.has_shared = has_shared ? 1 : 0;
  p.routed_top_k = top_k - p.has_shared;
  p.routed_experts = num_experts - p.has_shared;
  if (p.routed_top_k < 0 || p.routed_top_k > p.routed_experts) {
    throw std::invalid_argument(
        "alphamoe_router: invalid top_k/num_experts for shared-expert "
        "configuration");
  }
  p.pairs = static_cast<int64_t>(num_tokens) * top_k;
  const int64_t nonempty = static_cast<int64_t>(num_experts) < p.pairs
                               ? num_experts
                               : p.pairs;
  p.max_blocks = nonempty + (p.pairs - nonempty) / block_m;
  p.slots = p.max_blocks * block_m;
  p.words_per_token = (static_cast<int64_t>(num_experts) + 31) / 32;
  p.twords = (static_cast<int64_t>(num_tokens) + 31) / 32;
  return p;
}

// Number of int32 scratch elements the generic path needs (zero otherwise).
inline int64_t alphamoe_router_scratch_ints(const AlphaMoeRouterParams& p) {
  return p.num_experts > kReduceMaxExperts ? p.num_tokens * p.words_per_token
                                           : 0;
}

// Launch the routing pipeline. All outputs must be sized per
// make_alphamoe_router_params. `scratch` must hold
// alphamoe_router_scratch_ints(p) int32 elements when that is non-zero.
inline void alphamoe_router_forward(const AlphaMoeRouterParams& p,
                                    const float* logits_ptr,
                                    float* topk_weights_ptr, int* topk_ids_ptr,
                                    int* expert_counts_ptr,
                                    int* expert_offsets_ptr,
                                    int* expert_scatter_offsets_ptr,
                                    int* num_tokens_post_padded_ptr,
                                    int* block_expert_ids_ptr,
                                    int* sorted_token_ids_ptr, int* scratch_ptr,
                                    cudaStream_t stream) {
  const int num_tokens = p.num_tokens;
  const int num_experts = p.num_experts;
  const int top_k = p.top_k;
  const int block_m = p.block_m;
  const int has_shared = p.has_shared;
  const int routed_top_k = p.routed_top_k;
  const int pairs = static_cast<int>(p.pairs);
  const int max_blocks = static_cast<int>(p.max_blocks);
  const int slots = static_cast<int>(p.slots);
  const int twords = static_cast<int>(p.twords);
  const int words_per_token = static_cast<int>(p.words_per_token);

  const bool small_ok = num_experts <= kReduceMaxExperts;
  const int64_t per_lane =
      (static_cast<int64_t>(p.routed_experts) + 31) / 32;

  if (small_ok) {
    const bool fused_ok = num_experts <= kFusedMaxExperts &&
                          pairs <= kFusedPairsCap &&
                          num_tokens <= kFusedThreads / 32;
    if (fused_ok) {
      if (per_lane <= 4) {
        launch_fused<4>(logits_ptr, topk_weights_ptr, topk_ids_ptr,
                        expert_counts_ptr, expert_offsets_ptr,
                        num_tokens_post_padded_ptr,
                        expert_scatter_offsets_ptr, block_expert_ids_ptr,
                        sorted_token_ids_ptr, num_tokens, num_experts, top_k,
                        block_m, routed_top_k, has_shared, pairs, max_blocks,
                        slots, twords, stream);
      } else if (per_lane <= 8) {
        launch_fused<8>(logits_ptr, topk_weights_ptr, topk_ids_ptr,
                        expert_counts_ptr, expert_offsets_ptr,
                        num_tokens_post_padded_ptr,
                        expert_scatter_offsets_ptr, block_expert_ids_ptr,
                        sorted_token_ids_ptr, num_tokens, num_experts, top_k,
                        block_m, routed_top_k, has_shared, pairs, max_blocks,
                        slots, twords, stream);
      } else {
        launch_fused<16>(logits_ptr, topk_weights_ptr, topk_ids_ptr,
                         expert_counts_ptr, expert_offsets_ptr,
                         num_tokens_post_padded_ptr,
                         expert_scatter_offsets_ptr, block_expert_ids_ptr,
                         sorted_token_ids_ptr, num_tokens, num_experts,
                         top_k, block_m, routed_top_k, has_shared, pairs,
                         max_blocks, slots, twords, stream);
      }
      return;
    }
    const bool use_finish = twords <= kSbMaxTwords;
    if (per_lane <= 4) {
      launch_small<4>(logits_ptr, topk_weights_ptr, topk_ids_ptr,
                      expert_counts_ptr, expert_offsets_ptr,
                      num_tokens_post_padded_ptr, expert_scatter_offsets_ptr,
                      block_expert_ids_ptr, sorted_token_ids_ptr, use_finish,
                      num_tokens, num_experts, top_k, block_m, routed_top_k,
                      has_shared, words_per_token, pairs, max_blocks, slots,
                      twords, stream);
    } else if (per_lane <= 8) {
      launch_small<8>(logits_ptr, topk_weights_ptr, topk_ids_ptr,
                      expert_counts_ptr, expert_offsets_ptr,
                      num_tokens_post_padded_ptr, expert_scatter_offsets_ptr,
                      block_expert_ids_ptr, sorted_token_ids_ptr, use_finish,
                      num_tokens, num_experts, top_k, block_m, routed_top_k,
                      has_shared, words_per_token, pairs, max_blocks, slots,
                      twords, stream);
    } else if (per_lane <= 16) {
      launch_small<16>(logits_ptr, topk_weights_ptr, topk_ids_ptr,
                       expert_counts_ptr, expert_offsets_ptr,
                       num_tokens_post_padded_ptr, expert_scatter_offsets_ptr,
                       block_expert_ids_ptr, sorted_token_ids_ptr, use_finish,
                       num_tokens, num_experts, top_k, block_m, routed_top_k,
                       has_shared, words_per_token, pairs, max_blocks, slots,
                       twords, stream);
    } else {
      launch_small<32>(logits_ptr, topk_weights_ptr, topk_ids_ptr,
                       expert_counts_ptr, expert_offsets_ptr,
                       num_tokens_post_padded_ptr, expert_scatter_offsets_ptr,
                       block_expert_ids_ptr, sorted_token_ids_ptr, use_finish,
                       num_tokens, num_experts, top_k, block_m, routed_top_k,
                       has_shared, words_per_token, pairs, max_blocks, slots,
                       twords, stream);
    }
    return;
  }

  // Generic path: grid-parallel select with global histogram atomics +
  // single-block scan + grid scatter over the route bitmap in scratch.
  unsigned int* bits_ptr = reinterpret_cast<unsigned int*>(scratch_ptr);
  cudaMemsetAsync(expert_counts_ptr, 0, sizeof(int) * num_experts, stream);

  const int select_blocks =
      static_cast<int>((num_tokens * 32 + kSelectThreads - 1) /
                       kSelectThreads);
  select_generic_kernel<<<select_blocks, kSelectThreads, 0, stream>>>(
      logits_ptr, topk_weights_ptr, topk_ids_ptr, expert_counts_ptr, bits_ptr,
      num_tokens, num_experts, top_k, routed_top_k, has_shared,
      words_per_token);

  scan_kernel<<<1, kScanThreads, 0, stream>>>(
      expert_counts_ptr, expert_offsets_ptr, num_tokens_post_padded_ptr,
      expert_scatter_offsets_ptr, block_expert_ids_ptr, num_experts, block_m,
      max_blocks);

  const int64_t work = pairs > slots ? pairs : slots;
  const int scatter_blocks =
      static_cast<int>((work + kScatterThreads - 1) / kScatterThreads);
  scatter_kernel<<<scatter_blocks, kScatterThreads, 0, stream>>>(
      topk_ids_ptr, expert_counts_ptr, expert_offsets_ptr, bits_ptr,
      sorted_token_ids_ptr, num_tokens, top_k, block_m, num_experts,
      words_per_token, pairs, slots);
}

}  // namespace flashinfer::fused_moe
