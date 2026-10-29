/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Lamport transport of the NVLink one-sided MoE all-to-all (flashinfer.moe_ep
// NVLinkOneSidedAlltoAll) for small steps: one dispatch and one combine kernel, no completion
// flags and no fences between data and its readiness.
//
// Lamport arrival. Every landing row a peer writes into is armed with a sentinel word that no
// payload word takes (senders rewrite the few words that could, see canonicalize_*). A receiver
// polls the data itself: a 16-byte vector has arrived when none of its four 32-bit words is the
// sentinel (32-bit stores are single-copy atomic, so a torn vector still shows one). The
// receiver copies the vector out and re-arms it; it is the vector's only reader.
//
// Landing rows of a rank's workspace, row (src, token) = token * ep_size + src (the same rows in
// every round, a token slot's rows adjacent):
//   payload  hidden states in wire format        sentinel 0xFFFFFFFF
//   scales   block scales in wire format         sentinel 0xFFFFFFFF
//   weights  top-k routing weights (optional)    sentinel 0xFFFFFFFF
//   ids      top-k expert ids                    sentinel 0x88888888   two sets
//   combine  partial output of this rank's token from rank src, row src * max_tokens_per_rank +
//            token (this step's tokens per rank)
// A source stores its ids row (src, token) on EVERY alive peer, all invalid_expert_id where the
// peer is no target: the ids row tells that slot `token` arrived and whether it holds a token.
//
// Re-arming. A dispatch ends only after every alive peer's ids rows of the round arrived, so no
// rank starts round n + 2 before every peer finished round n. Payload, scale and weights rows go
// to targets only, and the sender's combine waits for every target's partial: a row is rewritten
// only after its reader re-armed it, so one set suffices. Ids rows also reach non-targets, which
// may still be in round n when the sender writes round n + 1, so they alternate two sets by an
// epoch in device memory (CUDA graphs replay it) that the last combine block advances; the
// combine re-arms them. A combine row is written in round n only by a rank that received the
// reader's round-n ids rows, sent after the reader's previous combine re-armed the row.
//
// Dispatch, grid max_tokens_per_rank x blocks_per_token (a block: one token slot, one chunk of
// kNumThreads 16-byte vectors of the wire row, payload vectors first, then scale vectors):
//   D1 route     warp 0 maps the token's top-k ids to target ranks (each rank once, at its first
//                slot; masked ranks dropped) and stages the ids and weights rows; thread 0 reads
//                the epoch.
//   D2 push      each thread stores its vector into the landing row (ep_rank, token) of every
//                target; for this rank itself it goes straight into the receive buffer.
//   D3 ids       the chunk-0 block stores the ids row to every alive peer, the weights to targets,
//                its threads taking the (peer, vector) stores in turn.
//   D4 poll ids  every source's ids row of slot `token` -> recv_source_mask; warp 0 lists the
//                sources that sent a token.
//   D5 receive   poll, copy out and re-arm this thread's vector of every source that sent one,
//                8 sources of the slot's source list per round.
//   D6 weights   in the same rounds, the chunk-0 block does the same for the weights rows.
// Rank loops walk the set bits of a rank mask in ascending rank order (pop_lowest_rank).
// Combine, same grid:
//   C1 sources   warp 0 lists the token's target ranks in top-k slot order.
//   C2 push      the expert output of row (src, token) goes to src's combine row (ep_rank, token).
//   C3 reduce    poll this rank's partials 8 sources at a time, re-arm them and add them in the
//                order of the fence combine (vectorized_combine_impl), then round to bf16.
//   C4 tail      the chunk-0 block re-arms the slot's ids rows; the last block advances the epoch.
// The only fence is the device-scope one before the combine's finished-block count.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

// The kernels are built for compute capability 10.0 and 10.3, where the transport is validated.
// Other architectures get kernels that trap, and moe_a2a_lamport_kernels_available() is false.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030)
#define MOE_A2A_LAMPORT_COMPILE 1
#else
#define MOE_A2A_LAMPORT_COMPILE 0
#endif

namespace flashinfer {
namespace moe_alltoall_lamport {

constexpr int kMaxRanks = 64;  // the active-rank mask is one 64-bit word
constexpr int kMaxTopK = 22;   // the NVLink one-sided kernels' bound
constexpr int kNumThreads = 128;
// Resident dispatch blocks per SM (a register bound); the default token limit keeps the dispatch
// grid within one wave of them.
constexpr int kDispatchBlocksPerSM = 6;
// Landing rows per source rank of a workspace, at most (bounds memory, not code).
constexpr int kMaxTokensPerRank = 512;

// 16-byte vectors of an ids or weights row (top_k 4-byte words, padded).
__host__ __device__ constexpr int topk_row_vectors(int top_k) { return (top_k + 3) / 4; }

constexpr uint32_t kIdsSentinel = 0x88888888u;
constexpr uint32_t kDataSentinel = 0xFFFFFFFFu;
// NVFP4 data words travel XORed with this, so that the one word that could equal the sentinel
// after canonicalisation (eight -0 nibbles) cannot.
constexpr uint32_t kNvfp4WireXor = 0x77777777u;

// Payload formats of the landing rows.
enum PayloadFormat : int {
  kBf16 = 0,
  kMxfp8 = 1,        // e4m3 [T, H] + UE8M0 scales [T, H / 32] (16-bit lanes on the wire)
  kFp8Block128 = 2,  // e4m3 [T, H] + fp32 scales [T, H / 128]
  kNvfp4 = 3,        // e2m1 x2 [T, H / 2] + e4m3 scales [T, H / 16]
  kNumPayloadFormats
};
// Kernel families by the payload's element width (scales are handled at run time).
enum PayloadKind : int { kKindBf16 = 0, kKindFp8 = 1, kKindNvfp4 = 2 };

// int32 words of the round state.
enum RoundStateWord : int {
  kEpoch = 0,
  kFinishedCombineBlocks = 1,
  kTimeoutReported = 2,
  kRoundStateWords = 4
};

// Landing regions of every rank's workspace slice (byte offsets).
struct LandingLayout {
  int64_t payload_offset;
  int64_t scale_offset;
  int64_t weights_offset;
  int64_t ids_offset;  // ids set 0; set 1 follows at + ids_set_bytes
  int64_t ids_set_bytes;
  int64_t combine_offset;
};

struct DispatchParams {
  // Every rank's workspace slice as mapped here; rank_workspaces[ep_rank] is this rank's.
  uint8_t* rank_workspaces[kMaxRanks];
  LandingLayout landing;
  // This rank's tokens.
  uint8_t const* hidden_states;           // [local_num_tokens, payload_row_bytes]
  uint8_t const* hidden_states_scale;     // [local_num_tokens, scale_row_bytes]; null for BF16
  int32_t const* token_selected_experts;  // [local_num_tokens, top_k]
  uint32_t const* topk_weights;           // [local_num_tokens, top_k] 4-byte words
  // Receive buffers, row (src, token) = src * max_tokens_per_rank + token.
  uint8_t* recv_hidden_states;
  uint8_t* recv_hidden_states_scale;
  int32_t* recv_topk_ids;
  uint32_t* recv_topk_weights;
  // Round state shared with the combine.
  int32_t* topk_target_ranks;  // [max_tokens_per_rank, top_k]: rank of slot k at its first
                               // occurrence in the token's top-k, else -1
  uint64_t* recv_source_mask;  // [max_tokens_per_rank]: bit s = rank s sent its token slot here
  int32_t* round_state;        // RoundStateWord
  int ep_size;
  int ep_rank;
  int num_experts;
  int experts_per_rank;  // num_experts / ep_size (an even split)
  int top_k;
  int local_num_tokens;
  int max_tokens_per_rank;  // of this step: the receive buffers' rows per source
  int has_topk_weights;     // the step carries routing weights (every rank alike)
  int mx_scales;            // MXFP8: UE8M0 scale bytes travel as 16-bit lanes
  int payload_row_bytes;    // landing and receive
  int scale_row_bytes;      // input and receive scale row
  int wire_scale_row_bytes;
  int payload_vectors;  // 16-byte vectors per payload row
  int wire_scale_vectors;
  int blocks_per_token;
  int32_t invalid_expert_id;
  uint64_t active_rank_mask;  // bit r: rank r takes part (rank-mask kernels only)
  int64_t timeout_cycles;
};

struct CombineParams {
  uint8_t* rank_workspaces[kMaxRanks];
  LandingLayout landing;
  uint8_t const* expert_output;  // [ep_size * max_tokens_per_rank, hidden_size] bf16, row (src, t)
  uint8_t* output;               // [local_num_tokens, hidden_size] bf16
  int32_t const* topk_target_ranks;
  uint64_t const* recv_source_mask;
  int32_t* round_state;
  int ep_size;
  int ep_rank;
  int top_k;
  int local_num_tokens;
  int max_tokens_per_rank;
  int hidden_size;
  int row_vectors;  // 16-byte vectors per bf16 row
  int blocks_per_token;
  uint64_t active_rank_mask;
  int64_t timeout_cycles;
};

#if MOE_A2A_LAMPORT_COMPILE
// ================================================================================ PTX helpers
__device__ __forceinline__ uint4 load_volatile(void const* ptr) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(ptr)
               : "memory");
  return v;
}

__device__ __forceinline__ int load_volatile_int(int const* ptr) {
  int v;
  asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(v) : "l"(ptr) : "memory");
  return v;
}

__device__ __forceinline__ void store_volatile(void* ptr, uint4 v) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(v.x), "r"(v.y),
               "r"(v.z), "r"(v.w)
               : "memory");
}

// Programmatic dependent launch: every kernel waits for its predecessor before it touches a
// buffer or the epoch, and lets its successor launch once its own stores are issued.
__device__ __forceinline__ void wait_for_previous_kernel() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}

__device__ __forceinline__ void allow_next_kernel() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

__device__ __forceinline__ uint4 splat(uint32_t word) { return make_uint4(word, word, word, word); }

__device__ __forceinline__ bool has_arrived(uint4 v, uint32_t sentinel) {
  return v.x != sentinel && v.y != sentinel && v.z != sentinel && v.w != sentinel;
}

// Row (source, token) of the landing regions: a token slot's rows from every source are adjacent.
__device__ __forceinline__ int64_t landing_row(int ep_size, int source, int token) {
  return static_cast<int64_t>(token) * ep_size + source;
}

// Polls the landing vectors whose bits are set in `pending` until each has arrived, every load in
// flight before any check; address(i) is vector i's address, and before_each_spin() runs at the
// start of every spin (e.g. to re-read address(i)'s inputs from shared memory, all before the
// first load). Returns the bits still pending when the wait timed out (0: all arrived).
template <int NUM_VECTORS, typename BeforeEachSpin, typename Address>
__device__ __forceinline__ uint32_t poll_until_arrived(uint32_t pending, uint32_t sentinel,
                                                       BeforeEachSpin const& before_each_spin,
                                                       Address const& address,
                                                       uint4 (&vectors)[NUM_VECTORS],
                                                       long long wait_start,
                                                       int64_t timeout_cycles) {
  for (int spins = 1; pending; ++spins) {
    before_each_spin();
#pragma unroll
    for (int i = 0; i < NUM_VECTORS; ++i)
      if ((pending >> i) & 1u) vectors[i] = load_volatile(address(i));
#pragma unroll
    for (int i = 0; i < NUM_VECTORS; ++i)
      if (((pending >> i) & 1u) && has_arrived(vectors[i], sentinel)) pending &= ~(1u << i);
    if (pending && spins % 64 == 0 && clock64() - wait_start > timeout_cycles) break;
  }
  return pending;
}

// A peer that never arrives: report once per launch, then trap (the CUDA context is lost, as in
// the fence kernels). [[noreturn]]: callers keep no register live
// across the call.
[[noreturn]] static __device__ __noinline__ void report_and_trap(int32_t* round_state, int ep_rank,
                                                                 char const* what) {
  if (atomicExch(round_state + kTimeoutReported, 1) == 0)
    printf("Lamport transport: rank %d %s\n", ep_rank, what);
  __threadfence_system();
  __trap();
}

// The rank mask is passed by value, as the fence kernels' is. Nothing is stored to or polled
// from a masked rank; a route to it is dropped like a repeated one.
template <bool ENABLE_RANK_MASK>
__device__ __forceinline__ bool is_rank_alive(uint64_t active_rank_mask, int rank) {
  if constexpr (ENABLE_RANK_MASK) {
    return (active_rank_mask >> rank) & 1ull;
  } else {
    return true;
  }
}

// Removes the lowest rank from a rank mask and returns it: successive calls walk the ranks in
// ascending order, at a cost per rank in the mask rather than per rank of the EP group.
__device__ __forceinline__ int pop_lowest_rank(uint64_t& ranks) {
  int const rank = __ffsll(static_cast<long long>(ranks)) - 1;
  ranks &= ranks - 1;
  return rank;
}

// Row vector `vec` (top-k slots 4 vec .. 4 vec + 3) into a receive row of top_k words.
__device__ __forceinline__ void store_topk_words(uint32_t* row, int top_k, int vec, uint4 v) {
  uint32_t const words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int w = 0; w < 4; ++w)
    if (4 * vec + w < top_k) row[4 * vec + w] = words[w];
}

// ============================================== sender-side rewrites: no wire word is a sentinel
// BF16 0xFFFFFFFF (two NaNs) -> 0x7FFF7FFF; FP8 data and fp32 / e4m3 scale or weight words
// 0xFFFFFFFF -> 0x7FFFFFFF (NaN stays NaN); an id equal to kIdsSentinel (an invalid id) ->
// invalid_expert_id. MXFP8 scales travel as 16-bit lanes 0x00XX, which never match.
__device__ __forceinline__ uint32_t canonicalize_bf16x2(uint32_t word) {
  return word == kDataSentinel ? 0x7FFF7FFFu : word;
}

__device__ __forceinline__ uint32_t canonicalize_word(uint32_t word) {
  return word == kDataSentinel ? 0x7FFFFFFFu : word;
}

__device__ __forceinline__ uint32_t canonicalize_expert_id(uint32_t id, uint32_t invalid_id) {
  return id == kIdsSentinel ? invalid_id : id;
}

// NVFP4: every -0 nibble (1000b) becomes +0; the wire word is this XOR kNvfp4WireXor.
__device__ __forceinline__ uint32_t canonicalize_fp4x8(uint32_t word) {
  uint32_t const magnitude = word & 0x77777777u;
  uint32_t const nonzero = (magnitude | (magnitude >> 1) | (magnitude >> 2)) & 0x11111111u;
  return word & (0x77777777u | (nonzero << 3));
}

// Wire <-> raw NVFP4 vector (an involution).
__device__ __forceinline__ uint4 nvfp4_wire_xor(uint4 v) {
  return make_uint4(v.x ^ kNvfp4WireXor, v.y ^ kNvfp4WireXor, v.z ^ kNvfp4WireXor,
                    v.w ^ kNvfp4WireXor);
}

// Combine partials: every bf16 lane 0xFFFF (a NaN) -> 0x7FFF.
__device__ __forceinline__ uint32_t canonicalize_bf16_lanes(uint32_t word) {
  if ((word & 0xFFFFu) == 0xFFFFu) word = (word & 0xFFFF0000u) | 0x7FFFu;
  if ((word >> 16) == 0xFFFFu) word = (word & 0x0000FFFFu) | 0x7FFF0000u;
  return word;
}

__device__ __forceinline__ uint4 canonicalize_partial(uint4 v) {
  return make_uint4(canonicalize_bf16_lanes(v.x), canonicalize_bf16_lanes(v.y),
                    canonicalize_bf16_lanes(v.z), canonicalize_bf16_lanes(v.w));
}

// MXFP8 scales: 8 UE8M0 bytes (raw) <-> 8 16-bit lanes 0x00XX (wire).
__device__ __forceinline__ uint4 mx_scales_to_wire(uint2 raw) {
  return make_uint4((raw.x & 0xFFu) | ((raw.x & 0xFF00u) << 8),
                    ((raw.x >> 16) & 0xFFu) | ((raw.x >> 8) & 0xFF0000u),
                    (raw.y & 0xFFu) | ((raw.y & 0xFF00u) << 8),
                    ((raw.y >> 16) & 0xFFu) | ((raw.y >> 8) & 0xFF0000u));
}

__device__ __forceinline__ uint2 mx_scales_from_wire(uint4 wire) {
  return make_uint2((wire.x & 0xFFu) | ((wire.x >> 8) & 0xFF00u) | ((wire.y & 0xFFu) << 16) |
                        (((wire.y >> 16) & 0xFFu) << 24),
                    (wire.z & 0xFFu) | ((wire.z >> 8) & 0xFF00u) | ((wire.w & 0xFFu) << 16) |
                        (((wire.w >> 16) & 0xFFu) << 24));
}

// ============================================================================ dispatch phases
// D1, warp 0, lane k = top-k slot k. Returns the token's target-rank mask (0 for a padding slot,
// token >= local_num_tokens). Stages the token's wire ids row for D3 in s_ids_row_words
// (canonical ids, invalid_expert_id past top_k) and, if topk_target_ranks_row is set, writes the
// rank of every slot at its first occurrence (else -1).
template <int TOP_K_CAPACITY, bool FIXED_TOP_K, bool ENABLE_RANK_MASK>
__device__ __forceinline__ uint64_t route_token(DispatchParams const& p, int token, int lane,
                                                uint32_t* s_ids_row_words,
                                                int32_t* topk_target_ranks_row) {
  int const top_k = FIXED_TOP_K ? TOP_K_CAPACITY : p.top_k;
  int expert_id = -1, rank = -1;
  if (token < p.local_num_tokens && lane < top_k) {
    expert_id = p.token_selected_experts[static_cast<int64_t>(token) * top_k + lane];
    // Experts split evenly (checked on the host). The fence kernels' compute_target_rank_id also
    // handles uneven splits, but its extra branch measurably slows this dispatch at decode sizes.
    rank = (expert_id >= 0 && expert_id < p.num_experts) ? expert_id / p.experts_per_rank : -1;
  }
  uint32_t const same_rank_lanes = __match_any_sync(0xFFFFFFFFu, rank);
  int target_rank = (rank >= 0 && __ffs(same_rank_lanes) - 1 == lane) ? rank : -1;
  uint32_t const low_ranks =
      __reduce_or_sync(0xFFFFFFFFu, rank >= 0 && rank < 32 ? 1u << rank : 0u);
  uint32_t const high_ranks = __reduce_or_sync(0xFFFFFFFFu, rank >= 32 ? 1u << (rank - 32) : 0u);
  uint64_t target_mask = static_cast<uint64_t>(low_ranks) | static_cast<uint64_t>(high_ranks) << 32;
  if constexpr (ENABLE_RANK_MASK) {
    target_mask &= p.active_rank_mask;
    if (target_rank >= 0 && !is_rank_alive<true>(p.active_rank_mask, target_rank)) target_rank = -1;
  }
  uint32_t const invalid_id = static_cast<uint32_t>(p.invalid_expert_id);
  if (lane < 4 * topk_row_vectors(TOP_K_CAPACITY))
    s_ids_row_words[lane] =
        lane < top_k ? canonicalize_expert_id(static_cast<uint32_t>(expert_id), invalid_id)
                     : invalid_id;
  if (lane < top_k && topk_target_ranks_row != nullptr && token < p.local_num_tokens)
    topk_target_ranks_row[lane] = target_rank;
  return target_mask;
}

// D2 / D5: where vector `vec` of a wire row lives (payload vectors first, then scale vectors).
struct WireVector {
  bool is_payload;
  int64_t region_offset;  // the landing region
  int row_bytes;          // landing row
  int offset_in_row;      // bytes
};

__device__ __forceinline__ WireVector locate_wire_vector(DispatchParams const& p, int vec) {
  if (vec < p.payload_vectors)
    return {true, p.landing.payload_offset, p.payload_row_bytes, vec * 16};
  return {false, p.landing.scale_offset, p.wire_scale_row_bytes, (vec - p.payload_vectors) * 16};
}

// D2: vector `vec` of the token's wire row, canonicalised.
template <int PAYLOAD_KIND>
__device__ __forceinline__ uint4 load_wire_vector(DispatchParams const& p, int token,
                                                  WireVector const& wv) {
  if (wv.is_payload) {
    uint4 const v = *reinterpret_cast<uint4 const*>(
        p.hidden_states + static_cast<int64_t>(token) * p.payload_row_bytes + wv.offset_in_row);
    if constexpr (PAYLOAD_KIND == kKindBf16) {
      return make_uint4(canonicalize_bf16x2(v.x), canonicalize_bf16x2(v.y),
                        canonicalize_bf16x2(v.z), canonicalize_bf16x2(v.w));
    } else if constexpr (PAYLOAD_KIND == kKindFp8) {
      return make_uint4(canonicalize_word(v.x), canonicalize_word(v.y), canonicalize_word(v.z),
                        canonicalize_word(v.w));
    } else {
      return nvfp4_wire_xor(make_uint4(canonicalize_fp4x8(v.x), canonicalize_fp4x8(v.y),
                                       canonicalize_fp4x8(v.z), canonicalize_fp4x8(v.w)));
    }
  }
  uint8_t const* scale_row =
      p.hidden_states_scale + static_cast<int64_t>(token) * p.scale_row_bytes;
  if (p.mx_scales)  // a wire vector = 8 raw UE8M0 bytes
    return mx_scales_to_wire(*reinterpret_cast<uint2 const*>(scale_row + wv.offset_in_row / 2));
  uint4 const v = *reinterpret_cast<uint4 const*>(scale_row + wv.offset_in_row);
  return make_uint4(canonicalize_word(v.x), canonicalize_word(v.y), canonicalize_word(v.z),
                    canonicalize_word(v.w));
}

// D2 / D5: store a wire vector (own or received) into receive row `recv_row`, in raw format.
template <int PAYLOAD_KIND>
__device__ __forceinline__ void store_received_vector(DispatchParams const& p, int64_t recv_row,
                                                      WireVector const& wv, uint4 wire) {
  if (wv.is_payload) {
    *reinterpret_cast<uint4*>(p.recv_hidden_states + recv_row * p.payload_row_bytes +
                              wv.offset_in_row) =
        PAYLOAD_KIND == kKindNvfp4 ? nvfp4_wire_xor(wire) : wire;
    return;
  }
  uint8_t* scale_row = p.recv_hidden_states_scale + recv_row * p.scale_row_bytes;
  if (p.mx_scales) {
    *reinterpret_cast<uint2*>(scale_row + wv.offset_in_row / 2) = mx_scales_from_wire(wire);
  } else {
    *reinterpret_cast<uint4*>(scale_row + wv.offset_in_row) = wire;
  }
}

// D2: store `wire` into the landing row (ep_rank, token) of every target rank; this rank's own
// copy goes straight into its receive row `recv_row`.
template <int PAYLOAD_KIND>
__device__ __forceinline__ void push_vector(DispatchParams const& p, uint64_t target_mask,
                                            int token, int64_t recv_row, WireVector const& wv,
                                            uint4 wire) {
  int64_t const offset =
      wv.region_offset + landing_row(p.ep_size, p.ep_rank, token) * wv.row_bytes + wv.offset_in_row;
  for (uint64_t targets = target_mask; targets;) {
    int const target = pop_lowest_rank(targets);
    if (target == p.ep_rank) {
      store_received_vector<PAYLOAD_KIND>(p, recv_row, wv, wire);
    } else {
      store_volatile(p.rank_workspaces[target] + offset, wire);
    }
  }
}

// D3, chunk-0 block: the ids row (ep_rank, token) goes to EVERY alive peer (all invalid_expert_id
// where the peer is no target; it still signals arrival), the weights row to the targets only;
// own rows go straight to the receive buffers. The (peer, vector) items are spread over the
// block's threads. s_ids_row / s_weights_row: the rows staged in D1.
template <bool ENABLE_RANK_MASK>
__device__ __forceinline__ void push_ids_and_weights(DispatchParams const& p, int top_k,
                                                     int ids_row_vectors, uint64_t target_mask,
                                                     int token, int64_t recv_row,
                                                     int64_t ids_set_offset, uint4 const* s_ids_row,
                                                     uint4 const* s_weights_row) {
  uint4 const no_target_ids = splat(static_cast<uint32_t>(p.invalid_expert_id));
  int64_t const offset =
      landing_row(p.ep_size, p.ep_rank, token) * (16 * ids_row_vectors);  // ids / weights row
  for (int item = threadIdx.x; item < p.ep_size * ids_row_vectors; item += kNumThreads) {
    int const peer = item / ids_row_vectors, vec = item % ids_row_vectors;
    bool const is_target = (target_mask >> peer) & 1ull;
    uint4 const ids = is_target ? s_ids_row[vec] : no_target_ids;
    bool const sends_weights = is_target && p.has_topk_weights;
    if (peer == p.ep_rank) {
      store_topk_words(reinterpret_cast<uint32_t*>(p.recv_topk_ids) + recv_row * top_k, top_k, vec,
                       ids);
      if (sends_weights)
        store_topk_words(p.recv_topk_weights + recv_row * top_k, top_k, vec, s_weights_row[vec]);
      continue;
    }
    if (!is_rank_alive<ENABLE_RANK_MASK>(p.active_rank_mask, peer)) continue;
    uint8_t* const peer_workspace = p.rank_workspaces[peer] + offset + vec * 16;
    store_volatile(peer_workspace + p.landing.ids_offset + ids_set_offset, ids);
    if (sends_weights)
      store_volatile(peer_workspace + p.landing.weights_offset, s_weights_row[vec]);
  }
}

// D4: record one received ids vector (source, vec): a source with any id other than
// invalid_expert_id sent a token; the chunk-0 block copies the ids out.
__device__ __forceinline__ void accept_ids_vector(DispatchParams const& p, int top_k, int token,
                                                  int chunk, int source, int vec, uint4 ids,
                                                  uint64_t& s_recv_source_mask) {
  uint32_t const invalid_id = static_cast<uint32_t>(p.invalid_expert_id);
  if (ids.x != invalid_id || ids.y != invalid_id || ids.z != invalid_id || ids.w != invalid_id)
    atomicOr(reinterpret_cast<unsigned long long*>(&s_recv_source_mask), 1ull << source);
  if (chunk != 0) return;
  int64_t const recv_row = static_cast<int64_t>(source) * p.max_tokens_per_rank + token;
  store_topk_words(reinterpret_cast<uint32_t*>(p.recv_topk_ids) + recv_row * top_k, top_k, vec,
                   ids);
}

// D4, a masked source: never polled, its row holds invalid ids.
__device__ __forceinline__ void reject_masked_source(DispatchParams const& p, int top_k, int token,
                                                     int chunk, int source, int vec) {
  if (chunk != 0) return;
  int64_t const recv_row = static_cast<int64_t>(source) * p.max_tokens_per_rank + token;
  store_topk_words(reinterpret_cast<uint32_t*>(p.recv_topk_ids) + recv_row * top_k, top_k, vec,
                   splat(static_cast<uint32_t>(p.invalid_expert_id)));
}

// D4: poll the ids rows (source, token) of every peer: ITEMS_PER_THREAD ids vectors (items) per
// thread, ep_size * ids_row_vectors items in all.
template <int ITEMS_PER_THREAD, bool ENABLE_RANK_MASK>
__device__ __forceinline__ void poll_ids_rows(DispatchParams const& p, int top_k,
                                              int ids_row_vectors, int token, int chunk,
                                              int64_t ids_set_offset, long long wait_start,
                                              uint64_t& s_recv_source_mask, bool& timed_out) {
  uint8_t const* ids_rows = p.rank_workspaces[p.ep_rank] + p.landing.ids_offset + ids_set_offset;
  int const num_items = p.ep_size * ids_row_vectors;  // item = (source, vec)
  int64_t const ids_row_bytes = 16 * ids_row_vectors;
  uint32_t pending = 0;
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    int const item = threadIdx.x + i * kNumThreads;
    int const source = item / ids_row_vectors;
    if (item >= num_items || source == p.ep_rank) continue;
    if (!is_rank_alive<ENABLE_RANK_MASK>(p.active_rank_mask, source)) {
      reject_masked_source(p, top_k, token, chunk, source, item % ids_row_vectors);
      continue;
    }
    pending |= 1u << i;
  }
  uint32_t const polled = pending;
  uint4 ids[ITEMS_PER_THREAD];
  auto address = [&](int i) {
    int const item = threadIdx.x + i * kNumThreads;
    return ids_rows + landing_row(p.ep_size, item / ids_row_vectors, token) * ids_row_bytes +
           (item % ids_row_vectors) * 16;
  };
  pending =
      poll_until_arrived(pending, kIdsSentinel, [] {}, address, ids, wait_start, p.timeout_cycles);
  timed_out = pending != 0;
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    if (!(((polled & ~pending) >> i) & 1u)) continue;
    int const item = threadIdx.x + i * kNumThreads;
    accept_ids_vector(p, top_k, token, chunk, item / ids_row_vectors, item % ids_row_vectors,
                      ids[i], s_recv_source_mask);
  }
}

// D4 -> D5, warp 0: the ranks of source_mask in ascending order into s_sources[0, n); lane 0
// stores n.
__device__ __forceinline__ void list_sources(uint64_t source_mask, int* s_sources,
                                             int* s_num_sources) {
  int const lane = threadIdx.x;
  uint32_t const low = static_cast<uint32_t>(source_mask), high = source_mask >> 32;
  uint32_t const below = (1u << lane) - 1u;
  if ((low >> lane) & 1u) s_sources[__popc(low & below)] = lane;
  if ((high >> lane) & 1u) s_sources[__popc(low) + __popc(high & below)] = 32 + lane;
  if (lane == 0) *s_num_sources = __popc(low) + __popc(high);
}

// D5 / D6: poll, copy out (wire -> raw) and re-arm this thread's wire vector of every source in
// s_sources[0, num_sources), 8 sources per round with every load in flight: n sources take
// ceil(n / 8) rounds whatever their ranks. A round's sources are re-read from shared memory before
// every spin's loads and before its copy-out (holding them across the round would exceed the
// dispatch's register bound). A thread
// with polls_weights (chunk 0, t < 8 * ids_row_vectors) also polls, as the round's vector 8,
// weights vector t % ids_row_vectors of the round's source t / ids_row_vectors.
template <int PAYLOAD_KIND>
__device__ __forceinline__ void receive_vectors(DispatchParams const& p, int top_k,
                                                int ids_row_vectors, int token, int vec,
                                                bool polls_weights, int const* s_sources,
                                                int num_sources, long long wait_start,
                                                bool& timed_out) {
  constexpr int kWeights = 8;  // the round's vector 8: this thread's weights vector
  bool const has_vector = vec < p.payload_vectors + p.wire_scale_vectors;
  WireVector const wv = locate_wire_vector(p, vec);
  uint8_t* const landing = p.rank_workspaces[p.ep_rank];
  uint8_t* const vector_column = landing + wv.region_offset + wv.offset_in_row;
  int const weights_round_slot =
      polls_weights ? static_cast<int>(threadIdx.x) / ids_row_vectors : kWeights;  // 8: none
  int const weights_vec = static_cast<int>(threadIdx.x) % ids_row_vectors;
  int64_t const ids_row_bytes = 16 * ids_row_vectors;
  uint8_t* const weights_column = landing + p.landing.weights_offset + weights_vec * 16;
  for (int first = 0; first < num_sources && !timed_out; first += 8) {
    int const round_size = min(8, num_sources - first);
    uint32_t pending = has_vector ? (1u << round_size) - 1u : 0u;
    int weights_source = 0;
    if (weights_round_slot < round_size) {
      weights_source = s_sources[first + weights_round_slot];
      pending |= 1u << kWeights;
    }
    uint32_t const polled = pending;
    uint4 vectors[kWeights + 1];
    int sources[8];  // the round's sources (past round_size: unused)
    auto read_sources = [&] {
#pragma unroll
      for (int j = 0; j < 8; ++j) sources[j] = s_sources[first + j];
    };
    auto address = [&](int i) {
      if (i == kWeights)
        return weights_column + landing_row(p.ep_size, weights_source, token) * ids_row_bytes;
      return vector_column + landing_row(p.ep_size, sources[i], token) * wv.row_bytes;
    };
    pending = poll_until_arrived(pending, kDataSentinel, read_sources, address, vectors, wait_start,
                                 p.timeout_cycles);
    timed_out = pending != 0;
    uint32_t const received = polled & ~pending;
    read_sources();
#pragma unroll
    for (int j = 0; j < kWeights; ++j) {
      if (!((received >> j) & 1u)) continue;
      int64_t const recv_row = static_cast<int64_t>(sources[j]) * p.max_tokens_per_rank + token;
      store_received_vector<PAYLOAD_KIND>(p, recv_row, wv, vectors[j]);
      store_volatile(address(j), splat(kDataSentinel));
    }
    if ((received >> kWeights) & 1u) {
      int64_t const recv_row = static_cast<int64_t>(weights_source) * p.max_tokens_per_rank + token;
      store_topk_words(p.recv_topk_weights + recv_row * top_k, top_k, weights_vec,
                       vectors[kWeights]);
      store_volatile(address(kWeights), splat(kDataSentinel));
    }
  }
}

// D4-D6 of every dispatch kernel, after the block's own pushes.
template <int PAYLOAD_KIND, bool ENABLE_RANK_MASK>
__device__ __forceinline__ void receive_token_slot(DispatchParams const& p, int top_k,
                                                   int ids_row_vectors, int token, int chunk,
                                                   int vec, uint64_t target_mask,
                                                   int64_t ids_set_offset,
                                                   uint64_t& s_recv_source_mask) {
  __shared__ __align__(16) int s_sources[kMaxRanks];
  __shared__ int s_num_sources;
  long long const wait_start = clock64();
  bool timed_out = false;
  if (threadIdx.x == 0 && ((target_mask >> p.ep_rank) & 1ull))
    atomicOr(reinterpret_cast<unsigned long long*>(&s_recv_source_mask), 1ull << p.ep_rank);
  if (p.ep_size * ids_row_vectors <= kNumThreads) {  // D4
    poll_ids_rows<1, ENABLE_RANK_MASK>(p, top_k, ids_row_vectors, token, chunk, ids_set_offset,
                                       wait_start, s_recv_source_mask, timed_out);
  } else {
    poll_ids_rows<4, ENABLE_RANK_MASK>(p, top_k, ids_row_vectors, token, chunk, ids_set_offset,
                                       wait_start, s_recv_source_mask, timed_out);
  }
  __syncthreads();
  uint64_t const recv_source_mask = s_recv_source_mask;
  if (chunk == 0 && threadIdx.x == 0) p.recv_source_mask[token] = recv_source_mask;
  if (threadIdx.x < 32)  // this rank's own row was received in D2
    list_sources(recv_source_mask & ~(1ull << p.ep_rank), s_sources, &s_num_sources);
  __syncthreads();
  bool const polls_weights =
      chunk == 0 && p.has_topk_weights && static_cast<int>(threadIdx.x) < 8 * ids_row_vectors;
  if ((vec < p.payload_vectors + p.wire_scale_vectors || polls_weights) && !timed_out)  // D5, D6
    receive_vectors<PAYLOAD_KIND>(p, top_k, ids_row_vectors, token, vec, polls_weights, s_sources,
                                  s_num_sources, wait_start, timed_out);
  if (timed_out)
    report_and_trap(p.round_state, p.ep_rank, "timed out waiting for a peer (dispatch)");
}
#endif  // MOE_A2A_LAMPORT_COMPILE

// ============================================================================= dispatch kernel
// Any payload format and top-k (fixed at compile time, or at most TOP_K_CAPACITY).
template <int TOP_K_CAPACITY, bool FIXED_TOP_K, int PAYLOAD_KIND, bool ENABLE_RANK_MASK>
__global__ void __launch_bounds__(kNumThreads, kDispatchBlocksPerSM)
    moeA2ALamportDispatchKernel(DispatchParams const p) {
#if MOE_A2A_LAMPORT_COMPILE
  constexpr int kRowCapacity = topk_row_vectors(TOP_K_CAPACITY);
  int const top_k = FIXED_TOP_K ? TOP_K_CAPACITY : p.top_k;
  int const ids_row_vectors = topk_row_vectors(top_k);
  int const token = blockIdx.x / p.blocks_per_token, chunk = blockIdx.x % p.blocks_per_token;
  int const vec = chunk * kNumThreads + threadIdx.x;
  __shared__ uint64_t s_target_mask, s_recv_source_mask;
  __shared__ int s_landing_set;
  __shared__ uint4 s_ids_row[kRowCapacity], s_weights_row[kRowCapacity];  // wire rows for D3
  wait_for_previous_kernel();
  if (threadIdx.x < 32) {  // D1
    int const lane = threadIdx.x;
    int epoch = 0;
    if (lane == 0) epoch = load_volatile_int(p.round_state + kEpoch);
    uint32_t weight = 0;  // loaded before routing, so that it is in flight with the expert id
    if (chunk == 0 && p.has_topk_weights && token < p.local_num_tokens && lane < top_k)
      weight = p.topk_weights[static_cast<int64_t>(token) * top_k + lane];
    uint64_t const target_mask = route_token<TOP_K_CAPACITY, FIXED_TOP_K, ENABLE_RANK_MASK>(
        p, token, lane, reinterpret_cast<uint32_t*>(s_ids_row),
        chunk == 0 ? p.topk_target_ranks + static_cast<int64_t>(token) * top_k : nullptr);
    if (lane < 4 * kRowCapacity)  // padding: weight 0
      reinterpret_cast<uint32_t*>(s_weights_row)[lane] =
          lane < top_k ? canonicalize_word(weight) : 0u;
    if (lane == 0) {
      s_landing_set = epoch & 1;
      s_target_mask = target_mask;
      s_recv_source_mask = 0;
    }
  }
  __syncthreads();
  uint64_t const target_mask = s_target_mask;
  int64_t const ids_set_offset = s_landing_set * p.landing.ids_set_bytes;
  int64_t const recv_row = static_cast<int64_t>(p.ep_rank) * p.max_tokens_per_rank + token;
  if (target_mask && vec < p.payload_vectors + p.wire_scale_vectors) {  // D2
    WireVector const wv = locate_wire_vector(p, vec);
    push_vector<PAYLOAD_KIND>(p, target_mask, token, recv_row, wv,
                              load_wire_vector<PAYLOAD_KIND>(p, token, wv));
  }
  if (chunk == 0)  // D3
    push_ids_and_weights<ENABLE_RANK_MASK>(p, top_k, ids_row_vectors, target_mask, token, recv_row,
                                           ids_set_offset, s_ids_row, s_weights_row);
  allow_next_kernel();
  receive_token_slot<PAYLOAD_KIND, ENABLE_RANK_MASK>(p, top_k, ids_row_vectors, token, chunk, vec,
                                                     target_mask, ids_set_offset,
                                                     s_recv_source_mask);
#else
  __trap();  // not built for this architecture (see MOE_A2A_LAMPORT_COMPILE)
#endif
}

#if MOE_A2A_LAMPORT_COMPILE
// ============================================================================= combine phases
// C1, warp 0: the ranks of the token's target slots (topk_target_ranks >= 0) in top-k slot
// order into s_sources[0, n); lane 0 stores n.
__device__ __forceinline__ void list_target_ranks(CombineParams const& p, int token, int top_k,
                                                  int* s_sources, int* s_num_sources) {
  int const lane = threadIdx.x;
  int const rank = (lane < top_k && token < p.local_num_tokens)
                       ? p.topk_target_ranks[static_cast<int64_t>(token) * top_k + lane]
                       : -1;
  uint32_t const source_lanes = __ballot_sync(0xFFFFFFFFu, rank >= 0);
  if (rank >= 0) s_sources[__popc(source_lanes & ((1u << lane) - 1u))] = rank;
  if (lane == 0) *s_num_sources = __popc(source_lanes);
}

// C2: push the expert output of row (source, token) to source's combine landing row
// (ep_rank, token), for every source that sent this slot a token but this rank (whose partial
// C3 reads directly). 8 sources at a time, every load issued before the stores (a store's
// "memory" clobber would otherwise wait for one load after another).
__device__ __forceinline__ void push_partials(CombineParams const& p, int token, int vec,
                                              uint64_t recv_source_mask) {
  int64_t const row_bytes = static_cast<int64_t>(p.hidden_size) * 2;
  int64_t const landing_offset =
      p.landing.combine_offset +
      (static_cast<int64_t>(p.ep_rank) * p.max_tokens_per_rank + token) * row_bytes + vec * 16;
  for (uint64_t unsent = recv_source_mask & ~(1ull << p.ep_rank); unsent;) {
    int sources[8];  // the round's sources, -1 past the last
    uint4 partials[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      sources[j] = unsent ? pop_lowest_rank(unsent) : -1;
      if (sources[j] >= 0)
        partials[j] = *reinterpret_cast<uint4 const*>(
            p.expert_output +
            (static_cast<int64_t>(sources[j]) * p.max_tokens_per_rank + token) * row_bytes +
            vec * 16);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j)
      if (sources[j] >= 0)
        store_volatile(p.rank_workspaces[sources[j]] + landing_offset,
                       canonicalize_partial(partials[j]));
  }
}

// Fence-combine order of up to 8 sources (zero-filled): s0 + s1 for n <= 2, else the pairwise
// tree ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7)). __fadd_rn is never fused and, like the
// fence combine's adds, flushes subnormals exactly when the module builds with -ftz (fast math).
__device__ __forceinline__ float sum_group(float const (&values)[8], int num_sources) {
  if (num_sources <= 2) return __fadd_rn(values[0], values[1]);
  return __fadd_rn(__fadd_rn(__fadd_rn(values[0], values[1]), __fadd_rn(values[2], values[3])),
                   __fadd_rn(__fadd_rn(values[4], values[5]), __fadd_rn(values[6], values[7])));
}

// C3, sources 8 * group .. 8 * group + 7 of the slot-ordered list: poll their partials of this
// rank's token (every load in flight), re-arm them, and sum each of the vector's 8 elements.
__device__ __forceinline__ void receive_group(CombineParams const& p, int token, int vec, int group,
                                              int num_sources, int const* s_sources,
                                              long long wait_start, bool& timed_out,
                                              float (&sums)[8]) {
#pragma unroll
  for (int e = 0; e < 8; ++e) sums[e] = 0.f;
  if (8 * group >= num_sources) return;
  int64_t const row_bytes = static_cast<int64_t>(p.hidden_size) * 2;
  uint8_t* combine_rows = p.rank_workspaces[p.ep_rank] + p.landing.combine_offset + vec * 16;
  int ranks[8];
#pragma unroll
  for (int j = 0; j < 8; ++j)
    ranks[j] = 8 * group + j < num_sources ? s_sources[8 * group + j] : -1;
  uint4 partials[8];
  uint32_t pending = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    partials[j] = make_uint4(0, 0, 0, 0);  // a missing source adds +0
    if (ranks[j] == p.ep_rank) {
      partials[j] = canonicalize_partial(*reinterpret_cast<uint4 const*>(
          p.expert_output +
          (static_cast<int64_t>(p.ep_rank) * p.max_tokens_per_rank + token) * row_bytes +
          vec * 16));
    } else if (ranks[j] >= 0) {
      pending |= 1u << j;
    }
  }
  uint32_t const polled = pending;
  auto address = [&](int j) {
    return combine_rows +
           (static_cast<int64_t>(ranks[j]) * p.max_tokens_per_rank + token) * row_bytes;
  };
  if (poll_until_arrived(
          pending, kDataSentinel, [] {}, address, partials, wait_start, p.timeout_cycles))
    timed_out = true;
#pragma unroll
  for (int j = 0; j < 8; ++j)
    if ((polled >> j) & 1u) store_volatile(address(j), splat(kDataSentinel));
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    float values[8];
#pragma unroll
    for (int j = 0; j < 8; ++j)
      values[j] = __bfloat162float(reinterpret_cast<__nv_bfloat16 const*>(&partials[j])[e]);
    sums[e] = sum_group(values, num_sources);
  }
}

// C3: join the group sums as the fence combine joins its lanes (n <= 8: g0; n <= 16: g0 + g1;
// else (g0 + g1) + (g2 + g3)) and round to bf16.
template <int NUM_GROUPS>
__device__ __forceinline__ uint4 join_groups(float const (&group_sums)[NUM_GROUPS][8],
                                             int num_sources) {
  uint4 out;
  __nv_bfloat16* out_elements = reinterpret_cast<__nv_bfloat16*>(&out);
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    float sum = group_sums[0][e];
    if constexpr (NUM_GROUPS > 1) {
      if (num_sources > 8) sum = __fadd_rn(sum, group_sums[1][e]);
    }
    if constexpr (NUM_GROUPS > 2) {
      if (num_sources > 16) {
        float last = 0.f;
        if constexpr (NUM_GROUPS > 3) last = group_sums[3][e];
        sum = __fadd_rn(sum, __fadd_rn(group_sums[2][e], last));
      }
    }
    out_elements[e] = __float2bfloat16_rn(sum);
  }
  return out;
}

// C4: re-arm this slot's ids rows of the round's landing set (their readers, the dispatch
// blocks, have all ended), then the last block closes the round and advances the epoch (every
// block read it first).
__device__ __forceinline__ void rearm_ids_and_advance_epoch(CombineParams const& p, int token,
                                                            int chunk, int ids_row_vectors,
                                                            int landing_set) {
  if (chunk == 0) {
    uint8_t* ids_rows =
        p.rank_workspaces[p.ep_rank] + p.landing.ids_offset + landing_set * p.landing.ids_set_bytes;
    int64_t const ids_row_bytes = 16 * ids_row_vectors;
    for (int item = threadIdx.x; item < p.ep_size * ids_row_vectors; item += kNumThreads) {
      int const source = item / ids_row_vectors, vec = item % ids_row_vectors;
      if (source == p.ep_rank) continue;  // never written: own rows go to the receive buffers
      store_volatile(ids_rows + landing_row(p.ep_size, source, token) * ids_row_bytes + vec * 16,
                     splat(kIdsSentinel));
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    int const finished_before = atomicAdd(p.round_state + kFinishedCombineBlocks, 1);
    if (finished_before == static_cast<int>(gridDim.x) - 1) {
      atomicExch(p.round_state + kFinishedCombineBlocks, 0);
      atomicAdd(p.round_state + kEpoch, 1);
    }
  }
}
#endif  // MOE_A2A_LAMPORT_COMPILE

// ============================================================================== combine kernel
template <int TOP_K_CAPACITY, bool FIXED_TOP_K, bool ENABLE_RANK_MASK>
__global__ void __launch_bounds__(kNumThreads) moeA2ALamportCombineKernel(CombineParams const p) {
#if MOE_A2A_LAMPORT_COMPILE
  constexpr int kNumGroups = (TOP_K_CAPACITY + 7) / 8;
  int const top_k = FIXED_TOP_K ? TOP_K_CAPACITY : p.top_k;
  int const token = blockIdx.x / p.blocks_per_token, chunk = blockIdx.x % p.blocks_per_token;
  int const vec = chunk * kNumThreads + threadIdx.x;
  __shared__ uint64_t s_recv_source_mask;
  __shared__ int s_sources[TOP_K_CAPACITY];
  __shared__ int s_num_sources, s_landing_set;
  wait_for_previous_kernel();  // the expert output, and the dispatch's routing state
  if (threadIdx.x < 32) {      // C1
    list_target_ranks(p, token, top_k, s_sources, &s_num_sources);
    if (threadIdx.x == 0) {
      s_landing_set = load_volatile_int(p.round_state + kEpoch) & 1;
      uint64_t mask = p.recv_source_mask[token];
      if constexpr (ENABLE_RANK_MASK) mask &= p.active_rank_mask;  // never push to a masked rank
      s_recv_source_mask = mask;
    }
  }
  __syncthreads();
  if (vec < p.row_vectors) push_partials(p, token, vec, s_recv_source_mask);  // C2
  allow_next_kernel();
  bool timed_out = false;
  if (token < p.local_num_tokens && vec < p.row_vectors) {  // C3
    long long const wait_start = clock64();
    float group_sums[kNumGroups][8];
#pragma unroll
    for (int group = 0; group < kNumGroups; ++group)
      receive_group(p, token, vec, group, s_num_sources, s_sources, wait_start, timed_out,
                    group_sums[group]);
    *reinterpret_cast<uint4*>(p.output + (static_cast<int64_t>(token) * p.hidden_size) * 2 +
                              vec * 16) = join_groups<kNumGroups>(group_sums, s_num_sources);
  }
  if (timed_out)
    report_and_trap(p.round_state, p.ep_rank, "timed out waiting for a peer (combine)");
  rearm_ids_and_advance_epoch(p, token, chunk, topk_row_vectors(top_k), s_landing_set);  // C4
#else
  __trap();  // not built for this architecture (see MOE_A2A_LAMPORT_COMPILE)
#endif
}

// Defined in csrc/nvlink_one_sided_lamport.cu (the kernels' only translation unit).
// Whether the code loaded for the current device carries the kernels (MOE_A2A_LAMPORT_COMPILE).
bool moe_a2a_lamport_kernels_available();
void moe_a2a_lamport_dispatch_launch(int payload_kind, DispatchParams const& params,
                                     bool enable_rank_mask, bool enable_pdl, cudaStream_t stream);
void moe_a2a_lamport_combine_launch(CombineParams const& params, bool enable_rank_mask,
                                    bool enable_pdl, cudaStream_t stream);

}  // namespace moe_alltoall_lamport
}  // namespace flashinfer
