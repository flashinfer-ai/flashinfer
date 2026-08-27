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
//
// MoE routing prologue and weighted-sum finalize for the Qwen3.6-A3B MoE
// geometry (hidden 2048, 256 experts, top-8, BLOCK_M 8) on SM120.
//
// TWO entry points, because that is the dataflow of a serving MoE block:
//
//   moe_routing_prologue(hidden, gate_w, shared_gate_w)
//       -> router_logits, topk_weights, topk_ids,
//          sorted_token_ids, expert_ids, num_tokens_post_pad, shared_gate
//              |
//              |  w13 GEMM -> activation -> w2 GEMM  (the expert GEMMs read the
//              |  descriptors this produced and emit `expert_out`)
//              v
//   moe_routing_finalize(expert_out, shared_out, topk_weights, shared_gate)
//       -> output
//
// The prologue's descriptors are what the expert GEMM needs in order to produce
// `expert_out`, so a single fused op that both consumes `expert_out` and emits
// the descriptors would need its own output as its input.  Splitting at the
// GEMM boundary is the only shape a serving block can call, and it also makes
// each half a plain, self-contained kernel: no CTA ever waits on another CTA,
// and there is no persistent device state of any kind.
//
// IMPORTANT contract: `expert_out` is the routed-expert down-projection output
// with the routing weights NOT yet applied.  The finalize owns the routing
// weights.  A caller whose expert GEMM folds them into its epilogue must turn
// that off (vLLM: `mul_topk_weights=False`), or they are applied twice.
//
// Three kernels, all plain:
//
//   moe_router_gemv        grid NUM_EXPERTS+1, 256 thr
//                          CTA e computes router logit row e for every token;
//                          CTA NUM_EXPERTS computes the shared-expert sigmoid
//                          gate.  Both are the same dot product over HID, so
//                          they share one code path.
//   moe_routing_descriptor grid 1, 256 thr  (shared by the prologue and the
//                          standalone align entry point)
//                          fp32 softmax top-k of the 256 bf16 logits per token
//                          (descending score, lower expert id on a tie),
//                          renormalised, then the block-aligned sort
//                          descriptors the expert GEMM consumes.
//   moe_finalize           grid FSPLIT*M, 64 thr
//                          expert-weighted sum plus, optionally, the gated
//                          shared expert; fp32 accumulation and a single bf16
//                          rounding.  The shared-expert operands are optional
//                          because an engine that combines the shared expert
//                          somewhere else (vLLM adds it outside the expert
//                          kernel) still wants the weighted sum fused.
//
// The router GEMV and the descriptor pass are two launches rather than one
// because the top-k of a token needs all 256 of its logits, and the GEMV needs
// the whole card: 1 MB of router weights cannot be read by one CTA.  Making
// them one launch would require a grid-wide rendezvous, which is exactly the
// complexity this split exists to delete.
//
// Host side: no allocation, no synchronization, no device limits and no stream
// attributes are touched, so every launch records into a CUDA graph.  Token
// counts the prologue cannot serve are a hard error, never a silent no-op.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>

#include "tvm_ffi_utils.h"

namespace {

#define NUM_EXPERTS 256
#define HID 2048
#define TOPK 8
#define BLOCK_M 8
#define MAXM 32 /* descriptor: M*TOPK assignments must fit one CTA */

#define PT_R 256           /* router GEMV threads per CTA          */
#define RWARPS (PT_R / 32) /* = 8, warps that split one HID row     */
#define RCHUNK 8           /* tokens whose GEMV is kept in flight  */

#define PT_D 256 /* descriptor threads (>= MAXM * TOPK)  */
#define DWARPS (PT_D / 32)

#define PT_F 64                      /* finalize threads per CTA  */
#define FVEC 8                       /* bf16 per thread (uint4)   */
#define FSPLIT (HID / (PT_F * FVEC)) /* = 4 CTAs per token        */
#define FVW (FVEC / 2)               /* 32-bit words per thread   */

// --------------------------- bit / memory helpers --------------------------
static __device__ __forceinline__ float bflo(unsigned int w) {
  return __int_as_float((int)(w << 16));
}
static __device__ __forceinline__ float bfhi(unsigned int w) {
  return __int_as_float((int)(w & 0xFFFF0000u));
}
static __device__ __forceinline__ unsigned int to_bf16(float x) {
  unsigned short r;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(x));
  return (unsigned int)r;
}
static __device__ __forceinline__ void unpack8(uint4 a, float* w) {
  w[0] = bflo(a.x);
  w[1] = bfhi(a.x);
  w[2] = bflo(a.y);
  w[3] = bfhi(a.y);
  w[4] = bflo(a.z);
  w[5] = bfhi(a.z);
  w[6] = bflo(a.w);
  w[7] = bfhi(a.w);
}
// 8-element dot with the weight side already unpacked: the router CTA reuses
// one weight row across every token, so unpacking it once is strictly cheaper.
// The summation order is fixed and matches the reduction below; it is part of
// the op's numerics, not an implementation detail.
static __device__ __forceinline__ float dot8p(const float* w, uint4 b) {
  float s0 = fmaf(w[0], bflo(b.x), w[1] * bfhi(b.x));
  float s1 = fmaf(w[2], bflo(b.y), w[3] * bfhi(b.y));
  float s2 = fmaf(w[4], bflo(b.z), w[5] * bfhi(b.z));
  float s3 = fmaf(w[6], bflo(b.w), w[7] * bfhi(b.w));
  return (s0 + s1) + (s2 + s3);
}

// bf16 bits -> order preserving key with the expert index folded into the low 8
// bits, so plain unsigned max means "highest score, lowest id on a tie".
static __device__ __forceinline__ unsigned int mkkey(unsigned int b, int e) {
  bool bad = ((b & 0x7F80u) == 0x7F80u) && (((b & 0x007Fu) != 0u) || ((b & 0x8000u) == 0u));
  unsigned int mono = (b & 0x8000u) ? ((~b) & 0xFFFFu) : (b | 0x8000u);
  if (bad) mono = 0u;
  return (mono << 8) | (unsigned int)(255 - e);
}
static __device__ __forceinline__ float mono_to_float(unsigned int mono) {
  if (mono == 0u) return -CUDART_INF_F;
  unsigned int b = (mono >= 0x8000u) ? (mono - 0x8000u) : ((~mono) & 0xFFFFu);
  return __int_as_float((int)(b << 16));
}

// ---------------------------------------------------------------------------
// Kernel 1: router GEMV for every expert, plus the shared-expert scalar gate.
//
// One CTA per expert row (plus one for the shared gate), 8 warps splitting the
// 2048-wide row so each lane holds exactly one uint4 of weights for the whole
// kernel.  Tokens are processed in register chunks so that, at the decode sizes
// this ships for, every token's load is in flight at once.
__global__ __launch_bounds__(PT_R) void moe_router_gemv(const __nv_bfloat16* __restrict__ hidden,
                                                        const __nv_bfloat16* __restrict__ gatew,
                                                        const __nv_bfloat16* __restrict__ sgatew,
                                                        unsigned short* __restrict__ o_logits,
                                                        unsigned short* __restrict__ o_gate,
                                                        int M) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int base = tid * 8;  // == warp * 256 + lane * 8, the lane's slice of HID
  const bool is_gate = (bid == NUM_EXPERTS);

  // The weight LDG is the first memory operation the block issues: on a cold
  // launch the instruction fetch is itself a serial dependency, so anything
  // ahead of the first load pushes the whole GEMV out.
  const __nv_bfloat16* wp = is_gate ? (sgatew + base) : (gatew + (size_t)bid * HID + base);
  const uint4 wv = *reinterpret_cast<const uint4*>(wp);
  float wf[8];
  unpack8(wv, wf);

  __shared__ float s_red[RCHUNK * RWARPS];

  for (int m0 = 0; m0 < M; m0 += RCHUNK) {
    const int n = min(RCHUNK, M - m0);

    float acc[RCHUNK];
#pragma unroll
    for (int i = 0; i < RCHUNK; ++i) {
      acc[i] = 0.f;
      if (i < n) {
        const uint4 hv = *reinterpret_cast<const uint4*>(hidden + (size_t)(m0 + i) * HID + base);
        acc[i] = dot8p(wf, hv);
      }
    }
#pragma unroll
    for (int i = 0; i < RCHUNK; ++i) {
      if (i < n) {
        float a = acc[i];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
        if (lane == 0) s_red[i * RWARPS + warp] = a;
      }
    }
    __syncthreads();

    if (tid < n) {
      const float* r = &s_red[tid * RWARPS];
      const float s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
      if (is_gate) {
        // bf16-round the logit, then sigmoid, then bf16-round the gate: the
        // rounding points are part of the op's semantics.
        const float gl = bflo(to_bf16(s));
        o_gate[m0 + tid] = (unsigned short)to_bf16(1.0f / (1.0f + expf(-gl)));
      } else {
        o_logits[(size_t)(m0 + tid) * NUM_EXPERTS + bid] = (unsigned short)to_bf16(s);
      }
    }
    __syncthreads();  // s_red is reused by the next chunk
  }
}

// ---------------------------------------------------------------------------
// Top-8 of one token's 256 bf16 logits, in one warp: descending score, ties to
// the lower expert index, renormalised in fp32.
static __device__ __noinline__ void topk8(const __nv_bfloat16* __restrict__ row, int lane,
                                          float* w_out, int* id_out) {
  const uint4 a = *reinterpret_cast<const uint4*>(row + lane * 8);
  const int e0 = lane * 8;
  unsigned int k0 = mkkey(a.x & 0xFFFFu, e0 + 0);
  unsigned int k1 = mkkey(a.x >> 16, e0 + 1);
  unsigned int k2 = mkkey(a.y & 0xFFFFu, e0 + 2);
  unsigned int k3 = mkkey(a.y >> 16, e0 + 3);
  unsigned int k4 = mkkey(a.z & 0xFFFFu, e0 + 4);
  unsigned int k5 = mkkey(a.z >> 16, e0 + 5);
  unsigned int k6 = mkkey(a.w & 0xFFFFu, e0 + 6);
  unsigned int k7 = mkkey(a.w >> 16, e0 + 7);

  // Batcher odd-even mergesort of the lane's 8 keys, descending (19 comparators,
  // depth 6).  Sorting once turns each of the 8 selection rounds into a single
  // REDUX plus a predicated shift, instead of re-scanning all 8 registers for
  // the local max and then re-scanning them again to clear the winner: 134
  // dynamic instructions instead of 216 on the tail that gates this kernel.
#define CE(x, y)                       \
  {                                    \
    const unsigned int t_ = max(x, y); \
    y = min(x, y);                     \
    x = t_;                            \
  }
  // clang-format off
  // One line per stage of the network; clang-format cannot lay out a run of
  // semicolon-less macro invocations idempotently (it rewraps it differently
  // on every pass), and the stage structure is the point of the sequence.
  CE(k0, k1) CE(k2, k3) CE(k4, k5) CE(k6, k7)
  CE(k0, k2) CE(k1, k3) CE(k4, k6) CE(k5, k7)
  CE(k1, k2) CE(k5, k6)
  CE(k0, k4) CE(k1, k5) CE(k2, k6) CE(k3, k7)
  CE(k2, k4) CE(k3, k5)
  CE(k1, k2) CE(k3, k4) CE(k5, k6)
  // clang-format on
#undef CE

          unsigned int mysel = 0u;
  for (int r = 0; r < 8; ++r) {
    const unsigned int best = __reduce_max_sync(0xffffffffu, k0);
    if (lane == r) mysel = best;
    const bool win = (k0 == best);
    k0 = win ? k1 : k0;
    k1 = win ? k2 : k1;
    k2 = win ? k3 : k2;
    k3 = win ? k4 : k3;
    k4 = win ? k5 : k4;
    k5 = win ? k6 : k5;
    k6 = win ? k7 : k6;
    k7 = win ? 0u : k7;
  }

  const float l0 = mono_to_float(__shfl_sync(0xffffffffu, mysel, 0) >> 8);
  float x = (lane < 8) ? expf(mono_to_float(mysel >> 8) - l0) : 0.f;
  float sum = x;
#pragma unroll
  for (int off = 4; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
  if (!(sum > 0.f)) sum = 1.f;
  if (lane < 8) {
    w_out[lane] = x * __frcp_rn(sum);
    id_out[lane] = 255 - (int)(mysel & 0xFFu);
  }
}

// An expert id is only safe to use as an index once it is known to be one.
static __device__ __forceinline__ bool expert_in_range(int e) {
  return static_cast<unsigned int>(e) < static_cast<unsigned int>(NUM_EXPERTS);
}

// ---------------------------------------------------------------------------
// Kernel 2: top-k + the block-aligned descriptors the expert GEMM consumes.
// One CTA: M * TOPK <= MAXM * TOPK == PT_D assignments is the whole problem, so
// the counting scan and the scatter are block-local and there is nothing to
// synchronize with outside this block.
// `logits` non-null  -> the prologue's use: score, select, then describe.
// `in_tid`  non-null  -> the align entry point's use: the caller already has
//                        topk_ids (its own router ran), so only describe.
// Exactly one of them is non-null, and the DESCRIPTOR CODE BELOW IS SHARED --
// two copies of it could drift, and a descriptor that disagrees with the one
// the expert GEMM was fed is a silent wrong answer.
// `o_tw`/`o_tid` are null when the caller already owns the routing outputs.
__global__ __launch_bounds__(PT_D) void moe_routing_descriptor(
    const __nv_bfloat16* __restrict__ logits, const int* __restrict__ in_tid,
    float* __restrict__ o_tw, int* __restrict__ o_tid, int* __restrict__ o_sti,
    int* __restrict__ o_eid, int* __restrict__ o_ntp, int M) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int numel = M * TOPK;

  __shared__ float s_w[MAXM * TOPK];
  __shared__ int s_id[MAXM * TOPK];

  // Padding sentinels do not depend on the routing: issue them first so the
  // stores are in flight while the top-k runs.
  for (int i = tid; i < 64 * M; i += PT_D) o_sti[i] = numel;
  for (int b = tid; b < BLOCK_M * M; b += PT_D) o_eid[b] = -1;

  if (logits != nullptr) {
    for (int mm = warp; mm < M; mm += DWARPS)
      topk8(logits + (size_t)mm * NUM_EXPERTS, lane, s_w + mm * TOPK, s_id + mm * TOPK);
  } else {
    for (int i = tid; i < numel; i += PT_D) s_id[i] = in_tid[i];
  }

  if (M <= 4) {
    // numel = 8*M <= 32 assignments: the whole descriptor fits in one warp with
    // no histogram, no block scan and no second barrier.  __match_any_sync
    // gives the per-expert multiplicity and the intra-expert rank at once.
    __syncthreads();
    if (warp == 0) {
      const int id = (lane < numel) ? s_id[lane] : (0x10000 + lane);
      const unsigned int act = (numel >= 32) ? 0xffffffffu : (unsigned int)((1u << numel) - 1u);
      const unsigned int mk = __match_any_sync(0xffffffffu, id) & act;
      const int pos = __popc(mk & ((1u << lane) - 1u));
      const bool first = (lane < numel) && (pos == 0);
      // Inactive lanes carry a key no active id can be below, so the count of
      // distinct experts ordered before `id` can scan the full warp.
      const int key = first ? id : 0x7FFFFFFF;
      int before = 0;
#pragma unroll
      for (int j = 0; j < 32; ++j) before += (__shfl_sync(0xffffffffu, key, j) < id);
      const int total = __popc(__ballot_sync(0xffffffffu, first));
      if (lane == 0) *o_ntp = total * BLOCK_M;
      if (lane < numel) {
        if (o_tw != nullptr) {
          o_tw[lane] = s_w[lane];
          o_tid[lane] = id;
        }
        o_sti[before * BLOCK_M + pos] = lane;
        if (first) o_eid[before] = id;
      }
    }
    return;
  }

  __shared__ int s_cnt[NUM_EXPERTS];
  __shared__ int s_cur[NUM_EXPERTS];
  __shared__ int s_start[NUM_EXPERTS];
  __shared__ int s_sw[DWARPS + 1];

  s_cnt[tid] = 0;
  s_cur[tid] = 0;
  __syncthreads();

  if (tid < numel) {
    const int e = s_id[tid];
    // Everything below indexes s_cnt / s_cur / s_start, which are NUM_EXPERTS
    // wide, with an id that on the align entry point came straight out of the
    // caller's `in_tid`.  An id outside [0, NUM_EXPERTS) would be an
    // out-of-bounds shared-memory access, so DROP the assignment: its slot in
    // `o_sti` keeps the `numel` sentinel prefilled at the top of this kernel,
    // which is the same "no token here" value the padding uses, so the
    // descriptor stays consistent instead of half-written.
    // (The prologue cannot produce one -- topk8 emits 255 - e for e in
    // [0, NUM_EXPERTS) -- and it is the only caller that passes o_tw/o_tid,
    // so this guard costs the shipped path nothing.)
    if (expert_in_range(e)) {
      if (o_tw != nullptr) {
        o_tw[tid] = s_w[tid];
        o_tid[tid] = e;
      }
      atomicAdd(&s_cnt[e], 1);
    }
  }
  __syncthreads();

  const int c = s_cnt[tid];
  const int pb = (c + BLOCK_M - 1) >> 3;
  int xs = pb;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, xs, off);
    if (lane >= off) xs += y;
  }
  if (lane == 31) s_sw[warp] = xs;
  __syncthreads();
  if (tid == 0) {
    int s = 0;
#pragma unroll
    for (int i = 0; i < DWARPS; ++i) {
      const int t = s_sw[i];
      s_sw[i] = s;
      s += t;
    }
    s_sw[DWARPS] = s;
  }
  __syncthreads();
  const int excl = xs - pb + s_sw[warp];
  if (tid == 0) *o_ntp = s_sw[DWARPS] * BLOCK_M;
  s_start[tid] = excl;

  for (int j = 0; j < pb; ++j) o_eid[excl + j] = tid;
  __syncthreads();
  if (tid < numel) {
    const int e = s_id[tid];
    // Same drop as the counting pass above, and it has to be the same
    // predicate: an id counted there is an id scattered here, and one without
    // the other would either lose a token or scatter into a block no expert
    // owns.
    if (expert_in_range(e)) {
      const int r = atomicAdd(&s_cur[e], 1);
      o_sti[s_start[e] * BLOCK_M + r] = tid;
    }
  }
}

// ---------------------------------------------------------------------------
// Kernel 3: expert-weighted sum + gated shared expert.
// FSPLIT CTAs per token, 64 threads each: the routed slice of one token is
// 8 * 512 bf16, and issuing all nine uint4 loads per thread up front puts the
// entire working set in flight before anything is consumed.
__global__ __launch_bounds__(PT_F) void moe_finalize(const __nv_bfloat16* __restrict__ eout,
                                                     const __nv_bfloat16* __restrict__ sout,
                                                     const float* __restrict__ tw,
                                                     const unsigned short* __restrict__ sgate,
                                                     __nv_bfloat16* __restrict__ o_out) {
  const int bid = blockIdx.x;
  const int m = bid / FSPLIT;
  const int part = bid - m * FSPLIT;
  const int tid = threadIdx.x;
  const int h0 = part * (PT_F * FVEC) + tid * FVEC;

  uint4 ev[TOPK];
  const __nv_bfloat16* ep = eout + ((size_t)m * TOPK) * HID + h0;
#pragma unroll
  for (int k = 0; k < TOPK; ++k, ep += HID) ev[k] = *reinterpret_cast<const uint4*>(ep);
  // The shared expert is optional: an engine that combines it elsewhere passes
  // neither operand and this is exactly the routed weighted sum.
  const bool has_shared = (sgate != nullptr);
  uint4 sv = make_uint4(0u, 0u, 0u, 0u);
  if (has_shared) sv = *reinterpret_cast<const uint4*>(sout + (size_t)m * HID + h0);

  // The weights and the gate are read per thread rather than staged through
  // shared memory: every lane wants the same nine values, so they are one L1
  // broadcast each, and skipping the barrier lets the whole working set --
  // 8 expert slices plus the shared slice -- stay in flight together.
  float wk[TOPK];
#pragma unroll
  for (int k = 0; k < TOPK; ++k) wk[k] = tw[m * TOPK + k];
  const float g = has_shared ? bflo((unsigned int)sgate[m]) : 0.f;

  float acc[FVEC];
  {
    const unsigned int* w = reinterpret_cast<const unsigned int*>(&ev[0]);
#pragma unroll
    for (int i = 0; i < FVW; ++i) {
      acc[2 * i] = wk[0] * bflo(w[i]);
      acc[2 * i + 1] = wk[0] * bfhi(w[i]);
    }
  }
#pragma unroll
  for (int k = 1; k < TOPK; ++k) {
    const unsigned int* w = reinterpret_cast<const unsigned int*>(&ev[k]);
#pragma unroll
    for (int i = 0; i < FVW; ++i) {
      acc[2 * i] = fmaf(wk[k], bflo(w[i]), acc[2 * i]);
      acc[2 * i + 1] = fmaf(wk[k], bfhi(w[i]), acc[2 * i + 1]);
    }
  }
  if (has_shared) {
    const unsigned int* w = reinterpret_cast<const unsigned int*>(&sv);
#pragma unroll
    for (int i = 0; i < FVW; ++i) {
      acc[2 * i] = fmaf(g, bflo(w[i]), acc[2 * i]);
      acc[2 * i + 1] = fmaf(g, bfhi(w[i]), acc[2 * i + 1]);
    }
  }
  unsigned int ow[FVW];
#pragma unroll
  for (int i = 0; i < FVW; ++i) ow[i] = to_bf16(acc[2 * i]) | (to_bf16(acc[2 * i + 1]) << 16);
  *reinterpret_cast<uint4*>(o_out + (size_t)m * HID + h0) = *reinterpret_cast<const uint4*>(ow);
}

}  // namespace

// ---------------------------------------------------------------------------
// Out tensors are pre-allocated by the caller; every input is read-only.  Both
// entry points are stateless on the host: no allocation, no synchronization, no
// device limits and no stream attributes, so they record into a CUDA graph.
void moe_routing_prologue_sm120(TensorView hidden_states, TensorView gate_weight,
                                TensorView shared_gate_weight, TensorView router_logits,
                                TensorView shared_gate, TensorView topk_weights,
                                TensorView topk_ids, TensorView sorted_token_ids,
                                TensorView expert_ids, TensorView num_tokens_post_pad) {
  CHECK_INPUT_AND_TYPE(hidden_states, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(gate_weight, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(shared_gate_weight, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(router_logits, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(shared_gate, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(topk_weights, dl_float32);
  CHECK_INPUT_AND_TYPE(topk_ids, dl_int32);
  CHECK_INPUT_AND_TYPE(sorted_token_ids, dl_int32);
  CHECK_INPUT_AND_TYPE(expert_ids, dl_int32);
  CHECK_INPUT_AND_TYPE(num_tokens_post_pad, dl_int32);

  CHECK_DIM(2, hidden_states);
  CHECK_DIM(2, gate_weight);
  CHECK_DIM(2, shared_gate_weight);
  CHECK_DIM(2, router_logits);
  CHECK_DIM(1, shared_gate);
  CHECK_DIM(2, topk_weights);
  CHECK_DIM(2, topk_ids);
  CHECK_DIM(1, sorted_token_ids);
  CHECK_DIM(1, expert_ids);
  CHECK_DIM(1, num_tokens_post_pad);

  const int M = static_cast<int>(hidden_states.size(0));

  // The kernel is written for exactly one MoE geometry; everything else must
  // never reach it.  M is a runtime parameter, but the descriptor pass keeps
  // every assignment in one CTA, so it has a hard ceiling that is REPORTED --
  // a launcher that silently does nothing here is a wrong answer with no error.
  TVM_FFI_ICHECK_GE(M, 1);
  TVM_FFI_ICHECK_LE(M, MAXM) << "moe_routing_prologue serves at most " << MAXM
                             << " tokens per call, got " << M;
  TVM_FFI_ICHECK_EQ(hidden_states.size(1), HID) << "hidden size must be " << HID;
  TVM_FFI_ICHECK_EQ(gate_weight.size(0), NUM_EXPERTS) << "expert count must be " << NUM_EXPERTS;
  TVM_FFI_ICHECK_EQ(gate_weight.size(1), HID);
  TVM_FFI_ICHECK_EQ(shared_gate_weight.size(0), 1);
  TVM_FFI_ICHECK_EQ(shared_gate_weight.size(1), HID);
  TVM_FFI_ICHECK_EQ(router_logits.size(0), M);
  TVM_FFI_ICHECK_EQ(router_logits.size(1), NUM_EXPERTS);
  TVM_FFI_ICHECK_EQ(shared_gate.size(0), M);
  TVM_FFI_ICHECK_EQ(topk_weights.size(0), M);
  TVM_FFI_ICHECK_EQ(topk_weights.size(1), TOPK) << "top-k must be " << TOPK;
  TVM_FFI_ICHECK_EQ(topk_ids.size(0), M);
  TVM_FFI_ICHECK_EQ(topk_ids.size(1), TOPK);
  TVM_FFI_ICHECK_EQ(sorted_token_ids.size(0), 64 * M)
      << "sorted_token_ids must hold max_num_tokens_padded = 64 * M entries";
  TVM_FFI_ICHECK_EQ(expert_ids.size(0), BLOCK_M * M)
      << "expert_ids must hold max_num_m_blocks = 8 * M entries";
  TVM_FFI_ICHECK_EQ(num_tokens_post_pad.numel(), 1);

  // Every operand, input and output alike, on the device this launch guards
  // to.  CHECK_INPUT_AND_TYPE above only says "some CUDA tensor"; an output
  // allocated on a different device would be written through a pointer that
  // does not belong to the current context -- an illegal access at best.  The
  // align and finalize entry points check all of theirs, so these do too.
  CHECK_DEVICE(gate_weight, hidden_states);
  CHECK_DEVICE(shared_gate_weight, hidden_states);
  CHECK_DEVICE(router_logits, hidden_states);
  CHECK_DEVICE(shared_gate, hidden_states);
  CHECK_DEVICE(topk_weights, hidden_states);
  CHECK_DEVICE(topk_ids, hidden_states);
  CHECK_DEVICE(sorted_token_ids, hidden_states);
  CHECK_DEVICE(expert_ids, hidden_states);
  CHECK_DEVICE(num_tokens_post_pad, hidden_states);

  ffi::CUDADeviceGuard device_guard(hidden_states.device().device_id);
  const cudaStream_t stream = get_stream(hidden_states.device());

  moe_router_gemv<<<NUM_EXPERTS + 1, PT_R, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(hidden_states.data_ptr()),
      static_cast<const __nv_bfloat16*>(gate_weight.data_ptr()),
      static_cast<const __nv_bfloat16*>(shared_gate_weight.data_ptr()),
      static_cast<unsigned short*>(router_logits.data_ptr()),
      static_cast<unsigned short*>(shared_gate.data_ptr()), M);

  moe_routing_descriptor<<<1, PT_D, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(router_logits.data_ptr()), nullptr,
      static_cast<float*>(topk_weights.data_ptr()), static_cast<int*>(topk_ids.data_ptr()),
      static_cast<int*>(sorted_token_ids.data_ptr()), static_cast<int*>(expert_ids.data_ptr()),
      static_cast<int*>(num_tokens_post_pad.data_ptr()), M);

  const cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "moe_routing_prologue_sm120 failed with error code " << cudaGetErrorString(status);
}

// Standalone descriptor build for a caller that already has `topk_ids` from its
// own router: the block-aligned sort descriptors the expert GEMM consumes, in
// one launch instead of the count / scan / scatter / prefill chain a serving
// engine spends here.  Same kernel, same code path, as the prologue's second
// stage -- see moe_routing_descriptor.
void moe_routing_align_sm120(TensorView topk_ids, TensorView sorted_token_ids,
                             TensorView expert_ids, TensorView num_tokens_post_pad,
                             int64_t num_experts, int64_t block_size_m) {
  CHECK_INPUT_AND_TYPE(topk_ids, dl_int32);
  CHECK_INPUT_AND_TYPE(sorted_token_ids, dl_int32);
  CHECK_INPUT_AND_TYPE(expert_ids, dl_int32);
  CHECK_INPUT_AND_TYPE(num_tokens_post_pad, dl_int32);

  CHECK_DIM(2, topk_ids);
  CHECK_DIM(1, sorted_token_ids);
  CHECK_DIM(1, expert_ids);
  CHECK_DIM(1, num_tokens_post_pad);

  const int M = static_cast<int>(topk_ids.size(0));
  TVM_FFI_ICHECK_GE(M, 1);
  TVM_FFI_ICHECK_LE(M, MAXM) << "moe_routing_align serves at most " << MAXM
                             << " tokens per call, got " << M;
  TVM_FFI_ICHECK_EQ(topk_ids.size(1), TOPK) << "top-k must be " << TOPK;
  TVM_FFI_ICHECK_EQ(num_experts, NUM_EXPERTS) << "expert count must be " << NUM_EXPERTS;
  TVM_FFI_ICHECK_EQ(block_size_m, BLOCK_M) << "block_size_m must be " << BLOCK_M;
  TVM_FFI_ICHECK_EQ(sorted_token_ids.size(0), 64 * M)
      << "sorted_token_ids must hold max_num_tokens_padded = 64 * M entries";
  TVM_FFI_ICHECK_EQ(expert_ids.size(0), BLOCK_M * M)
      << "expert_ids must hold max_num_m_blocks = 8 * M entries";
  TVM_FFI_ICHECK_EQ(num_tokens_post_pad.numel(), 1);

  CHECK_DEVICE(sorted_token_ids, topk_ids);
  CHECK_DEVICE(expert_ids, topk_ids);
  CHECK_DEVICE(num_tokens_post_pad, topk_ids);

  ffi::CUDADeviceGuard device_guard(topk_ids.device().device_id);
  const cudaStream_t stream = get_stream(topk_ids.device());

  moe_routing_descriptor<<<1, PT_D, 0, stream>>>(
      nullptr, static_cast<const int*>(topk_ids.data_ptr()), nullptr, nullptr,
      static_cast<int*>(sorted_token_ids.data_ptr()), static_cast<int*>(expert_ids.data_ptr()),
      static_cast<int*>(num_tokens_post_pad.data_ptr()), M);

  const cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "moe_routing_align_sm120 failed with error code " << cudaGetErrorString(status);
}

// NOTE: `Optional` must be qualified.  csrc/tvm_ffi_utils.h pulls `Tensor` and
// `TensorView` into the global namespace but NOT `Optional`; the csrc files that
// use it bare get it transitively from a flashinfer header they also include.
// Relying on that is how this signature failed to compile.
void moe_routing_finalize_sm120(TensorView expert_out, ffi::Optional<TensorView> maybe_shared_out,
                                TensorView topk_weights,
                                ffi::Optional<TensorView> maybe_shared_gate, TensorView output) {
  CHECK_INPUT_AND_TYPE(expert_out, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(topk_weights, dl_float32);
  CHECK_INPUT_AND_TYPE(output, dl_bfloat16);

  CHECK_DIM(3, expert_out);
  CHECK_DIM(2, topk_weights);
  CHECK_DIM(2, output);

  const int M = static_cast<int>(expert_out.size(0));
  TVM_FFI_ICHECK_GE(M, 1);
  TVM_FFI_ICHECK_EQ(expert_out.size(1), TOPK) << "top-k must be " << TOPK;
  TVM_FFI_ICHECK_EQ(expert_out.size(2), HID) << "hidden size must be " << HID;
  TVM_FFI_ICHECK_EQ(topk_weights.size(0), M);
  TVM_FFI_ICHECK_EQ(topk_weights.size(1), TOPK);
  TVM_FFI_ICHECK_EQ(output.size(0), M);
  TVM_FFI_ICHECK_EQ(output.size(1), HID);

  CHECK_DEVICE(topk_weights, expert_out);
  CHECK_DEVICE(output, expert_out);

  // The shared expert is all-or-nothing: half of it would silently drop the
  // gate or read an unset buffer.
  TVM_FFI_ICHECK_EQ(maybe_shared_out.has_value(), maybe_shared_gate.has_value())
      << "shared_out and shared_gate must be given together or not at all";
  const __nv_bfloat16* sout_ptr = nullptr;
  const unsigned short* sgate_ptr = nullptr;
  if (maybe_shared_out.has_value()) {
    const auto& shared_out = maybe_shared_out.value();
    const auto& shared_gate = maybe_shared_gate.value();
    CHECK_INPUT_AND_TYPE(shared_out, dl_bfloat16);
    CHECK_INPUT_AND_TYPE(shared_gate, dl_bfloat16);
    CHECK_DIM(2, shared_out);
    CHECK_DIM(1, shared_gate);
    TVM_FFI_ICHECK_EQ(shared_out.size(0), M);
    TVM_FFI_ICHECK_EQ(shared_out.size(1), HID);
    TVM_FFI_ICHECK_EQ(shared_gate.size(0), M);
    CHECK_DEVICE(shared_out, expert_out);
    CHECK_DEVICE(shared_gate, expert_out);
    sout_ptr = static_cast<const __nv_bfloat16*>(shared_out.data_ptr());
    sgate_ptr = static_cast<const unsigned short*>(shared_gate.data_ptr());
  }

  ffi::CUDADeviceGuard device_guard(expert_out.device().device_id);
  const cudaStream_t stream = get_stream(expert_out.device());

  moe_finalize<<<FSPLIT * M, PT_F, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(expert_out.data_ptr()), sout_ptr,
      static_cast<const float*>(topk_weights.data_ptr()), sgate_ptr,
      static_cast<__nv_bfloat16*>(output.data_ptr()));

  const cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "moe_routing_finalize_sm120 failed with error code " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_routing_prologue_sm120, moe_routing_prologue_sm120);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_routing_align_sm120, moe_routing_align_sm120);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_routing_finalize_sm120, moe_routing_finalize_sm120);

#undef NUM_EXPERTS
#undef HID
#undef TOPK
#undef BLOCK_M
#undef MAXM
#undef PT_R
#undef RWARPS
#undef RCHUNK
#undef PT_D
#undef DWARPS
#undef PT_F
#undef FVEC
#undef FSPLIT
#undef FVW
