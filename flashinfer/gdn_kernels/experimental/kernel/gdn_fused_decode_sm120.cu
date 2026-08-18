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

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

#include "tvm_ffi_utils.h"

namespace {

// Fused GDN decode step for a fixed traced layer geometry: one persistent
// kernel covering the in_proj_ba GEMV, the depthwise causal conv1d update
// (width 4, silu), the q/k/v head split, and the gated delta-rule state
// update with qk-L2-norm, replacing the multi-launch serving chain.
constexpr int HIDDEN = 5120;
constexpr int N_BA = 96;
constexpr int QKV_DIM = 10240;
constexpr int H_Q = 16;
constexpr int HV = 48;
constexpr int D = 128;
constexpr int CONV_WIDTH = 4;
constexpr int CONV_STATE_LEN = 3;
constexpr int HEADS_PER_QK = 3;
constexpr int ROWS_PER_WARP = 8;
constexpr int GEMV_NSPLIT = 160;
// The gate reduction below unrolls the GEMV partials as 5 warp-wide loads.
static_assert(GEMV_NSPLIT == 5 * 32, "gate reduction assumes 5 warp-strided loads");

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float siluf(float x) { return x / (1.0f + __expf(-x)); }
__device__ __forceinline__ float sigmoidf(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float softplusf(float x) { return x > 20.0f ? x : log1pf(__expf(x)); }
__device__ __forceinline__ float warp_reduce(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
  return v;
}

// Reusable device-wide barrier (regular launch -> graph-capturable) that
// needs no host-side reset between launches: barrier[0] is the arrival
// counter (reset by the releasing block before it releases), barrier[1] is a
// monotonic release generation. Each block samples the generation at its own
// arrival; release of the current launch cannot precede that sample, and the
// previous launch's release is stream-ordered before this kernel starts, so
// the sampled value is uniform. The generation spin uses equality, so counter
// wrap-around is harmless, and the arrival reset keeps the barrier correct
// when consecutive launches use different grid sizes.
// Requires: barrier zero-initialized once at allocation, all blocks
// co-resident (grid capped to occupancy), and at most one kernel using the
// buffer in flight (single-stream use; the in-place state pools already
// require this).
__device__ __forceinline__ void grid_barrier(volatile unsigned* barrier, int nblocks) {
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned gen = barrier[1];
    __threadfence();
    unsigned arrived = atomicAdd((unsigned*)&barrier[0], 1u) + 1u;
    if (arrived == (unsigned)nblocks) {
      barrier[0] = 0u;  // reset the arrival counter for the next launch
      __threadfence();
      atomicAdd((unsigned*)&barrier[1], 1u);  // release everyone
    } else {
      while (barrier[1] == gen) { /* spin */
      }
    }
  }
  __syncthreads();
}

// Single persistent kernel: gemv+conv -> [barrier] -> delta. Regular launch.
// The kB1 instantiation specializes the serving-hot B=1 case: batch/split
// index math collapses to compile-time constants and the fp32 state rows for
// the (single) delta task of each warp are prefetched before the barrier so
// their long-scoreboard latency overlaps the gemv/conv phases (the state pool
// is only written after the barrier, each row by the warp that prefetched it).
//
// Aliasing: the op updates both state pools IN PLACE, so the launcher passes
// the same pointer for (conv_state, updated_conv) and for (ssm_state,
// ssm_out).  Those four parameters therefore carry no __restrict__ -- the
// pools are read and written through two different parameters, which is
// exactly what restrict promises does not happen, and promising it would let
// the compiler reorder a pool load across a pool store.  The remaining
// pointers are genuinely disjoint buffers and keep the qualifier.  The read
// path pays for this: loads from the pools can no longer be promoted to
// ld.global.nc.  Only this impl is affected -- the registry prefers the
// CuTe-DSL kernel for every shipped row -- and correctness is not a thing to
// trade for a read-only-cache hint.
template <bool kB1>
__global__ void gdn_fused_decode_kernel(
    const bf16* __restrict__ hidden, const bf16* __restrict__ w_ba,
    const bf16* __restrict__ mixed_qkv, const bf16* __restrict__ conv_weight,
    const bf16* __restrict__ conv_bias, const bf16* conv_state, const float* __restrict__ A_log,
    const bf16* __restrict__ dt_bias, const float* ssm_state, const int* __restrict__ state_indices,
    float scale, long state_stride_0, long qkv_stride, long conv_stride_p, long conv_stride_c,
    long conv_stride_t, bf16* __restrict__ output, bf16* updated_conv, float* ssm_out,
    float* __restrict__ ba_part, bf16* __restrict__ conv_out, unsigned* __restrict__ barrier,
    int B) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;
  const int Beff = kB1 ? 1 : B;

  // ---- Phase A1: GEMV partials ----
  // tasks: (split in 0..GEMV_NSPLIT-1) x (b) x (col in 0..95). Partials are
  // stored split-major per (col, b) so the gate reduction reads them with
  // warp-coalesced loads.
  long gemv_tasks = (long)GEMV_NSPLIT * Beff * N_BA;
  for (long t = tid; t < gemv_tasks; t += nthreads) {
    int col = t % N_BA;
    long r = t / N_BA;
    int b;
    int split;
    if constexpr (kB1) {
      b = 0;
      split = (int)r;
    } else {
      b = r % B;
      split = r / B;
    }
    const bf16* hrow = hidden + (long)b * HIDDEN;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    int k = split;
    for (; k + 3 * GEMV_NSPLIT < HIDDEN; k += 4 * GEMV_NSPLIT) {
      a0 += __bfloat162float(hrow[k]) * __bfloat162float(w_ba[(long)k * N_BA + col]);
      a1 += __bfloat162float(hrow[k + GEMV_NSPLIT]) *
            __bfloat162float(w_ba[(long)(k + GEMV_NSPLIT) * N_BA + col]);
      a2 += __bfloat162float(hrow[k + 2 * GEMV_NSPLIT]) *
            __bfloat162float(w_ba[(long)(k + 2 * GEMV_NSPLIT) * N_BA + col]);
      a3 += __bfloat162float(hrow[k + 3 * GEMV_NSPLIT]) *
            __bfloat162float(w_ba[(long)(k + 3 * GEMV_NSPLIT) * N_BA + col]);
    }
    for (; k < HIDDEN; k += GEMV_NSPLIT)
      a0 += __bfloat162float(hrow[k]) * __bfloat162float(w_ba[(long)k * N_BA + col]);
    ba_part[((long)col * Beff + b) * GEMV_NSPLIT + split] = (a0 + a1) + (a2 + a3);
  }

  // ---- Phase A2: conv (independent of gemv) ----
  long conv_tasks = (long)Beff * QKV_DIM;
  for (long t = tid; t < conv_tasks; t += nthreads) {
    int b;
    int c;
    if constexpr (kB1) {
      b = 0;
      c = (int)t;
    } else {
      b = t / QKV_DIM;
      c = t % QKV_DIM;
    }
    int idx = state_indices[b];
    // conv_state addressing is stride-parameterized: the pool arrives as a
    // logical [P, QKV_DIM, CONV_STATE_LEN] view of either a DS-dense pool
    // (strides p,3,1 -> per-thread 3-element rows) or a transposed SD pool
    // (strides p,1,QKV_DIM -> fully coalesced across channels, the vLLM
    // default). Pure index arithmetic; the update math is identical.
    const bf16* st = conv_state + (long)idx * conv_stride_p + (long)c * conv_stride_c;
    bf16 s0 = st[0], s1 = st[conv_stride_t], s2 = st[2 * conv_stride_t];
    // mixed_qkv rows may be strided (e.g. a view into a wider projection).
    bf16 xr = mixed_qkv[(long)b * qkv_stride + c];
    const bf16* w = conv_weight + (long)c * CONV_WIDTH;
    float y = __bfloat162float(s0) * __bfloat162float(w[0]) +
              __bfloat162float(s1) * __bfloat162float(w[1]) +
              __bfloat162float(s2) * __bfloat162float(w[2]) +
              __bfloat162float(xr) * __bfloat162float(w[3]) + __bfloat162float(conv_bias[c]);
    conv_out[(long)b * QKV_DIM + c] = __float2bfloat16(siluf(y));
    bf16* uc = updated_conv + (long)idx * conv_stride_p + (long)c * conv_stride_c;
    uc[0] = s1;
    uc[conv_stride_t] = s2;
    uc[2 * conv_stride_t] = xr;
  }

  // ---- Pre-barrier prefetch of this warp's first delta task's state rows.
  // The state pool is read-only until phase C, and each row is written only
  // by the warp that owns (and prefetched) it, so this is race-free.
  int gwarp = tid >> 5;
  int lane = threadIdx.x & 31;
  int nwarps = nthreads >> 5;
  long total_rows = (long)Beff * HV * D;
  long warps_needed = (total_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;

  float4 s_pre[ROWS_PER_WARP];
  long pre_row_base = -1;
  if (gwarp < warps_needed) {
    long first_row = (long)gwarp * ROWS_PER_WARP;
    int v0;
    int h;
    int b;
    if constexpr (kB1) {
      v0 = (int)(first_row & (D - 1));
      h = (int)(first_row >> 7);
      b = 0;
    } else {
      v0 = first_row % D;
      long tmp = first_row / D;
      h = tmp % HV;
      b = tmp / HV;
    }
    int idx = state_indices[b];
    pre_row_base = (long)idx * state_stride_0 + (long)h * (D * D);
    const float4* base_srow = (const float4*)(ssm_state + pre_row_base + (long)v0 * D);
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) s_pre[r] = base_srow[r * (D / 4) + lane];
  }

  grid_barrier(barrier, gridDim.x);

  // ---- Phase C: delta (gate reduced inline from ba_part) ----
  for (long w = gwarp; w < warps_needed; w += nwarps) {
    long first_row = w * ROWS_PER_WARP;
    int v0;
    int h;
    int b;
    if constexpr (kB1) {
      v0 = (int)(first_row & (D - 1));
      h = (int)(first_row >> 7);
      b = 0;
    } else {
      v0 = first_row % D;
      long tmp = first_row / D;
      h = tmp % HV;
      b = tmp / HV;
    }
    int j = h / HEADS_PER_QK;
    const bf16* co = conv_out + (long)b * QKV_DIM;
    const bf16* qb = co + j * D;
    const bf16* kb = co + H_Q * D + j * D;
    int k0 = lane * 4;
    int idx = state_indices[b];
    long row_base = (long)idx * state_stride_0 + (long)h * (D * D);
    float4 s4[ROWS_PER_WARP];
    if (w == gwarp && row_base == pre_row_base) {
      // First iteration (the only one for B=1): use the prefetched rows.
#pragma unroll
      for (int r = 0; r < ROWS_PER_WARP; ++r) s4[r] = s_pre[r];
    } else {
      const float4* base_srow = (const float4*)(ssm_state + row_base + (long)v0 * D);
#pragma unroll
      for (int r = 0; r < ROWS_PER_WARP; ++r) s4[r] = base_srow[r * (D / 4) + lane];
    }
    // Issue the gate-partial loads early (10 concurrent warp-wide loads) so
    // they overlap the qk-norm compute below.
    const float* base_b = ba_part + ((long)h * Beff + b) * GEMV_NSPLIT;
    const float* base_a = ba_part + ((long)(HV + h) * Beff + b) * GEMV_NSPLIT;
    float b0 = base_b[lane + 0];
    float a0v = base_a[lane + 0];
    float b1 = base_b[lane + 32];
    float a1v = base_a[lane + 32];
    float b2 = base_b[lane + 64];
    float a2v = base_a[lane + 64];
    float b3 = base_b[lane + 96];
    float a3v = base_a[lane + 96];
    float b4 = base_b[lane + 128];
    float a4v = base_a[lane + 128];
    float qraw[4], kraw[4];
    float qss = 0.f, kss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      qraw[i] = __bfloat162float(qb[k0 + i]);
      kraw[i] = __bfloat162float(kb[k0 + i]);
      qss += qraw[i] * qraw[i];
      kss += kraw[i] * kraw[i];
    }
    qss = warp_reduce(qss);
    kss = warp_reduce(kss);
    qss = __shfl_sync(0xffffffff, qss, 0);
    kss = __shfl_sync(0xffffffff, kss, 0);
    float qn = rsqrtf(qss + 1e-6f), kn = rsqrtf(kss + 1e-6f);
    float qh[4], kh[4], QKp = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      qh[i] = qraw[i] * qn;
      kh[i] = kraw[i] * kn;
      QKp += qh[i] * kh[i];
    }
    QKp = warp_reduce(QKp);
    float QK = __shfl_sync(0xffffffff, QKp, 0);
    // gate g,beta reduced from the split-major partials (values now arrived)
    float accb = ((b0 + b1) + (b2 + b3)) + b4;
    float acca = ((a0v + a1v) + (a2v + a3v)) + a4v;
    accb = warp_reduce(accb);
    acca = warp_reduce(acca);
    accb = __shfl_sync(0xffffffff, accb, 0);
    acca = __shfl_sync(0xffffffff, acca, 0);
    // The bf16 round-trip of the two gate sums is load-bearing, not a leftover:
    // the composable path materializes `ba = (hidden @ w_ba)` as a bf16 tensor
    // and only then widens it for the gates, so the values it feeds
    // sigmoid/softplus are bf16-rounded. Keeping the fp32 accumulator here
    // would make this kernel *more* precise than the operation it implements
    // and move the gates off the reference by up to one bf16 ulp -- amplified
    // by exp() in the decay gate. Track the composable path's `ba` dtype, not
    // the accumulator's.
    float beta = sigmoidf(__bfloat162float(__float2bfloat16(accb)));
    float xg = __bfloat162float(__float2bfloat16(acca)) + __bfloat162float(dt_bias[h]);
    float g = __expf(-__expf(A_log[h]) * softplusf(xg));
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
      int v = v0 + r;
      float s[4] = {s4[r].x, s4[r].y, s4[r].z, s4[r].w};
      float vv = __bfloat162float(co[2 * H_Q * D + h * D + v]);
      float kSp = 0.f, qSp = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        kSp += kh[i] * s[i];
        qSp += qh[i] * s[i];
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        kSp += __shfl_down_sync(0xffffffff, kSp, o);
        qSp += __shfl_down_sync(0xffffffff, qSp, o);
      }
      float kS = __shfl_sync(0xffffffff, kSp, 0);
      float qS = __shfl_sync(0xffffffff, qSp, 0);
      float old_v = g * kS;
      float delta = beta * (vv - old_v);
      float out_v = scale * (g * qS + delta * QK);
      float4* Sorow = (float4*)(ssm_out + row_base + (long)v * D);
      float4 o4;
      o4.x = g * s[0] + kh[0] * delta;
      o4.y = g * s[1] + kh[1] * delta;
      o4.z = g * s[2] + kh[2] * delta;
      o4.w = g * s[3] + kh[3] * delta;
      Sorow[lane] = o4;
      if (lane == 0) output[((long)b * HV + h) * D + v] = __float2bfloat16(out_v);
    }
  }
}

void gdn_fused_decode_launch(const void* hidden, const void* w_ba, const void* mixed_qkv,
                             const void* conv_weight, const void* conv_bias, void* conv_state,
                             const void* A_log, const void* dt_bias, float scale, void* ssm_state,
                             const void* state_indices, void* output, void* conv_out_scratch,
                             void* ba_scratch, void* barrier, long state_stride_0, long qkv_stride,
                             long conv_stride_p, long conv_stride_c, long conv_stride_t, int B,
                             cudaStream_t stream) {
  constexpr int kBlock = 256;
  constexpr int kMaxDevices = 64;
  static int blocks_per_sm_b1[kMaxDevices] = {0};
  static int blocks_per_sm_gen[kMaxDevices] = {0};
  static int num_sm[kMaxDevices] = {0};
  int dev = 0;
  cudaGetDevice(&dev);
  if (dev < 0 || dev >= kMaxDevices) dev = 0;
  if (num_sm[dev] == 0) {
    cudaDeviceGetAttribute(&num_sm[dev], cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm_b1[dev],
                                                  gdn_fused_decode_kernel<true>, kBlock, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm_gen[dev],
                                                  gdn_fused_decode_kernel<false>, kBlock, 0);
  }
  // Grid must be <= resident capacity (persistent barrier needs co-residency),
  // but no larger than needed. Delta single-pass needs ceil(rows/ROWS_PER_WARP/8) blocks.
  long rows = (long)B * HV * D;
  long warps_needed = (rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
  int delta_blocks = (int)((warps_needed + 7) / 8);
  int cap = (B == 1 ? blocks_per_sm_b1[dev] : blocks_per_sm_gen[dev]) * num_sm[dev];
  int grid = delta_blocks < cap ? delta_blocks : cap;
  if (grid < 1) grid = 1;
  // No per-call barrier reset: the barrier is zero-initialized at allocation
  // and self-resetting across launches (see grid_barrier).
  if (B == 1) {
    gdn_fused_decode_kernel<true><<<grid, kBlock, 0, stream>>>(
        static_cast<const bf16*>(hidden), static_cast<const bf16*>(w_ba),
        static_cast<const bf16*>(mixed_qkv), static_cast<const bf16*>(conv_weight),
        static_cast<const bf16*>(conv_bias), static_cast<const bf16*>(conv_state),
        static_cast<const float*>(A_log), static_cast<const bf16*>(dt_bias),
        static_cast<const float*>(ssm_state), static_cast<const int*>(state_indices), scale,
        state_stride_0, qkv_stride, conv_stride_p, conv_stride_c, conv_stride_t,
        static_cast<bf16*>(output), static_cast<bf16*>(conv_state), static_cast<float*>(ssm_state),
        static_cast<float*>(ba_scratch), static_cast<bf16*>(conv_out_scratch),
        static_cast<unsigned*>(barrier), B);
  } else {
    gdn_fused_decode_kernel<false><<<grid, kBlock, 0, stream>>>(
        static_cast<const bf16*>(hidden), static_cast<const bf16*>(w_ba),
        static_cast<const bf16*>(mixed_qkv), static_cast<const bf16*>(conv_weight),
        static_cast<const bf16*>(conv_bias), static_cast<const bf16*>(conv_state),
        static_cast<const float*>(A_log), static_cast<const bf16*>(dt_bias),
        static_cast<const float*>(ssm_state), static_cast<const int*>(state_indices), scale,
        state_stride_0, qkv_stride, conv_stride_p, conv_stride_c, conv_stride_t,
        static_cast<bf16*>(output), static_cast<bf16*>(conv_state), static_cast<float*>(ssm_state),
        static_cast<float*>(ba_scratch), static_cast<bf16*>(conv_out_scratch),
        static_cast<unsigned*>(barrier), B);
  }
}

}  // namespace

void gdn_fused_decode(TensorView hidden_states, TensorView w_ba, TensorView mixed_qkv,
                      TensorView conv_weight, TensorView conv_bias, TensorView conv_state,
                      TensorView A_log, TensorView dt_bias, TensorView ssm_state,
                      TensorView state_indices, TensorView output, TensorView conv_out_scratch,
                      TensorView ba_scratch, TensorView barrier, double scale) {
  CHECK_INPUT_TYPE(hidden_states, dl_bfloat16);
  CHECK_INPUT_TYPE(w_ba, dl_bfloat16);
  CHECK_INPUT_TYPE(mixed_qkv, dl_bfloat16);
  CHECK_INPUT_TYPE(conv_weight, dl_bfloat16);
  CHECK_INPUT_TYPE(conv_bias, dl_bfloat16);
  CHECK_INPUT_TYPE(conv_state, dl_bfloat16);
  CHECK_INPUT_TYPE(A_log, dl_float32);
  CHECK_INPUT_TYPE(dt_bias, dl_bfloat16);
  CHECK_INPUT_TYPE(ssm_state, dl_float32);
  CHECK_INPUT_TYPE(state_indices, dl_int32);
  CHECK_INPUT_TYPE(output, dl_bfloat16);
  CHECK_INPUT_TYPE(conv_out_scratch, dl_bfloat16);
  CHECK_INPUT_TYPE(ba_scratch, dl_float32);
  CHECK_INPUT_TYPE(barrier, dl_int32);

  CHECK_DIM(2, hidden_states);
  CHECK_DIM(2, w_ba);
  CHECK_DIM(2, mixed_qkv);
  CHECK_DIM(2, conv_weight);
  CHECK_DIM(1, conv_bias);
  CHECK_DIM(3, conv_state);
  CHECK_DIM(1, A_log);
  CHECK_DIM(1, dt_bias);
  CHECK_DIM(4, ssm_state);
  CHECK_DIM(1, state_indices);
  CHECK_DIM(4, output);
  CHECK_DIM(2, conv_out_scratch);
  CHECK_DIM(1, ba_scratch);
  CHECK_DIM(1, barrier);

  const int B = static_cast<int>(hidden_states.size(0));
  TVM_FFI_ICHECK_GT(B, 0);
  TVM_FFI_ICHECK_EQ(hidden_states.size(1), HIDDEN);
  TVM_FFI_ICHECK_EQ(w_ba.size(0), HIDDEN);
  TVM_FFI_ICHECK_EQ(w_ba.size(1), N_BA);
  TVM_FFI_ICHECK_EQ(mixed_qkv.size(0), B);
  TVM_FFI_ICHECK_EQ(mixed_qkv.size(1), QKV_DIM);
  TVM_FFI_ICHECK_EQ(conv_weight.size(0), QKV_DIM);
  TVM_FFI_ICHECK_EQ(conv_weight.size(1), CONV_WIDTH);
  TVM_FFI_ICHECK_EQ(conv_bias.size(0), QKV_DIM);
  TVM_FFI_ICHECK_EQ(conv_state.size(1), QKV_DIM);
  TVM_FFI_ICHECK_EQ(conv_state.size(2), CONV_STATE_LEN);
  TVM_FFI_ICHECK_EQ(A_log.size(0), HV);
  TVM_FFI_ICHECK_EQ(dt_bias.size(0), HV);
  TVM_FFI_ICHECK_EQ(ssm_state.size(1), HV);
  TVM_FFI_ICHECK_EQ(ssm_state.size(2), D);
  TVM_FFI_ICHECK_EQ(ssm_state.size(3), D);
  TVM_FFI_ICHECK_EQ(state_indices.size(0), B);
  TVM_FFI_ICHECK_EQ(output.size(0), B);
  TVM_FFI_ICHECK_EQ(output.size(1), 1);
  TVM_FFI_ICHECK_EQ(output.size(2), HV);
  TVM_FFI_ICHECK_EQ(output.size(3), D);
  TVM_FFI_ICHECK_EQ(conv_out_scratch.size(0), B);
  TVM_FFI_ICHECK_EQ(conv_out_scratch.size(1), QKV_DIM);
  TVM_FFI_ICHECK_EQ(ba_scratch.size(0), (long)GEMV_NSPLIT * B * N_BA);
  TVM_FFI_ICHECK_EQ(barrier.size(0), 2);

  // Dense inner layout everywhere; the fp32 state pool may have a padded row
  // stride (stride(0) >= HV*D*D), which the kernel consumes directly.
  TVM_FFI_ICHECK_EQ(hidden_states.stride(1), 1);
  TVM_FFI_ICHECK_EQ(hidden_states.stride(0), HIDDEN);
  TVM_FFI_ICHECK_EQ(w_ba.stride(1), 1);
  TVM_FFI_ICHECK_EQ(w_ba.stride(0), N_BA);
  TVM_FFI_ICHECK_EQ(mixed_qkv.stride(1), 1);
  // mixed_qkv rows may be strided (a view into a wider fused projection).
  TVM_FFI_ICHECK_GE(mixed_qkv.stride(0), QKV_DIM);
  TVM_FFI_ICHECK_EQ(conv_weight.stride(1), 1);
  TVM_FFI_ICHECK_EQ(conv_weight.stride(0), CONV_WIDTH);
  // conv_state is a logical [P, QKV_DIM, CONV_STATE_LEN] view over one of the
  // two serving pool layouts (page stride may be padded):
  //  - DS pool (dim-first, dense rows): strides (p, CONV_STATE_LEN, 1);
  //  - SD pool (vLLM default; physical [P, CONV_STATE_LEN, QKV_DIM] passed as
  //    its transpose): strides (p, 1, QKV_DIM).
  {
    const long conv_stride_c = static_cast<long>(conv_state.stride(1));
    const long conv_stride_t = static_cast<long>(conv_state.stride(2));
    const bool conv_ds = conv_stride_c == CONV_STATE_LEN && conv_stride_t == 1;
    const bool conv_sd = conv_stride_c == 1 && conv_stride_t == QKV_DIM;
    TVM_FFI_ICHECK(conv_ds || conv_sd)
        << "conv_state must be a DS-dense pool or a transposed SD pool view";
    TVM_FFI_ICHECK_GE(conv_state.stride(0), (long)QKV_DIM * CONV_STATE_LEN);
  }
  TVM_FFI_ICHECK_EQ(ssm_state.stride(3), 1);
  TVM_FFI_ICHECK_EQ(ssm_state.stride(2), D);
  TVM_FFI_ICHECK_EQ(ssm_state.stride(1), (long)D * D);
  TVM_FFI_ICHECK_GE(ssm_state.stride(0), (long)HV * D * D);
  TVM_FFI_ICHECK_EQ(output.stride(3), 1);
  TVM_FFI_ICHECK_EQ(output.stride(2), D);

  const long state_stride_0 = static_cast<long>(ssm_state.stride(0));
  const long qkv_stride = static_cast<long>(mixed_qkv.stride(0));
  const long conv_stride_p = static_cast<long>(conv_state.stride(0));
  const long conv_stride_c = static_cast<long>(conv_state.stride(1));
  const long conv_stride_t = static_cast<long>(conv_state.stride(2));

  ffi::CUDADeviceGuard device_guard(hidden_states.device().device_id);
  const cudaStream_t stream = get_stream(hidden_states.device());
  gdn_fused_decode_launch(
      hidden_states.data_ptr(), w_ba.data_ptr(), mixed_qkv.data_ptr(), conv_weight.data_ptr(),
      conv_bias.data_ptr(), conv_state.data_ptr(), A_log.data_ptr(), dt_bias.data_ptr(),
      static_cast<float>(scale), ssm_state.data_ptr(), state_indices.data_ptr(), output.data_ptr(),
      conv_out_scratch.data_ptr(), ba_scratch.data_ptr(), barrier.data_ptr(), state_stride_0,
      qkv_stride, conv_stride_p, conv_stride_c, conv_stride_t, B, stream);

  cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "gdn_fused_decode failed with error code " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gdn_fused_decode, gdn_fused_decode);
