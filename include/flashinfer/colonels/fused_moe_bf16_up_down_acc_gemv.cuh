#pragma once

#include <cooperative_groups.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <stdio.h>

#include <cassert>
#include <cstdint>
#include <cuda/ptx>
#include <cuda/std/limits>

#include "fused_moe_activation.cuh"
#include "fused_moe_helpers.cuh"

using bf16 = __nv_bfloat16;
using bf16_2 = __nv_bfloat162;

#define AU 32
#define AD 16

__device__ __forceinline__ float vec_fma_bf16_bf16(const float4 a, const float4 b) {
  float sum = 0;
  const bf16_2* a_bf162_1 = (bf16_2*)&a.x;
  const bf16_2* a_bf162_2 = (bf16_2*)&a.y;
  const bf16_2* a_bf162_3 = (bf16_2*)&a.z;
  const bf16_2* a_bf162_4 = (bf16_2*)&a.w;
  const bf16_2* b_bf16_1 = (bf16_2*)&b.x;
  const bf16_2* b_bf16_2 = (bf16_2*)&b.y;
  const bf16_2* b_bf16_3 = (bf16_2*)&b.z;
  const bf16_2* b_bf16_4 = (bf16_2*)&b.w;
  sum += static_cast<float>(a_bf162_1->x) * static_cast<float>(b_bf16_1->x);
  sum += static_cast<float>(a_bf162_1->y) * static_cast<float>(b_bf16_1->y);
  sum += static_cast<float>(a_bf162_2->x) * static_cast<float>(b_bf16_2->x);
  sum += static_cast<float>(a_bf162_2->y) * static_cast<float>(b_bf16_2->y);
  sum += static_cast<float>(a_bf162_3->x) * static_cast<float>(b_bf16_3->x);
  sum += static_cast<float>(a_bf162_3->y) * static_cast<float>(b_bf16_3->y);
  sum += static_cast<float>(a_bf162_4->x) * static_cast<float>(b_bf16_4->x);
  sum += static_cast<float>(a_bf162_4->y) * static_cast<float>(b_bf16_4->y);
  return sum;
}

__device__ __forceinline__ float vec_fma_fp32_bf16(const float4 a, const float2 b) {
  float sum = 0;
  const bf16* b_bf16 = reinterpret_cast<const bf16*>(&b);
  sum += a.x * (float)b_bf16[0];
  sum += a.y * (float)b_bf16[1];
  sum += a.z * (float)b_bf16[2];
  sum += a.w * (float)b_bf16[3];
  return sum;
}

template <int THREADS = 32>
__device__ __forceinline__ float reduce_sum_bfly(float sum) {
  if (THREADS >= 32) sum += __shfl_xor_sync(0xffffffff, sum, 16);
  if (THREADS >= 16) sum += __shfl_xor_sync(0xffffffff, sum, 8);
  if (THREADS >= 8) sum += __shfl_xor_sync(0xffffffff, sum, 4);
  if (THREADS >= 4) sum += __shfl_xor_sync(0xffffffff, sum, 2);
  if (THREADS >= 2) sum += __shfl_xor_sync(0xffffffff, sum, 1);
  return sum;
}

template <int BN, int WARPS, int Z_SPLIT, moe_activation::Kind ACTIVATION>
__global__ __launch_bounds__(WARPS * 32, 1) void fused_moe_bf16_wgmma_up_down_acc_gemv_kernel(
    const bf16* x, const bf16* w13, const bf16* w2, bf16* __restrict__ out,
    const int* __restrict__ topk_ids, const float* __restrict__ topk_weights, const int top_k,
    int M, int N, int K, int num_experts, float scaling_factor,
    typename moe_activation::Traits<ACTIVATION>::Params activation_params) {
  using Activation = moe_activation::Traits<ACTIVATION>;
  constexpr int UP_FACTOR = Activation::GATED ? 2 : 1;
  const int K2 = N / UP_FACTOR;
  const int N2 = K;

  // Loads per warp
  constexpr int F4_TO_BF16_RATIO = sizeof(float4) / sizeof(bf16);
  constexpr int F2_TO_BF16_RATIO = sizeof(float2) / sizeof(bf16);
  constexpr int BK = F4_TO_BF16_RATIO * 32;
  // After up projection we store x in float, so each lane loads a
  // float2 (= 4 bf16) per K2 stride in the down proj.
  //
  // Gated activations halve the UP width; ungated activations retain it.
  // Extra lanes split across output rows when the K2 slice is narrow.
  constexpr int K2_PER_BLOCK = BN / UP_FACTOR;
  constexpr int LANES_K_DOWN = K2_PER_BLOCK / 4 < 32 ? K2_PER_BLOCK / 4 : 32;
  constexpr int N_SPLIT_LANES = 32 / LANES_K_DOWN;
  // bf16 elements covered per warp-stage along K2 (= 4 * LANES_K_DOWN).
  constexpr int BK2 = 4 * LANES_K_DOWN;
  static_assert(BK2 <= K2_PER_BLOCK, "BK2 exceeds the K2 slice produced by the activation");

  const int N_STAGES_UP = K / BK;
  constexpr int N_STAGES_DOWN = K2_PER_BLOCK / BK2;
  static_assert(N_STAGES_DOWN > 0);

  constexpr int WN = BN / WARPS;
  // w1 and w3 are interleaved in blocks of 8 and we need
  // results from 2 blocks to do swiglu
  static_assert(WN % 16 == 0);

  constexpr int MAX_ACC_UP = WN < AU ? WN : AU;
  // MAX_ACC_DOWN controls how many N2 rows each group processes per outer
  // iteration of the down loop.  The final store packs 8 bf16 (= one float4)
  // per group per chunk, so MAX_ACC_DOWN must be at least 8 (and a multiple
  // of 8).
  constexpr int MAX_ACC_DOWN_RAW = AD / N_SPLIT_LANES;
  constexpr int MAX_ACC_DOWN = MAX_ACC_DOWN_RAW < 8 ? 8 : MAX_ACC_DOWN_RAW;
  static_assert(MAX_ACC_DOWN % 8 == 0);

  const int lane_id = threadIdx.x % 32;
  const int warp_id = threadIdx.x / 32;
  const int32_t warpM = blockIdx.y;
  const int32_t warpN = blockIdx.x * WARPS + warp_id;
  int token = warpM / top_k;

  const int expert_id = topk_ids[warpM];
  // vLLM uses -1 for padded routes, including CUDA graph capture inputs.
  if (expert_id < 0 || expert_id >= num_experts) {
    return;
  }

  const float4* x_local_vec = reinterpret_cast<const float4*>(x + token * K);
  // The full expert bank can contain more than INT_MAX elements.
  const float4* w13_local_vec =
      reinterpret_cast<const float4*>(w13 + size_t(expert_id) * N * K + warpN * WN * K);

  const bf16* w2_local_vec = w2 + size_t(expert_id) * N2 * K2 + blockIdx.z * (N2 / Z_SPLIT) * K2 +
                             blockIdx.x * K2_PER_BLOCK;

  // lane_id is decomposed as
  //   group_id     = lane_id / LANES_K_DOWN   in [0, N_SPLIT_LANES)
  //   lane_in_grp  = lane_id % LANES_K_DOWN   in [0, LANES_K_DOWN)
  // Lanes with the same group_id cooperate on a K2-reduction across
  // LANES_K_DOWN lanes; different groups handle disjoint slices of N2.
  const int group_id = lane_id / LANES_K_DOWN;
  const int lane_in_grp = lane_id % LANES_K_DOWN;

  float acc[MAX_ACC_UP];
  __shared__ float4 x2[K2_PER_BLOCK / 4];

  constexpr int SMEM_COLS = K2_PER_BLOCK;
  constexpr int SMEM_WARP_ROWS = MAX_ACC_DOWN * N_SPLIT_LANES;
  constexpr int SMEM_WARP_OFF = SMEM_WARP_ROWS * SMEM_COLS;
  constexpr int SMEM_ROWS = SMEM_WARP_ROWS * WARPS;
  constexpr int SMEM_SIZE = SMEM_ROWS * SMEM_COLS;
  // Total N2 rows owned by this block (across all (warp, group) outputs).
  const int BN2 = N2 / Z_SPLIT;
  extern __shared__ __align__(1024) uint8_t sh[];
  bf16* w2_smem = reinterpret_cast<bf16*>(sh);
  bf16* result = w2_smem + 2 * SMEM_SIZE;
  const float2* w2_local_smem = reinterpret_cast<float2*>(w2_smem + warp_id * SMEM_WARP_OFF);
  int load_stage = 0;
  int smem_stage = 0;
  auto load_w2 = [&](int acc_idx) {
    for (int i = threadIdx.x * F4_TO_BF16_RATIO; i < SMEM_SIZE;
         i += blockDim.x * F4_TO_BF16_RATIO) {
      int k_off = i % SMEM_COLS;
      int r = i / SMEM_COLS;
      int n_off = (r + acc_idx) * K2;
      int n_off_smem = r * SMEM_COLS;

      int smem_off = (load_stage * SMEM_SIZE + n_off_smem + k_off);
      int gmem_off = n_off + k_off;
      uint32_t smem = __cvta_generic_to_shared(w2_smem + smem_off);
      CP_ASYNC_CG(smem, w2_local_vec + gmem_off, 16);
    }
    load_stage ^= 1;

    CP_ASYNC_COMMIT_GROUP();
  };
  // W2 is prescheduled so that we have more bytes in flight
  load_w2(0);
  if (SMEM_ROWS < BN2) load_w2(SMEM_ROWS);

  for (int acc_idx = 0; acc_idx < WN; acc_idx += MAX_ACC_UP) {
    memset(acc, 0, sizeof(acc));
    for (int compute_stage = 0; compute_stage < N_STAGES_UP; compute_stage += 1) {
      int k_off = compute_stage * 32 + lane_id;
      const float4 x_val = x_local_vec[k_off];
      for (int n = 0; n < MAX_ACC_UP; n++) {
        // K is vector-aligned; divide the stride before forming each row offset.
        int n_off = (acc_idx + n) * (K / F4_TO_BF16_RATIO);
        const float4 w13_val = w13_local_vec[n_off + k_off];
        acc[n] += vec_fma_bf16_bf16(x_val, w13_val);
      }
    }
    for (int i = 0; i < MAX_ACC_UP; i++) {
      float local_sum = acc[i];
      float warp_sum = reduce_sum_bfly(local_sum);
      acc[i] = warp_sum;
    }

    constexpr int ACC_PER_RESULT = 4 * UP_FACTOR;
    constexpr int RES_PER_WARP = MAX_ACC_UP / ACC_PER_RESULT;
    const float topk_scale = topk_weights[warpM] * scaling_factor;

#pragma unroll
    for (int i = 0; i < RES_PER_WARP; i++) {
      // A bit hacky way but others look like dynamic indexing to the compiler
      // and it spills to local
      if (lane_id == i) {
        float4 activated;
        if constexpr (Activation::GATED) {
          // Gate and value rows are interleaved in groups of eight.
          const int x_idx = (i / 2) * 16 + (i % 2) * 4;
          const int w_idx = x_idx + 8;
          activated.x =
              moe_activation::apply<Activation>(acc[x_idx], acc[w_idx], activation_params);
          activated.y =
              moe_activation::apply<Activation>(acc[x_idx + 1], acc[w_idx + 1], activation_params);
          activated.z =
              moe_activation::apply<Activation>(acc[x_idx + 2], acc[w_idx + 2], activation_params);
          activated.w =
              moe_activation::apply<Activation>(acc[x_idx + 3], acc[w_idx + 3], activation_params);
        } else {
          activated.x = moe_activation::apply<Activation>(acc[i * 4], activation_params);
          activated.y = moe_activation::apply<Activation>(acc[i * 4 + 1], activation_params);
          activated.z = moe_activation::apply<Activation>(acc[i * 4 + 2], activation_params);
          activated.w = moe_activation::apply<Activation>(acc[i * 4 + 3], activation_params);
        }
        activated.x *= topk_scale;
        activated.y *= topk_scale;
        activated.z *= topk_scale;
        activated.w *= topk_scale;
        x2[(warp_id * WN + acc_idx) / ACC_PER_RESULT + i] = activated;
      }
    }
  }
  __syncthreads();
  float acc2[MAX_ACC_DOWN];
  bf16 acc2_bf16[MAX_ACC_DOWN];
  // Linearised (warp, group) id; identifies which MAX_ACC_DOWN rows of the
  // current SMEM_ROWS-row slab chunk this lane owns.
  const int gi = warp_id * N_SPLIT_LANES + group_id;
  for (int slab_row = 0; slab_row < BN2; slab_row += SMEM_ROWS) {
    memset(acc2, 0, sizeof(acc2));
    if (slab_row + SMEM_ROWS < BN2) {
      CP_ASYNC_WAIT_GROUP(1);
    } else {
      CP_ASYNC_WAIT_GROUP(0);
    }
    __syncthreads();

    for (int compute_stage = 0; compute_stage < N_STAGES_DOWN; compute_stage += 1) {
      int k_off = compute_stage * LANES_K_DOWN + lane_in_grp;
      float4 x_val = x2[k_off];
#pragma unroll
      for (int n = 0; n < MAX_ACC_DOWN; n++) {
        int n_off = (group_id * MAX_ACC_DOWN + n) * SMEM_COLS;
        int smem_off = smem_stage * SMEM_SIZE + n_off + k_off * F2_TO_BF16_RATIO;
        const float2 w2_val = w2_local_smem[smem_off / F2_TO_BF16_RATIO];

        acc2[n] += vec_fma_fp32_bf16(x_val, w2_val);
      }
    }
    __syncthreads();
    smem_stage ^= 1;
    if (slab_row + 2 * SMEM_ROWS < BN2) load_w2(slab_row + 2 * SMEM_ROWS);
    for (int i = 0; i < MAX_ACC_DOWN; i++) {
      acc2_bf16[i] = reduce_sum_bfly<LANES_K_DOWN>(acc2[i]);
    }
#pragma unroll
    for (int i = 0; i < MAX_ACC_DOWN / 8; i++) {
      if (lane_in_grp == i) {
        const int row = slab_row + gi * MAX_ACC_DOWN + i * 8;
        *reinterpret_cast<float4*>(result + row) = *reinterpret_cast<float4*>(&acc2_bf16[i * 8]);
      }
    }
  }

  cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
  __syncthreads();
  if (warp_id == 0 && elect_sync()) {
    cuda::ptx::cp_reduce_async_bulk(cuda::ptx::space_global, cuda::ptx::space_shared,
                                    cuda::ptx::op_add, out + token * K + blockIdx.z * (K / Z_SPLIT),
                                    result, (K / Z_SPLIT) * sizeof(__nv_bfloat16));

    cuda::ptx::cp_async_bulk_commit_group();
  }
}

template <int BN, int WARPS, int Z_SPLIT, moe_activation::Kind ACTIVATION>
bool launch_fused_moe_kernel_up_down_acc_gemv(
    const bf16* x, bf16* w, bf16* w2, bf16* out, const int* topk_ids, const float* topk_weights,
    const int top_k, int M, int N, int K, int num_experts, float scaling_factor,
    typename moe_activation::Traits<ACTIVATION>::Params activation_params, cudaStream_t stream) {
  constexpr int UP_FACTOR = moe_activation::Traits<ACTIVATION>::GATED ? 2 : 1;
  constexpr int K2_PER_BLOCK = BN / UP_FACTOR;
  constexpr int LANES_K_DOWN = K2_PER_BLOCK / 4 < 32 ? K2_PER_BLOCK / 4 : 32;
  constexpr int N_SPLIT_LANES = 32 / LANES_K_DOWN;

  constexpr int MAX_ACC_DOWN_RAW = AD / N_SPLIT_LANES;
  constexpr int MAX_ACC_DOWN = MAX_ACC_DOWN_RAW < 8 ? 8 : MAX_ACC_DOWN_RAW;

  constexpr int SMEM_COLS = K2_PER_BLOCK;
  constexpr int SMEM_ROWS = MAX_ACC_DOWN * N_SPLIT_LANES * WARPS;
  constexpr int SMEM_SIZE = SMEM_ROWS * SMEM_COLS;
  constexpr int STAGES = 2;
  const int smem_size = sizeof(bf16) * (STAGES * SMEM_SIZE + K / Z_SPLIT);
  if (smem_size + sizeof(float4) * (K2_PER_BLOCK / 4) > SM90_MAX_SMEM) {
    fprintf(stderr,
            "fused_moe gemv: BN=%d WARPS=%d requires %d bytes of "
            "shared memory but SM90 max is %zu — skipping launch\n",
            BN, WARPS, smem_size, SM90_MAX_SMEM);
    return false;
  }
  if (K > 0 && K % 256 == 0 && BN <= N && N % BN == 0 && K % (Z_SPLIT * SMEM_ROWS) == 0) {
    dim3 dimBlock(32 * WARPS, 1, 1);
    dim3 dimGrid(std::ceil((float)N / (BN)), M * top_k, Z_SPLIT);
    auto kernel = fused_moe_bf16_wgmma_up_down_acc_gemv_kernel<BN, WARPS, Z_SPLIT, ACTIVATION>;
    gpuErrchk(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    kernel<<<dimGrid, dimBlock, smem_size, stream>>>(x, w, w2, out, topk_ids, topk_weights, top_k,
                                                     M, N, K, num_experts, scaling_factor,
                                                     activation_params);
    return true;
  }
  return false;
}
