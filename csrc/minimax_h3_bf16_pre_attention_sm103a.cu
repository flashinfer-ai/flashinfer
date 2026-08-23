#include <cuda.h>
#include <cuda_bf16.h>

#include <cstdint>

struct __align__(128) FlashInferTensorMap {
  uint64_t opaque[16];
};
static_assert(sizeof(FlashInferTensorMap) == 128, "tensor-map ABI size mismatch");
static_assert(alignof(FlashInferTensorMap) == 128, "tensor-map ABI alignment mismatch");
static_assert(sizeof(CUtensorMap) == 128, "CUDA tensor-map ABI size mismatch");
static_assert(alignof(CUtensorMap) == 128, "CUDA tensor-map ABI alignment mismatch");

#define MINIMAX_H3_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_ACC0_OFFSET 0
#define TMEM_ACC1_OFFSET 128
#define TMEM_ACC2_OFFSET 256
#define TMEM_ACC3_OFFSET 384
#define NUM_MAINLOOP_STAGES 2
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B0_OFF 33792
#define SMEM_SMEM_B0_STAGE_BYTES 16384
#define SMEM_SMEM_B0_STRIDE 16384
#define SMEM_SMEM_B1_OFF 66560
#define SMEM_SMEM_B1_STAGE_BYTES 16384
#define SMEM_SMEM_B1_STRIDE 16384
#define SMEM_SMEM_B2_OFF 99328
#define SMEM_SMEM_B2_STAGE_BYTES 16384
#define SMEM_SMEM_B2_STRIDE 16384
#define SMEM_SMEM_B3_OFF 132096
#define SMEM_SMEM_B3_STAGE_BYTES 16384
#define SMEM_SMEM_B3_STRIDE 16384
#define SMEM_SMEM_RSTD_OFF 164864
#define SMEM_SMEM_RSTD_STAGE_BYTES 512
#define SMEM_SMEM_RSTD_STRIDE 512
#define SMEM_SMEM_ADALN_INDEX_OFF 165376
#define SMEM_SMEM_ADALN_INDEX_STAGE_BYTES 512
#define SMEM_SMEM_ADALN_INDEX_STRIDE 512
#define SMEM_SMEM_NORM_PARTIALS_OFF 165888
#define SMEM_SMEM_NORM_PARTIALS_STAGE_BYTES 64
#define SMEM_SMEM_NORM_PARTIALS_STRIDE 64
#define SMEM_TOTAL 166016
#define THREADS 640
#define BLOCK_M 128
#define BLOCK_N 128
#define BLOCK_K 64
#define GROUP_N 4
#define N_GROUPS 42
#define NUM_K_ITERS 84
#define NUM_STAGES 2
#define PREP_WARPS 16
#define HIDDEN 5376
#define NUM_HEADS 56
#define QKV_KINDS 3
#define HEAD_DIM 128
#define ROPE_DIM 96
#define ROPE_HALF 48
#define A_STAGE_BYTES 16384
#define B_STAGE_BYTES 16384
#define B_GROUP_BYTES 65536

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
  uint32_t pred = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred %%px;\n\t"
      "elect.sync _|%%px, %1;\n\t"
      "@%%px mov.s32 %0, 1;\n\t"
      "}\n"
      : "+r"(pred)
      : "r"(0xFFFFFFFF));
  return pred;
}

__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(mbar_addr), "r"(count));
}

__device__ __forceinline__ uint32_t mbarrier_try_wait(int mbar_addr, int phase) {
  uint32_t token;
  asm volatile(
      "{\n\t"
      ".reg .pred P1;\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
      " P1, [%1], %2;\n\t"
      "selp.u32 %0, 1, 0, P1;\n\t"
      "}\n"
      : "=r"(token)
      : "r"(mbar_addr), "r"(phase)
      : "memory");
  return token;
}

__device__ __forceinline__ uint32_t mbarrier_try_wait_cluster(int mbar_addr, int phase) {
  uint32_t token;
  asm volatile(
      "{\n\t"
      ".reg .pred P1;\n\t"
      "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
      " P1, [%1], %2;\n\t"
      "selp.u32 %0, 1, 0, P1;\n\t"
      "}\n"
      : "=r"(token)
      : "r"(mbar_addr), "r"(phase)
      : "memory");
  return token;
}

__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
  uint32_t ticks = 0x989680;
  asm volatile(
      "{\n\t"
      ".reg .pred P1;\n\t"
      "LAB_WAIT:\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
      " P1, [%0], %1, %2;\n\t"
      "@P1 bra.uni DONE;\n\t"
      "bra.uni LAB_WAIT;\n\t"
      "DONE:\n\t"
      "}\n" ::"r"(mbar_addr),
      "r"(phase), "r"(ticks)
      : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
  uint32_t ticks = 0x989680;
  asm volatile(
      "{\n\t"
      ".reg .pred P1;\n\t"
      "LAB_WAIT_CLUSTER:\n\t"
      "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
      " P1, [%0], %1, %2;\n\t"
      "@P1 bra.uni DONE_CLUSTER;\n\t"
      "bra.uni LAB_WAIT_CLUSTER;\n\t"
      "DONE_CLUSTER:\n\t"
      "}\n" ::"r"(mbar_addr),
      "r"(phase), "r"(ticks)
      : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
  if (token == 0) {
    mbarrier_wait(mbar_addr, phase);
  }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster(int mbar_addr, int phase,
                                                            uint32_t token) {
  if (token == 0) {
    mbarrier_wait_cluster(mbar_addr, phase);
  }
}

__device__ __forceinline__ void tcgen05_mma_f16(int taddr, uint64_t a_desc, uint64_t b_desc,
                                                uint32_t i_desc, int enable_input_d) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
      "}\n" ::"r"(taddr),
      "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(enable_input_d));
}

__device__ __forceinline__ uint64_t desc_encode(uint64_t x) { return (x & 0x3FFFFULL) >> 4ULL; }

__device__ __forceinline__ void mma_ss_step(int a_lo, int b_lo, int taddr, uint32_t i_desc,
                                            int enable_d, uint32_t a_dhi, uint32_t b_dhi) {
  asm volatile(
      "{\n\t"
      ".reg .pred leader, p;\n\t"
      ".reg .b32 adhi, bdhi;\n\t"
      ".reg .b64 da, db;\n\t"
      "elect.sync _|leader, 0xFFFFFFFF;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "mov.b32 adhi, %5;\n\t"
      "mov.b32 bdhi, %6;\n\t"
      "mov.b64 da, {%0, adhi};\n\t"
      "mov.b64 db, {%1, bdhi};\n\t"
      "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %3, p;\n\t"
      "}\n" ::"r"(a_lo),
      "r"(b_lo), "r"(taddr), "r"(i_desc), "r"(enable_d), "r"(a_dhi), "r"(b_dhi));
}

__device__ __forceinline__ void elect_commit(int mbar_addr) {
  asm volatile(
      "{\n\t"
      ".reg .pred leader;\n\t"
      "elect.sync _|leader, 0xFFFFFFFF;\n\t"
      "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
      ".shared::cluster.b64 [%0];\n\t"
      "}\n" ::"r"(mbar_addr));
}

__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(mbar_addr) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(mbar_addr),
      "r"(bytes)
      : "memory");
}

__device__ __forceinline__ void fence_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
  const int SBO = 1024;
  return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
}

__device__ __forceinline__ void tma_3d_gmem2smem(int dst, const void* tmap_ptr, int x, int y, int z,
                                                 int mbar_addr) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global"
      ".mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(dst),
      "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr)
      : "memory");
}

__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
      ".shared::cluster.b64 [%0];" ::"r"(mbar_addr)
      : "memory");
}

__device__ __forceinline__ void tmem_ld_x8(float* dst, int tmem_addr) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x8.b32"
      " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]), "=f"(dst[4]), "=f"(dst[5]),
        "=f"(dst[6]), "=f"(dst[7])
      : "r"(tmem_addr));
}

__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
  uint32_t result;
  asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;" : "=r"(result) : "r"(val));
  return result;
}

extern "C" {

__global__
__launch_bounds__(640, 1) void kernel_minimax_h3_bf16_pre_attention_destination_major_005f_v1(
    __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ x_norm_weight,
    __nv_bfloat16* __restrict__ adaln_scale, __nv_bfloat16* __restrict__ adaln_shift,
    int* __restrict__ adaln_index, __nv_bfloat16* __restrict__ q_norm_weight,
    __nv_bfloat16* __restrict__ k_norm_weight, __nv_bfloat16* __restrict__ rope_cos_sin,
    __nv_bfloat16* __restrict__ out, const __grid_constant__ CUtensorMap qkv_weight, int M, int P,
    float eps) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  int smem;
  smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  // Kernel setup ops
  __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
  const int smem_a_addr = smem + 1024;
  __nv_bfloat16* smem_b0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
  const int smem_b0_addr = smem + 33792;
  __nv_bfloat16* smem_b1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
  const int smem_b1_addr = smem + 66560;
  __nv_bfloat16* smem_b2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
  const int smem_b2_addr = smem + 99328;
  __nv_bfloat16* smem_b3 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
  const int smem_b3_addr = smem + 132096;
  float* smem_rstd = reinterpret_cast<float*>(smem_raw + 164864);
  const int smem_rstd_addr = smem + 164864;
  int* smem_adaln_index = reinterpret_cast<int*>(smem_raw + 165376);
  const int smem_adaln_index_addr = smem + 165376;
  float* smem_norm_partials = reinterpret_cast<float*>(smem_raw + 165888);
  const int smem_norm_partials_addr = smem + 165888;

  // Mbarrier init (4 groups, 7 barriers)
  // Mbarriers at smem_raw[0..56)

  if (warp == 0) {
    uint32_t leader = elect_sync();
    if (leader) {
      // --- pipeline 'mainloop' ---
      // a_full: 2 barriers, init_count=1
      mbarrier_init(smem + 0, 1);
      mbarrier_init(smem + 8, 1);
      // b_full: 2 barriers, init_count=1
      mbarrier_init(smem + 16, 1);
      mbarrier_init(smem + 24, 1);
      // stage_empty: 2 barriers, init_count=4
      mbarrier_init(smem + 32, 4);
      mbarrier_init(smem + 40, 4);
      // projection_done: 1 barriers, init_count=1
      mbarrier_init(smem + 48, 1);
      asm volatile("fence.mbarrier_init.release.cluster;");
    }
  }

  __syncwarp();

  // TMEM alloc (512 columns, 512 used)
  volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 56);
  if (warp == 0) {
    int _tmem_hold = smem + 56;
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(_tmem_hold),
        "r"(512)
        : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }

  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;");

  const int mbar_base = smem;
#define a_full_addr (mbar_base + 0)
#define b_full_addr (mbar_base + 16)
#define stage_empty_addr (mbar_base + 32)
#define projection_done_addr (mbar_base + 48)
  const int taddr = tmem_addr_storage[0];

  // Kernel post-init ops
  const int tmem_acc0 = taddr;
  const int tmem_acc1 = taddr + 128;
  const int tmem_acc2 = taddr + 256;
  const int tmem_acc3 = taddr + 384;

  // ---- Register redistribution for WGs split across roles ----
  // Dec phase frees registers before any WG attempts inc.
  if (warp >= 16 && warp <= 19) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 32;");
  }

  // ---- Role: prepare ----
  if (warp <= 15) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 96;");
    {  // prepare_main
      int tile_m = bid / N_GROUPS;
      int group_n = bid % N_GROUPS;
      int off_m = tile_m * BLOCK_M;
      int off_n = group_n * GROUP_N * BLOCK_N;
#pragma unroll
      for (int row_group = 0; row_group < BLOCK_M / 4; row_group++) {
        int row_slot = warp / 4;
        int warp_in_row = warp % 4;
        int thread_in_row = warp_in_row * 32 + lane;
        int local_row = row_group * 4 + row_slot;
        int global_row = off_m + local_row;
        float sum_sq = 0.0f;
        if (global_row < M) {
          unsigned long long row_base = (unsigned long long)global_row * (unsigned long long)HIDDEN;
#pragma unroll
          for (int vec_iter = 0; vec_iter < 11; vec_iter++) {
            int vec_index = thread_in_row + vec_iter * 128;
            if (vec_index < HIDDEN / 4) {
              float _vec_load_0[4];
              {
                uint2 _vld_0;
                _vld_0 = *reinterpret_cast<const uint2*>(
                    x + (row_base + (unsigned long long)(vec_index * 4)) + 0);
                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
#pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                  asm volatile(
                      "{\n\t"
                      "shl.b32 %0, %2, 16;\n\t"
                      "and.b32 %1, %2, 0xffff0000;\n\t"
                      "}\n"
                      : "=f"((&_vec_load_0[0 + _pair * 2])[0]),
                        "=f"((&_vec_load_0[0 + _pair * 2])[1])
                      : "r"(_vpairs_0[_pair]));
                }
              }
#pragma unroll
              for (int j = 0; j < 4; j++) {
                sum_sq += _vec_load_0[j] * _vec_load_0[j];
              }
            }
          }
        }
        float _shfl_down_0 = __shfl_down_sync(0xFFFFFFFF, sum_sq, 16, 32);
        float sh16 = _shfl_down_0;
        sum_sq += sh16;
        float _shfl_down_1 = __shfl_down_sync(0xFFFFFFFF, sum_sq, 8, 32);
        float sh8 = _shfl_down_1;
        sum_sq += sh8;
        float _shfl_down_2 = __shfl_down_sync(0xFFFFFFFF, sum_sq, 4, 32);
        float sh4 = _shfl_down_2;
        sum_sq += sh4;
        float _shfl_down_3 = __shfl_down_sync(0xFFFFFFFF, sum_sq, 2, 32);
        float sh2 = _shfl_down_3;
        sum_sq += sh2;
        float _shfl_down_4 = __shfl_down_sync(0xFFFFFFFF, sum_sq, 1, 32);
        float sh1 = _shfl_down_4;
        sum_sq += sh1;
        if (lane == 0) {
          if (warp_in_row >= 2) {
            smem_norm_partials[row_slot * 2 + warp_in_row - 2] = sum_sq;
          }
        }
        asm volatile("barrier.sync 1, 512;" ::: "memory");
        if (lane == 0) {
          if (warp_in_row < 2) {
            sum_sq = sum_sq + smem_norm_partials[row_slot * 2 + warp_in_row];
          }
        }
        asm volatile("barrier.sync 1, 512;" ::: "memory");
        if (lane == 0) {
          if (warp_in_row == 1) {
            smem_norm_partials[row_slot] = sum_sq;
          }
        }
        asm volatile("barrier.sync 1, 512;" ::: "memory");
        if (lane == 0) {
          if (warp_in_row == 0) {
            sum_sq = sum_sq + smem_norm_partials[row_slot];
            if (global_row < M) {
              float _rsqrt_0 = rsqrtf(sum_sq / (float)HIDDEN + eps);
              smem_rstd[local_row] = _rsqrt_0;
              smem_adaln_index[local_row] = adaln_index[global_row];
            } else {
              smem_rstd[local_row] = 0.0f;
              smem_adaln_index[local_row] = 0;
            }
          }
        }
        asm volatile("barrier.sync 1, 512;" ::: "memory");
      }
      unsigned int load_stage = 0;
      unsigned int _phase_stage_empty = 1;
#pragma unroll 1
      for (int iter_k = 0; iter_k < NUM_K_ITERS; iter_k++) {
        mbarrier_wait(stage_empty_addr + (load_stage) * 8, _phase_stage_empty);
        int off_k = iter_k * BLOCK_K;
        if (warp == 0) {
          if (elect_sync()) {
            tma_3d_gmem2smem(smem_b0_addr + load_stage * 16384, (&qkv_weight), 0, off_n, off_k / 64,
                             b_full_addr + (load_stage) * 8);
            tma_3d_gmem2smem(smem_b1_addr + load_stage * 16384, (&qkv_weight), 0, off_n + BLOCK_N,
                             off_k / 64, b_full_addr + (load_stage) * 8);
            tma_3d_gmem2smem(smem_b2_addr + load_stage * 16384, (&qkv_weight), 0,
                             off_n + 2 * BLOCK_N, off_k / 64, b_full_addr + (load_stage) * 8);
            tma_3d_gmem2smem(smem_b3_addr + load_stage * 16384, (&qkv_weight), 0,
                             off_n + 3 * BLOCK_N, off_k / 64, b_full_addr + (load_stage) * 8);
            mbarrier_arrive_expect_tx(b_full_addr + (load_stage) * 8, B_GROUP_BYTES);
          }
        }
        int thread_local_k = tid % (BLOCK_K / 8) * 8;
        float _vec_load_1[8];
        {
          const uint4* _vptr_1 =
              reinterpret_cast<const uint4*>(x_norm_weight + (off_k + thread_local_k) + 0);
          uint4 _vld_1[1];
#pragma unroll
          for (int _blk = 0; _blk < 1; _blk++) {
            _vld_1[_blk] = _vptr_1[_blk];
            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
#pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
              asm volatile(
                  "{\n\t"
                  "shl.b32 %0, %2, 16;\n\t"
                  "and.b32 %1, %2, 0xffff0000;\n\t"
                  "}\n"
                  : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]),
                    "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                  : "r"(_vpairs_1[_pair]));
            }
          }
        }
#pragma unroll
        for (int row_iter = 0; row_iter < 2; row_iter++) {
          int linear_vec = tid + row_iter * (PREP_WARPS * 32);
          int local_row_1 = linear_vec / (BLOCK_K / 8);
          int local_k = linear_vec % (BLOCK_K / 8) * 8;
          int global_row_1 = off_m + local_row_1;
          int k = off_k + local_k;
          float adaln_values[8];
#pragma unroll
          for (int j_1 = 0; j_1 < 8; j_1++) {
            adaln_values[j_1] = 0.0f;
          }
          if (global_row_1 < M) {
            int table_row = smem_adaln_index[local_row_1];
            if (table_row >= 0 && table_row < 9) {
              unsigned long long row_base_1 =
                  (unsigned long long)global_row_1 * (unsigned long long)HIDDEN;
              unsigned long long table_offset =
                  (unsigned long long)table_row * (unsigned long long)HIDDEN +
                  (unsigned long long)k;
              float _vec_load_2[8];
              {
                const uint4* _vptr_2 =
                    reinterpret_cast<const uint4*>(x + (row_base_1 + (unsigned long long)k) + 0);
                uint4 _vld_2[1];
#pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                  _vld_2[_blk] = _vptr_2[_blk];
                  uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
#pragma unroll
                  for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[0]),
                          "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_2[_pair]));
                  }
                }
              }
              float _vec_load_3[8];
              {
                const uint4* _vptr_3 =
                    reinterpret_cast<const uint4*>(adaln_scale + table_offset + 0);
                uint4 _vld_3[1];
#pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                  _vld_3[_blk] = _vptr_3[_blk];
                  uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
#pragma unroll
                  for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[0]),
                          "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_3[_pair]));
                  }
                }
              }
              float _vec_load_4[8];
              {
                const uint4* _vptr_4 =
                    reinterpret_cast<const uint4*>(adaln_shift + table_offset + 0);
                uint4 _vld_4[1];
#pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                  _vld_4[_blk] = _vptr_4[_blk];
                  uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
#pragma unroll
                  for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_4[0 + _blk * 8 + _pair * 2])[0]),
                          "=f"((&_vec_load_4[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_4[_pair]));
                  }
                }
              }
#pragma unroll
              for (int j_2 = 0; j_2 < 8; j_2++) {
                float x_value = _vec_load_2[j_2];
                float norm_weight = _vec_load_1[j_2];
                float scaled_input = smem_rstd[local_row_1] * x_value;
                __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(norm_weight * scaled_input);
                float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
                float norm_value = _cvt_f32_0;
                float scale_value = _vec_load_3[j_2];
                float shift_value = _vec_load_4[j_2];
                __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(scale_value + 1.0f);
                float _cvt_f32_1 = __bfloat162float(_cvt_bf16_1);
                float scale_plus_one = _cvt_f32_1;
                float _fma_0 = __fmaf_rn(norm_value, scale_plus_one, shift_value);
                __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(_fma_0);
                float _cvt_f32_2 = __bfloat162float(_cvt_bf16_2);
                adaln_values[j_2] = _cvt_f32_2;
              }
            }
          }
          unsigned int packed_values[4];
#pragma unroll
          for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                make_float2(adaln_values[_lp * 2 + 0], adaln_values[_lp * 2 + 1 + 0]));
            packed_values[_lp] = *(uint32_t*)&_bf2;
          }
          int a_vec_store_addr =
              (smem_a_addr + load_stage * 16384 +
               (unsigned int)(local_k / 64 * 16384 + local_row_1 * 128 + local_k % 64 * 2 ^
                              (local_k / 64 * 16384 + local_row_1 * 128 + local_k % 64 * 2 >> 7 & 7)
                                  << 4));
          asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(a_vec_store_addr),
                       "r"(packed_values[0]), "r"(packed_values[1]), "r"(packed_values[2]),
                       "r"(packed_values[3])
                       : "memory");
        }
        asm volatile("barrier.sync 1, 512;" ::: "memory");
        if (warp == 0) {
          if (elect_sync()) {
            asm volatile("fence.proxy.async;");
            mbarrier_arrive(a_full_addr + (load_stage) * 8);
          }
        }
        load_stage += 1;
        if (load_stage == 2) {
          load_stage = 0;
          _phase_stage_empty ^= 1;
        }
      }
      unsigned int _phase_projection_done_0 = 0;
      mbarrier_wait(projection_done_addr, _phase_projection_done_0);
      _phase_projection_done_0 ^= 1;
      asm volatile("tcgen05.fence::after_thread_sync;");
      int warp_id_in_role = (warp - 0);
      int prepare_warp_id = warp_id_in_role;
      int tile_m_0 = bid / N_GROUPS;
      int group_n_1 = bid % N_GROUPS;
      int off_m_2 = tile_m_0 * BLOCK_M;
      int off_n_3 = group_n_1 * GROUP_N * BLOCK_N;
      int warp_in_wg = warp % 4;
      int row_addr = warp_in_wg * 32 << 16;
      int local_row_2 = warp_in_wg * 32 + lane;
      int global_row_2 = off_m_2 + local_row_2;
      int safe_global_row = ((global_row_2 < M) ? global_row_2 : M - 1);
      int heads_per_destination = NUM_HEADS / P;
#pragma unroll 1
      for (int subgroup = prepare_warp_id / 4; subgroup < prepare_warp_id / 4 + 1; subgroup++) {
        int native_group = group_n_1 * GROUP_N + subgroup;
        int head = native_group / QKV_KINDS;
        int kind = native_group % QKV_KINDS;
        int destination = head / heads_per_destination;
        int local_head = head % heads_per_destination;
        unsigned long long out_base = ((((unsigned long long)destination * (unsigned long long)M +
                                         (unsigned long long)global_row_2) *
                                            (unsigned long long)heads_per_destination +
                                        (unsigned long long)local_head) *
                                           (unsigned long long)QKV_KINDS +
                                       (unsigned long long)kind) *
                                      (unsigned long long)HEAD_DIM;
        int tmem_base = taddr + (unsigned int)row_addr + (unsigned int)(subgroup * BLOCK_N);
        if (kind < 2) {
          float sum_partials[32];
#pragma unroll
          for (int chunk = 0; chunk < HEAD_DIM / 8; chunk++) {
            float _tmem_load_0[8];
            tmem_ld_x8(&_tmem_load_0[0], tmem_base + chunk * 8);
            float chunk_sum_lo = 0.0f;
            float chunk_sum_hi = 0.0f;
#pragma unroll
            for (int j_3 = 0; j_3 < 4; j_3++) {
              __nv_bfloat16 _cvt_bf16_3 = __float2bfloat16(_tmem_load_0[j_3]);
              float _cvt_f32_3 = __bfloat162float(_cvt_bf16_3);
              float rounded = _cvt_f32_3;
              chunk_sum_lo += rounded * rounded;
              __nv_bfloat16 _cvt_bf16_4 = __float2bfloat16(_tmem_load_0[j_3 + 4]);
              float _cvt_f32_4 = __bfloat162float(_cvt_bf16_4);
              float rounded_hi = _cvt_f32_4;
              chunk_sum_hi += rounded_hi * rounded_hi;
            }
            sum_partials[chunk * 2] = chunk_sum_lo;
            sum_partials[chunk * 2 + 1] = chunk_sum_hi;
          }
#pragma unroll
          for (int i = 0; i < 16; i++) {
            sum_partials[i] = sum_partials[i] + sum_partials[i + 16];
          }
#pragma unroll
          for (int i_1 = 0; i_1 < 8; i_1++) {
            sum_partials[i_1] = sum_partials[i_1] + sum_partials[i_1 + 8];
          }
#pragma unroll
          for (int i_2 = 0; i_2 < 4; i_2++) {
            sum_partials[i_2] = sum_partials[i_2] + sum_partials[i_2 + 4];
          }
#pragma unroll
          for (int i_3 = 0; i_3 < 2; i_3++) {
            sum_partials[i_3] = sum_partials[i_3] + sum_partials[i_3 + 2];
          }
          float sum_sq_1 = sum_partials[0] + sum_partials[1];
          float _rsqrt_1 = rsqrtf(sum_sq_1 / (float)HEAD_DIM + eps);
          float rstd = _rsqrt_1;
#pragma unroll
          for (int chunk_1 = 0; chunk_1 < ROPE_HALF / 8; chunk_1++) {
            int col_lo = chunk_1 * 8;
            int col_hi = col_lo + ROPE_HALF;
            float _tmem_load_1[8];
            tmem_ld_x8(&_tmem_load_1[0], tmem_base + col_lo);
            float _tmem_load_2[8];
            tmem_ld_x8(&_tmem_load_2[0], tmem_base + col_hi);
            float _vec_load_5[8];
            {
              const uint4* _vptr_5 = reinterpret_cast<const uint4*>(
                  ((kind == 0) ? q_norm_weight + col_lo : k_norm_weight + col_lo) + 0);
              uint4 _vld_5[1];
#pragma unroll
              for (int _blk = 0; _blk < 1; _blk++) {
                _vld_5[_blk] = _vptr_5[_blk];
                uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
#pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                  asm volatile(
                      "{\n\t"
                      "shl.b32 %0, %2, 16;\n\t"
                      "and.b32 %1, %2, 0xffff0000;\n\t"
                      "}\n"
                      : "=f"((&_vec_load_5[0 + _blk * 8 + _pair * 2])[0]),
                        "=f"((&_vec_load_5[0 + _blk * 8 + _pair * 2])[1])
                      : "r"(_vpairs_5[_pair]));
                }
              }
            }
            float _vec_load_6[8];
            {
              const uint4* _vptr_6 = reinterpret_cast<const uint4*>(
                  ((kind == 0) ? q_norm_weight + col_hi : k_norm_weight + col_hi) + 0);
              uint4 _vld_6[1];
#pragma unroll
              for (int _blk = 0; _blk < 1; _blk++) {
                _vld_6[_blk] = _vptr_6[_blk];
                uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
#pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                  asm volatile(
                      "{\n\t"
                      "shl.b32 %0, %2, 16;\n\t"
                      "and.b32 %1, %2, 0xffff0000;\n\t"
                      "}\n"
                      : "=f"((&_vec_load_6[0 + _blk * 8 + _pair * 2])[0]),
                        "=f"((&_vec_load_6[0 + _blk * 8 + _pair * 2])[1])
                      : "r"(_vpairs_6[_pair]));
                }
              }
            }
            float _vec_load_7[8];
            {
              const uint4* _vptr_7 = reinterpret_cast<const uint4*>(
                  rope_cos_sin +
                  ((unsigned long long)safe_global_row * (unsigned long long)ROPE_DIM +
                   (unsigned long long)col_lo) +
                  0);
              uint4 _vld_7[1];
#pragma unroll
              for (int _blk = 0; _blk < 1; _blk++) {
                _vld_7[_blk] = _vptr_7[_blk];
                uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
#pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                  asm volatile(
                      "{\n\t"
                      "shl.b32 %0, %2, 16;\n\t"
                      "and.b32 %1, %2, 0xffff0000;\n\t"
                      "}\n"
                      : "=f"((&_vec_load_7[0 + _blk * 8 + _pair * 2])[0]),
                        "=f"((&_vec_load_7[0 + _blk * 8 + _pair * 2])[1])
                      : "r"(_vpairs_7[_pair]));
                }
              }
            }
            float _vec_load_8[8];
            {
              const uint4* _vptr_8 = reinterpret_cast<const uint4*>(
                  rope_cos_sin +
                  ((unsigned long long)safe_global_row * (unsigned long long)ROPE_DIM +
                   (unsigned long long)ROPE_HALF + (unsigned long long)col_lo) +
                  0);
              uint4 _vld_8[1];
#pragma unroll
              for (int _blk = 0; _blk < 1; _blk++) {
                _vld_8[_blk] = _vptr_8[_blk];
                uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
#pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                  asm volatile(
                      "{\n\t"
                      "shl.b32 %0, %2, 16;\n\t"
                      "and.b32 %1, %2, 0xffff0000;\n\t"
                      "}\n"
                      : "=f"((&_vec_load_8[0 + _blk * 8 + _pair * 2])[0]),
                        "=f"((&_vec_load_8[0 + _blk * 8 + _pair * 2])[1])
                      : "r"(_vpairs_8[_pair]));
                }
              }
            }
            float rotated_lo[8];
            float rotated_hi[8];
#pragma unroll
            for (int j_4 = 0; j_4 < 8; j_4++) {
              __nv_bfloat16 _cvt_bf16_5 = __float2bfloat16(_tmem_load_1[j_4]);
              float _cvt_f32_5 = __bfloat162float(_cvt_bf16_5);
              float rounded_lo = _cvt_f32_5;
              __nv_bfloat16 _cvt_bf16_6 = __float2bfloat16(_tmem_load_2[j_4]);
              float _cvt_f32_6 = __bfloat162float(_cvt_bf16_6);
              float rounded_hi_1 = _cvt_f32_6;
              float scaled_lo = rstd * rounded_lo;
              float scaled_hi = rstd * rounded_hi_1;
              __nv_bfloat16 _cvt_bf16_7 = __float2bfloat16(_vec_load_5[j_4] * scaled_lo);
              float _cvt_f32_7 = __bfloat162float(_cvt_bf16_7);
              float norm_lo = _cvt_f32_7;
              __nv_bfloat16 _cvt_bf16_8 = __float2bfloat16(_vec_load_6[j_4] * scaled_hi);
              float _cvt_f32_8 = __bfloat162float(_cvt_bf16_8);
              float norm_hi = _cvt_f32_8;
              float cos_lo = norm_lo * _vec_load_7[j_4];
              float cos_hi = norm_hi * _vec_load_7[j_4];
              float sin_lo = norm_lo * _vec_load_8[j_4];
              float sin_hi = norm_hi * _vec_load_8[j_4];
              rotated_lo[j_4] = cos_lo - sin_hi;
              rotated_hi[j_4] = cos_hi + sin_lo;
            }
            if (global_row_2 < M) {
              {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(rotated_lo[0 + 0], rotated_lo[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(rotated_lo[0 + 2], rotated_lo[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(rotated_lo[0 + 4], rotated_lo[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(rotated_lo[0 + 6], rotated_lo[0 + 7]);
                *reinterpret_cast<uint4*>(
                    &((__nv_bfloat16*)(out + (out_base + (unsigned long long)col_lo)))[0]) =
                    *reinterpret_cast<uint4*>(&_pk[0]);
              }
              {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(rotated_hi[0 + 0], rotated_hi[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(rotated_hi[0 + 2], rotated_hi[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(rotated_hi[0 + 4], rotated_hi[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(rotated_hi[0 + 6], rotated_hi[0 + 7]);
                *reinterpret_cast<uint4*>(
                    &((__nv_bfloat16*)(out + (out_base + (unsigned long long)col_hi)))[0]) =
                    *reinterpret_cast<uint4*>(&_pk[0]);
              }
            }
          }
#pragma unroll
          for (int chunk_2 = ROPE_DIM / 8; chunk_2 < HEAD_DIM / 8; chunk_2++) {
            int col = chunk_2 * 8;
            float _tmem_load_3[8];
            tmem_ld_x8(&_tmem_load_3[0], tmem_base + col);
            float _vec_load_9[8];
            {
              const uint4* _vptr_9 = reinterpret_cast<const uint4*>(
                  ((kind == 0) ? q_norm_weight + col : k_norm_weight + col) + 0);
              uint4 _vld_9[1];
#pragma unroll
              for (int _blk = 0; _blk < 1; _blk++) {
                _vld_9[_blk] = _vptr_9[_blk];
                uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9[_blk]);
#pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                  asm volatile(
                      "{\n\t"
                      "shl.b32 %0, %2, 16;\n\t"
                      "and.b32 %1, %2, 0xffff0000;\n\t"
                      "}\n"
                      : "=f"((&_vec_load_9[0 + _blk * 8 + _pair * 2])[0]),
                        "=f"((&_vec_load_9[0 + _blk * 8 + _pair * 2])[1])
                      : "r"(_vpairs_9[_pair]));
                }
              }
            }
            float normalized[8];
#pragma unroll
            for (int j_5 = 0; j_5 < 8; j_5++) {
              __nv_bfloat16 _cvt_bf16_9 = __float2bfloat16(_tmem_load_3[j_5]);
              float _cvt_f32_9 = __bfloat162float(_cvt_bf16_9);
              float rounded_1 = _cvt_f32_9;
              float scaled = rstd * rounded_1;
              normalized[j_5] = _vec_load_9[j_5] * scaled;
            }
            if (global_row_2 < M) {
              {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(normalized[0 + 0], normalized[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(normalized[0 + 2], normalized[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(normalized[0 + 4], normalized[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(normalized[0 + 6], normalized[0 + 7]);
                *reinterpret_cast<uint4*>(
                    &((__nv_bfloat16*)(out + (out_base + (unsigned long long)col)))[0]) =
                    *reinterpret_cast<uint4*>(&_pk[0]);
              }
            }
          }
        } else {
#pragma unroll
          for (int chunk_3 = 0; chunk_3 < HEAD_DIM / 8; chunk_3++) {
            int col_1 = chunk_3 * 8;
            float _tmem_load_4[8];
            tmem_ld_x8(&_tmem_load_4[0], tmem_base + col_1);
            if (global_row_2 < M) {
              {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(_tmem_load_4[0 + 0], _tmem_load_4[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(_tmem_load_4[0 + 2], _tmem_load_4[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(_tmem_load_4[0 + 4], _tmem_load_4[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(_tmem_load_4[0 + 6], _tmem_load_4[0 + 7]);
                *reinterpret_cast<uint4*>(
                    &((__nv_bfloat16*)(out + (out_base + (unsigned long long)col_1)))[0]) =
                    *reinterpret_cast<uint4*>(&_pk[0]);
              }
            }
          }
        }
      }
      asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
      asm volatile("tcgen05.fence::before_thread_sync;");
    }
  }
  // ---- Role: mma ----
  if (warp == 16) {
    {  // mma_main
      int tile_m_1 = bid / N_GROUPS;
      int group_n_2 = bid % N_GROUPS;
      int off_m_1 = tile_m_1 * BLOCK_M;
      int off_n_1 = group_n_2 * GROUP_N * BLOCK_N;
      unsigned int mma_stage = 0;
      unsigned int _phase_a_full = 0;
      unsigned int _phase_b_full = 0;
#pragma unroll 1
      for (int iter_k_1 = 0; iter_k_1 < NUM_K_ITERS; iter_k_1++) {
        mbarrier_wait(a_full_addr + (mma_stage) * 8, _phase_a_full);
        mbarrier_wait(b_full_addr + (mma_stage) * 8, _phase_b_full);
        int init_flag = ((iter_k_1 == 0) ? 1 : 0);
        int _mma_a_lo_0 = make_warp_uniform((((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 1024);
        int _mma_b_lo_0 = make_warp_uniform((((smem_b0_addr) >> 4) & 0x3FFF) + (mma_stage) * 1024);
        asm volatile(
            "{\n\t"
            ".reg .pred leader, p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
            ".reg .b64 da, db;\n\t"
            "elect.sync _|leader, 0xFFFFFFFF;\n\t"
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            ""
            "mov.b32 adhi, 0x40004040;\n\t"
            "mov.b32 bdhi, 0x40004040;\n\t"
            "mov.b32 id, 136316048;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "}\n" ::"r"(_mma_a_lo_0),
            "r"(_mma_b_lo_0), "r"(tmem_acc0), "r"(((init_flag) ? 0 : 1)));
        elect_commit(stage_empty_addr + (mma_stage) * 8);
        int _mma_a_lo_1 = make_warp_uniform((((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 1024);
        int _mma_b_lo_1 = make_warp_uniform((((smem_b1_addr) >> 4) & 0x3FFF) + (mma_stage) * 1024);
        asm volatile(
            "{\n\t"
            ".reg .pred leader, p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
            ".reg .b64 da, db;\n\t"
            "elect.sync _|leader, 0xFFFFFFFF;\n\t"
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            ""
            "mov.b32 adhi, 0x40004040;\n\t"
            "mov.b32 bdhi, 0x40004040;\n\t"
            "mov.b32 id, 136316048;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "}\n" ::"r"(_mma_a_lo_1),
            "r"(_mma_b_lo_1), "r"(tmem_acc1), "r"(((init_flag) ? 0 : 1)));
        elect_commit(stage_empty_addr + (mma_stage) * 8);
        int _mma_a_lo_2 = make_warp_uniform((((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 1024);
        int _mma_b_lo_2 = make_warp_uniform((((smem_b2_addr) >> 4) & 0x3FFF) + (mma_stage) * 1024);
        asm volatile(
            "{\n\t"
            ".reg .pred leader, p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
            ".reg .b64 da, db;\n\t"
            "elect.sync _|leader, 0xFFFFFFFF;\n\t"
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            ""
            "mov.b32 adhi, 0x40004040;\n\t"
            "mov.b32 bdhi, 0x40004040;\n\t"
            "mov.b32 id, 136316048;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "}\n" ::"r"(_mma_a_lo_2),
            "r"(_mma_b_lo_2), "r"(tmem_acc2), "r"(((init_flag) ? 0 : 1)));
        elect_commit(stage_empty_addr + (mma_stage) * 8);
        int _mma_a_lo_3 = make_warp_uniform((((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 1024);
        int _mma_b_lo_3 = make_warp_uniform((((smem_b3_addr) >> 4) & 0x3FFF) + (mma_stage) * 1024);
        asm volatile(
            "{\n\t"
            ".reg .pred leader, p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
            ".reg .b64 da, db;\n\t"
            "elect.sync _|leader, 0xFFFFFFFF;\n\t"
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            ""
            "mov.b32 adhi, 0x40004040;\n\t"
            "mov.b32 bdhi, 0x40004040;\n\t"
            "mov.b32 id, 136316048;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "}\n" ::"r"(_mma_a_lo_3),
            "r"(_mma_b_lo_3), "r"(tmem_acc3), "r"(((init_flag) ? 0 : 1)));
        if (iter_k_1 == NUM_K_ITERS - 1) {
          elect_commit(projection_done_addr);
        } else {
          elect_commit(stage_empty_addr + (mma_stage) * 8);
        }
        mma_stage += 1;
        if (mma_stage == 2) {
          mma_stage = 0;
          _phase_a_full ^= 1;
          _phase_b_full ^= 1;
        }
      }
    }
  }
  // ---- Role: empty ----
  if (warp >= 17 && warp <= 19) {
    // idle — no tasks assigned
  }

  // Cleanup
  __syncthreads();  // barrier before TMEM dealloc

  if (warp == 0) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem_addr_storage[0]),
        "r"(512));
  }
}

}  // extern "C"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <mutex>
#include <vector>

#include "tvm_ffi_utils.h"

namespace {

constexpr int64_t kHidden = 5376;
constexpr int64_t kNumHeads = 56;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kQkvKinds = 3;
constexpr int64_t kQkvWidth = kNumHeads * kQkvKinds * kHeadDim;
constexpr int64_t kRopeDim = 96;
constexpr int64_t kAdalnRows = 9;
constexpr int kBlockM = 128;
constexpr int kNumColumnGroups = 42;
constexpr int kThreads = 640;
constexpr int kDynamicSmemBytes = 166016;
constexpr double kEps = 1.0e-5;

void CheckContiguousTensor(const TensorView& tensor, const char* name,
                           std::initializer_list<int64_t> shape, DLDataType dtype,
                           DLDevice device) {
  TVM_FFI_CHECK(tensor.device().device_type == kDLCUDA, ValueError)
      << name << " must be a CUDA tensor";
  TVM_FFI_CHECK(tensor.device().device_id == device.device_id, ValueError)
      << name << " must be on the same CUDA device as x";
  TVM_FFI_CHECK(encode_dlpack_dtype(tensor.dtype()) == encode_dlpack_dtype(dtype), ValueError)
      << name << " has an unsupported dtype";
  TVM_FFI_CHECK(tensor.ndim() == static_cast<int64_t>(shape.size()), ValueError)
      << name << " has an invalid rank";

  int64_t expected_stride = 1;
  int64_t dim = tensor.ndim();
  for (auto it = shape.end(); it != shape.begin();) {
    --it;
    --dim;
    TVM_FFI_CHECK(tensor.size(dim) == *it, ValueError) << name << " has an invalid shape";
    TVM_FFI_CHECK(tensor.stride(dim) == expected_stride, ValueError)
        << name << " must be contiguous";
    expected_stride *= *it;
  }
}

CUtensorMap EncodeQkvWeight(const TensorView& qkv_weight) {
  uint64_t global_dim[3] = {64, static_cast<uint64_t>(kQkvWidth),
                            static_cast<uint64_t>(kHidden / 64)};
  uint64_t global_strides[2] = {static_cast<uint64_t>(kHidden * 2), 128};
  uint32_t box_dim[3] = {64, 128, 1};
  uint32_t element_strides[3] = {1, 1, 1};
  CUtensorMap descriptor{};
  CUresult result = cuTensorMapEncodeTiled(
      &descriptor, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, qkv_weight.data_ptr(), global_dim,
      global_strides, box_dim, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "failed to encode qkv_weight tensor map: CUresult=" << static_cast<int>(result);
  return descriptor;
}

void ConfigureKernel() {
  static std::mutex mutex;
  static std::vector<int> configured_devices;
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "failed to get the active CUDA device: " << cudaGetErrorString(status);

  std::lock_guard<std::mutex> lock(mutex);
  for (int configured_device : configured_devices) {
    if (configured_device == device) {
      return;
    }
  }
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, device);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "failed to query CUDA device properties: " << cudaGetErrorString(status);
  TVM_FFI_CHECK(properties.major == 10 && properties.minor == 3, RuntimeError)
      << "MiniMax-H3 BF16 pre-attention requires compute capability 10.3";
  status = cudaFuncSetAttribute(kernel_minimax_h3_bf16_pre_attention_destination_major_005f_v1,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSmemBytes);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "failed to opt in to dynamic shared memory: " << cudaGetErrorString(status);
  configured_devices.push_back(device);
}

}  // namespace

void minimax_h3_bf16_pre_attention(TensorView x, TensorView x_norm_weight, TensorView adaln_scale,
                                   TensorView adaln_shift, TensorView adaln_index,
                                   TensorView qkv_weight, TensorView q_norm_weight,
                                   TensorView k_norm_weight, TensorView rope_cos_sin,
                                   TensorView out, int64_t m, int64_t ulysses_degree, double eps) {
  TVM_FFI_CHECK(m > 0 && m <= std::numeric_limits<int>::max(), ValueError)
      << "M must be a positive int32 value";
  TVM_FFI_CHECK(m == x.size(0), ValueError) << "M must equal x.size(0)";
  TVM_FFI_CHECK(
      ulysses_degree == 1 || ulysses_degree == 2 || ulysses_degree == 4 || ulysses_degree == 8,
      ValueError)
      << "ulysses_degree must be one of 1, 2, 4, or 8";
  TVM_FFI_CHECK(eps == kEps, ValueError) << "eps must be 1e-5";

  const DLDevice device = x.device();
  CheckContiguousTensor(x, "x", {m, kHidden}, dl_bfloat16, device);
  CheckContiguousTensor(x_norm_weight, "x_norm_weight", {kHidden}, dl_bfloat16, device);
  CheckContiguousTensor(adaln_scale, "adaln_scale", {kAdalnRows, kHidden}, dl_bfloat16, device);
  CheckContiguousTensor(adaln_shift, "adaln_shift", {kAdalnRows, kHidden}, dl_bfloat16, device);
  CheckContiguousTensor(adaln_index, "adaln_index", {m}, dl_int32, device);
  CheckContiguousTensor(qkv_weight, "qkv_weight", {kQkvWidth, kHidden}, dl_bfloat16, device);
  CheckContiguousTensor(q_norm_weight, "q_norm_weight", {kHeadDim}, dl_bfloat16, device);
  CheckContiguousTensor(k_norm_weight, "k_norm_weight", {kHeadDim}, dl_bfloat16, device);
  CheckContiguousTensor(rope_cos_sin, "rope_cos_sin", {m, kRopeDim}, dl_bfloat16, device);
  CheckContiguousTensor(out, "out",
                        {ulysses_degree, m, kNumHeads / ulysses_degree, kQkvKinds, kHeadDim},
                        dl_bfloat16, device);

  ffi::CUDADeviceGuard device_guard(device.device_id);
  const cudaStream_t stream = get_stream(device);
  ConfigureKernel();
  const CUtensorMap qkv_descriptor = EncodeQkvWeight(qkv_weight);
  const int grid = static_cast<int>((m + kBlockM - 1) / kBlockM) * kNumColumnGroups;
  kernel_minimax_h3_bf16_pre_attention_destination_major_005f_v1<<<grid, kThreads,
                                                                   kDynamicSmemBytes, stream>>>(
      static_cast<__nv_bfloat16*>(x.data_ptr()),
      static_cast<__nv_bfloat16*>(x_norm_weight.data_ptr()),
      static_cast<__nv_bfloat16*>(adaln_scale.data_ptr()),
      static_cast<__nv_bfloat16*>(adaln_shift.data_ptr()),
      static_cast<int32_t*>(adaln_index.data_ptr()),
      static_cast<__nv_bfloat16*>(q_norm_weight.data_ptr()),
      static_cast<__nv_bfloat16*>(k_norm_weight.data_ptr()),
      static_cast<__nv_bfloat16*>(rope_cos_sin.data_ptr()),
      static_cast<__nv_bfloat16*>(out.data_ptr()), qkv_descriptor, static_cast<int>(m),
      static_cast<int>(ulysses_degree), static_cast<float>(eps));
  const cudaError_t status = cudaGetLastError();
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "MiniMax-H3 BF16 pre-attention launch failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(minimax_h3_bf16_pre_attention, minimax_h3_bf16_pre_attention);
