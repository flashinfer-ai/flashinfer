/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

// Common preamble (typedefs, tensor-map ABI, compiler helpers) shared by this export's kernels.
#include "cake_mla_nvfp4_paged_decode_device_common.cuh"

// Hardware QMUL4 for the NVFP4 latent cache: E2M1 x E4M3 -> E4M3 with one rounding (round to nearest even,
// satfinite) through the PTX ISA 9.4 packed multiply `mul.rn.e4m3x4.e2m1x4.e4m3x4.satfinite` (CUDA 13.4 or newer;
// sm_100a / sm_103a).  kVariant 5 multiplies the low four E2M1 nibbles of `src`, 6 the high four; `scale` carries
// the E4M3 scale byte in all four lanes (the kernel broadcasts it with one prmt; the native QMUL4 reads lane 0).
#if !(__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
#error "the Cake NVFP4 MLA decode kernels require CUDA 13.4 or newer (PTX ISA 9.4 mul.e4m3x4.e2m1x4)"
#endif
template <int kVariant>
__device__ __forceinline__ uint32_t cake_mla_nvfp4_qmul4(uint32_t src, uint32_t scale) {
  static_assert(kVariant == 5 || kVariant == 6, "invalid Cake NVFP4 MLA QMUL4 variant (LOWER4 / HIGHER4 only)");
  const uint16_t a = static_cast<uint16_t>(kVariant == 5 ? src : (src >> 16));
  uint32_t d;
  asm("mul.rn.e4m3x4.e2m1x4.e4m3x4.satfinite %0, %1, %2;" : "=r"(d) : "h"(a), "r"(scale));
  return d;
}

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 480
#define TMEM_TMEM_ACC_OFFSET 0
#define TMEM_SFA0_OFFSET 320
#define TMEM_SFA1_OFFSET 328
#define TMEM_SFA2_OFFSET 336
#define TMEM_SFA3_OFFSET 344
#define TMEM_SFB0_0_OFFSET 352
#define TMEM_SFB0_1_OFFSET 360
#define TMEM_SFB0_2_OFFSET 368
#define TMEM_SFB0_3_OFFSET 376
#define TMEM_SFB1_0_OFFSET 384
#define TMEM_SFB1_1_OFFSET 392
#define TMEM_SFB1_2_OFFSET 400
#define TMEM_SFB1_3_OFFSET 408
#define TMEM_TMEM_Q_OFFSET 416
#define NUM_RAW_PIPE_STAGES 3
#define NUM_V_PIPE_STAGES 2
#define NUM_P_PIPE_STAGES 2
#define SMEM_SMEM_V0_OFF 1024
#define SMEM_SMEM_V0_STAGE_BYTES 4096
#define SMEM_SMEM_V0_STRIDE 4096
#define SMEM_SMEM_V1_OFF 5120
#define SMEM_SMEM_V1_STAGE_BYTES 4096
#define SMEM_SMEM_V1_STRIDE 4096
#define SMEM_SMEM_V2_OFF 9216
#define SMEM_SMEM_V2_STAGE_BYTES 4096
#define SMEM_SMEM_V2_STRIDE 4096
#define SMEM_SMEM_V3_OFF 13312
#define SMEM_SMEM_V3_STAGE_BYTES 4096
#define SMEM_SMEM_V3_STRIDE 4096
#define SMEM_SMEM_QS_OFF 17408
#define SMEM_SMEM_QS_STAGE_BYTES 2048
#define SMEM_SMEM_QS_STRIDE 2048
#define SMEM_SMEM_QR_OFF 19456
#define SMEM_SMEM_QR_STAGE_BYTES 4096
#define SMEM_SMEM_QR_STRIDE 4096
#define SMEM_SMEM_V6_OFF 23552
#define SMEM_SMEM_V6_STAGE_BYTES 2048
#define SMEM_SMEM_V6_STRIDE 32768
#define SMEM_SMEM_V7_OFF 27648
#define SMEM_SMEM_V7_STAGE_BYTES 2048
#define SMEM_SMEM_V7_STRIDE 32768
#define SMEM_SMEM_V8_OFF 31744
#define SMEM_SMEM_V8_STAGE_BYTES 2048
#define SMEM_SMEM_V8_STRIDE 32768
#define SMEM_SMEM_V9_OFF 35840
#define SMEM_SMEM_V9_STAGE_BYTES 2048
#define SMEM_SMEM_V9_STRIDE 32768
#define SMEM_SMEM_V10_OFF 25600
#define SMEM_SMEM_V10_STAGE_BYTES 2048
#define SMEM_SMEM_V10_STRIDE 32768
#define SMEM_SMEM_V11_OFF 29696
#define SMEM_SMEM_V11_STAGE_BYTES 2048
#define SMEM_SMEM_V11_STRIDE 32768
#define SMEM_SMEM_V12_OFF 33792
#define SMEM_SMEM_V12_STAGE_BYTES 2048
#define SMEM_SMEM_V12_STRIDE 32768
#define SMEM_SMEM_V13_OFF 37888
#define SMEM_SMEM_V13_STAGE_BYTES 2048
#define SMEM_SMEM_V13_STRIDE 32768
#define SMEM_SMEM_V14_OFF 23552
#define SMEM_SMEM_V14_STAGE_BYTES 4096
#define SMEM_SMEM_V14_STRIDE 32768
#define SMEM_SMEM_V15_OFF 27648
#define SMEM_SMEM_V15_STAGE_BYTES 4096
#define SMEM_SMEM_V15_STRIDE 32768
#define SMEM_SMEM_V16_OFF 31744
#define SMEM_SMEM_V16_STAGE_BYTES 4096
#define SMEM_SMEM_V16_STRIDE 32768
#define SMEM_SMEM_V17_OFF 35840
#define SMEM_SMEM_V17_STAGE_BYTES 4096
#define SMEM_SMEM_V17_STRIDE 32768
#define SMEM_SMEM_V18_OFF 25600
#define SMEM_SMEM_V18_STAGE_BYTES 4096
#define SMEM_SMEM_V18_STRIDE 32768
#define SMEM_SMEM_V19_OFF 29696
#define SMEM_SMEM_V19_STAGE_BYTES 4096
#define SMEM_SMEM_V19_STRIDE 32768
#define SMEM_SMEM_V20_OFF 33792
#define SMEM_SMEM_V20_STAGE_BYTES 4096
#define SMEM_SMEM_V20_STRIDE 32768
#define SMEM_SMEM_V21_OFF 37888
#define SMEM_SMEM_V21_STAGE_BYTES 4096
#define SMEM_SMEM_V21_STRIDE 32768
#define SMEM_SMEM_V22_OFF 39936
#define SMEM_SMEM_V22_STAGE_BYTES 2048
#define SMEM_SMEM_V22_STRIDE 32768
#define SMEM_SMEM_V23_OFF 44032
#define SMEM_SMEM_V23_STAGE_BYTES 2048
#define SMEM_SMEM_V23_STRIDE 32768
#define SMEM_SMEM_V24_OFF 41984
#define SMEM_SMEM_V24_STAGE_BYTES 2048
#define SMEM_SMEM_V24_STRIDE 32768
#define SMEM_SMEM_V25_OFF 46080
#define SMEM_SMEM_V25_STAGE_BYTES 2048
#define SMEM_SMEM_V25_STRIDE 32768
#define SMEM_SMEM_RAW_FLAT_OFF 23552
#define SMEM_SMEM_RAW_FLAT_STAGE_BYTES 98304
#define SMEM_SMEM_RAW_FLAT_STRIDE 98304
#define SMEM_SMEM_KS_OFF 48128
#define SMEM_SMEM_KS_STAGE_BYTES 4096
#define SMEM_SMEM_KS_STRIDE 32768
#define SMEM_SMEM_V28_OFF 52224
#define SMEM_SMEM_V28_STAGE_BYTES 2048
#define SMEM_SMEM_V28_STRIDE 32768
#define SMEM_SMEM_V29_OFF 54272
#define SMEM_SMEM_V29_STAGE_BYTES 2048
#define SMEM_SMEM_V29_STRIDE 32768
#define SMEM_SMEM_V30_OFF 121856
#define SMEM_SMEM_V30_STAGE_BYTES 16384
#define SMEM_SMEM_V30_STRIDE 32768
#define SMEM_SMEM_V31_OFF 138240
#define SMEM_SMEM_V31_STAGE_BYTES 16384
#define SMEM_SMEM_V31_STRIDE 32768
#define SMEM_SMEM_P_OFF 187392
#define SMEM_SMEM_P_STAGE_BYTES 8192
#define SMEM_SMEM_P_STRIDE 8192
#define SMEM_ROW_STATE_OFF 203776
#define SMEM_ROW_STATE_STAGE_BYTES 512
#define SMEM_ROW_STATE_STRIDE 512
#define SMEM_SMEM_XCHG_MAX_OFF 204288
#define SMEM_SMEM_XCHG_MAX_STAGE_BYTES 512
#define SMEM_SMEM_XCHG_MAX_STRIDE 512
#define SMEM_SMEM_XCHG_SUM_OFF 204800
#define SMEM_SMEM_XCHG_SUM_STAGE_BYTES 512
#define SMEM_SMEM_XCHG_SUM_STRIDE 512
#define SMEM_TOTAL 205824
#define THREADS 512
#define LAUNCH_MIN_BLOCKS 1

extern "C" {

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_mla_nvfp4_paged_decode_e7cb458f5e586352cfcb(const __grid_constant__ CUtensorMap tmap_qn, const __grid_constant__ CUtensorMap tmap_qs, const __grid_constant__ CUtensorMap tmap_qr, const __grid_constant__ CUtensorMap tmap_k, const __grid_constant__ CUtensorMap tmap_ks, const __grid_constant__ CUtensorMap tmap_kr, const __grid_constant__ CUtensorMap tmap_po, float* __restrict__ q_scale, __nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_max, float* __restrict__ partial_sum, float* __restrict__ lse, int* __restrict__ seq_lens, int* __restrict__ kv_len_global, int* __restrict__ cum_seq_lens_q, int* __restrict__ page_table, float softmax_scale_log2, float bmm2_scale, int num_heads, int num_split, int max_pages_per_seq, int page_shift, int cp_world, int cp_rank, int has_lse)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 8)
    #define raw_full_addr (mbar_base + 16)
    #define raw_empty_addr (mbar_base + 40)
    #define sfa_full_addr (mbar_base + 64)
    #define sf_full_addr (mbar_base + 72)
    #define sf_empty_addr (mbar_base + 80)
    #define s_full_addr (mbar_base + 88)
    #define s_empty_addr (mbar_base + 96)
    #define v_full_addr (mbar_base + 104)
    #define v_empty_addr (mbar_base + 120)
    #define corr_sig_addr (mbar_base + 136)
    #define corr_empty_addr (mbar_base + 144)
    #define p_full_addr (mbar_base + 152)
    #define p_empty_addr (mbar_base + 168)
    #define pv_done_addr (mbar_base + 184)
    #define o_full_addr (mbar_base + 192)
    #define o_empty_addr (mbar_base + 200)
    #define epi_done_addr (mbar_base + 208)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    uint8_t* smem_v0 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V0_OFF);
    const int smem_v0_addr = smem + SMEM_SMEM_V0_OFF;
    uint8_t* smem_v1 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V1_OFF);
    const int smem_v1_addr = smem + SMEM_SMEM_V1_OFF;
    uint8_t* smem_v2 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V2_OFF);
    const int smem_v2_addr = smem + SMEM_SMEM_V2_OFF;
    uint8_t* smem_v3 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V3_OFF);
    const int smem_v3_addr = smem + SMEM_SMEM_V3_OFF;
    uint8_t* smem_qs = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_QS_OFF);
    const int smem_qs_addr = smem + SMEM_SMEM_QS_OFF;
    uint8_t* smem_qr = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_QR_OFF);
    const int smem_qr_addr = smem + SMEM_SMEM_QR_OFF;
    uint8_t* smem_v6 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V6_OFF);
    const int smem_v6_addr = smem + SMEM_SMEM_V6_OFF;
    uint8_t* smem_v7 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V7_OFF);
    const int smem_v7_addr = smem + SMEM_SMEM_V7_OFF;
    uint8_t* smem_v8 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V8_OFF);
    const int smem_v8_addr = smem + SMEM_SMEM_V8_OFF;
    uint8_t* smem_v9 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V9_OFF);
    const int smem_v9_addr = smem + SMEM_SMEM_V9_OFF;
    uint8_t* smem_v10 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V10_OFF);
    const int smem_v10_addr = smem + SMEM_SMEM_V10_OFF;
    uint8_t* smem_v11 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V11_OFF);
    const int smem_v11_addr = smem + SMEM_SMEM_V11_OFF;
    uint8_t* smem_v12 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V12_OFF);
    const int smem_v12_addr = smem + SMEM_SMEM_V12_OFF;
    uint8_t* smem_v13 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V13_OFF);
    const int smem_v13_addr = smem + SMEM_SMEM_V13_OFF;
    uint8_t* smem_v14 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V14_OFF);
    const int smem_v14_addr = smem + SMEM_SMEM_V14_OFF;
    uint8_t* smem_v15 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V15_OFF);
    const int smem_v15_addr = smem + SMEM_SMEM_V15_OFF;
    uint8_t* smem_v16 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V16_OFF);
    const int smem_v16_addr = smem + SMEM_SMEM_V16_OFF;
    uint8_t* smem_v17 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V17_OFF);
    const int smem_v17_addr = smem + SMEM_SMEM_V17_OFF;
    uint8_t* smem_v18 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V18_OFF);
    const int smem_v18_addr = smem + SMEM_SMEM_V18_OFF;
    uint8_t* smem_v19 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V19_OFF);
    const int smem_v19_addr = smem + SMEM_SMEM_V19_OFF;
    uint8_t* smem_v20 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V20_OFF);
    const int smem_v20_addr = smem + SMEM_SMEM_V20_OFF;
    uint8_t* smem_v21 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V21_OFF);
    const int smem_v21_addr = smem + SMEM_SMEM_V21_OFF;
    uint8_t* smem_v22 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V22_OFF);
    const int smem_v22_addr = smem + SMEM_SMEM_V22_OFF;
    uint8_t* smem_v23 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V23_OFF);
    const int smem_v23_addr = smem + SMEM_SMEM_V23_OFF;
    uint8_t* smem_v24 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V24_OFF);
    const int smem_v24_addr = smem + SMEM_SMEM_V24_OFF;
    uint8_t* smem_v25 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V25_OFF);
    const int smem_v25_addr = smem + SMEM_SMEM_V25_OFF;
    uint8_t* smem_raw_flat = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_RAW_FLAT_OFF);
    const int smem_raw_flat_addr = smem + SMEM_SMEM_RAW_FLAT_OFF;
    uint8_t* smem_ks = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_KS_OFF);
    const int smem_ks_addr = smem + SMEM_SMEM_KS_OFF;
    uint8_t* smem_v28 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V28_OFF);
    const int smem_v28_addr = smem + SMEM_SMEM_V28_OFF;
    uint8_t* smem_v29 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V29_OFF);
    const int smem_v29_addr = smem + SMEM_SMEM_V29_OFF;
    uint8_t* smem_v30 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V30_OFF);
    const int smem_v30_addr = smem + SMEM_SMEM_V30_OFF;
    uint8_t* smem_v31 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V31_OFF);
    const int smem_v31_addr = smem + SMEM_SMEM_V31_OFF;
    uint8_t* smem_p = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_P_OFF);
    const int smem_p_addr = smem + SMEM_SMEM_P_OFF;
    float* row_state = reinterpret_cast<float*>(smem_raw + SMEM_ROW_STATE_OFF);
    const int row_state_addr = smem + SMEM_ROW_STATE_OFF;
    float* smem_xchg_max = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_XCHG_MAX_OFF);
    const int smem_xchg_max_addr = smem + SMEM_SMEM_XCHG_MAX_OFF;
    float* smem_xchg_sum = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_XCHG_SUM_OFF);
    const int smem_xchg_sum_addr = smem + SMEM_SMEM_XCHG_SUM_OFF;

    // Mbarrier init (19 pipeline groups, 0 ordered-sequence groups, 27 barriers)
    // Mbarriers at smem_raw[0..216)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'raw_pipe' ---
            // raw_full: 3 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // raw_empty: 3 barriers, init_count=321
            mbarrier_init(smem + 40, 321);
            mbarrier_init(smem + 48, 321);
            mbarrier_init(smem + 56, 321);
            // sfa_full: 1 barriers, init_count=256
            mbarrier_init(smem + 64, 256);
            // sf_full: 1 barriers, init_count=256
            mbarrier_init(smem + 72, 256);
            // sf_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // s_full: 1 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            // s_empty: 1 barriers, init_count=256
            mbarrier_init(smem + 96, 256);
            // --- pipeline 'v_pipe' ---
            // v_full: 2 barriers, init_count=384
            mbarrier_init(smem + 104, 384);
            mbarrier_init(smem + 112, 384);
            // v_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            // corr_sig: 1 barriers, init_count=128
            mbarrier_init(smem + 136, 128);
            // corr_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 144, 128);
            // --- pipeline 'p_pipe' ---
            // p_full: 2 barriers, init_count=512
            mbarrier_init(smem + 152, 512);
            mbarrier_init(smem + 160, 512);
            // p_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            // pv_done: 1 barriers, init_count=1
            mbarrier_init(smem + 184, 1);
            // o_full: 1 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            // o_empty: 1 barriers, init_count=256
            mbarrier_init(smem + 200, 256);
            // epi_done: 1 barriers, init_count=128
            mbarrier_init(smem + 208, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 480 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 216);
    if (warp == 0) {
        int _tmem_hold = smem + 216;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_acc = taddr;
    const int tmem_sfa0 = taddr + 320;
    const int tmem_sfa1 = taddr + 328;
    const int tmem_sfa2 = taddr + 336;
    const int tmem_sfa3 = taddr + 344;
    const int tmem_sfb0_0 = taddr + 352;
    const int tmem_sfb0_1 = taddr + 360;
    const int tmem_sfb0_2 = taddr + 368;
    const int tmem_sfb0_3 = taddr + 376;
    const int tmem_sfb1_0 = taddr + 384;
    const int tmem_sfb1_1 = taddr + 392;
    const int tmem_sfb1_2 = taddr + 400;
    const int tmem_sfb1_3 = taddr + 408;
    const int tmem_tmem_q = taddr + 416;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 96;");
    }

    // ---- Role: softmax ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 208;");
        { // softmax_main
            int quadrant = make_warp_uniform(warp % 4);
            int n_half = make_warp_uniform(quadrant / 2);
            int row_c = quadrant % 2 * 32 + lane;
            int lane_base = quadrant * 32 << 16;
            int xchg_idx = n_half * 64 + row_c;
            int xchg_bar = make_warp_uniform(2 + quadrant % 2);
            int rank_s = cta_rank;
            int p_stage_s = 0;
            int p_phase_s = 1;
            int row_in_tile = rank_s * 64 + row_c;
            int cx = blockIdx.x / 2;
            int n_clusters = gridDim.x / 2;
            int bz = blockIdx.z;
            int q_end_bz = cum_seq_lens_q[bz + 1];
            int q_start_bz = cum_seq_lens_q[bz];
            int kv_len_bz = seq_lens[bz];
            int g_len_bz = kv_len_global[bz];
            int rows_bz = (q_end_bz - q_start_bz) * num_heads;
            int pieces = (rows_bz + 128 - 1) / 128 * num_split;
            int left = pieces - cx;
            int n_pieces_raw = (left + n_clusters - 1) / n_clusters;
            int n_pieces = ((left <= 0) ? 0 : n_pieces_raw);
            unsigned int _phase_s_full_0 = 0;
            unsigned int _phase_corr_empty_0 = 1;
            unsigned int _phase_o_full_0 = 0;
            #pragma unroll 1
            for (int pi_s = 0; pi_s < n_pieces; pi_s++) {
                int p_s = cx + pi_s * n_clusters;
                int b = blockIdx.z;
                int split_idx = p_s % num_split;
                int rows_b0 = (q_end_bz - q_start_bz) * num_heads;
                int m_tiles_b = (rows_b0 + 128 - 1) / 128;
                int m_tile = m_tiles_b - 1 - p_s / num_split;
                int q_start = q_start_bz;
                int q_len_b = q_end_bz - q_start_bz;
                int kv_len = kv_len_bz;
                int g_len = g_len_bz;
                int rows_b = q_len_b * num_heads;
                int row0 = m_tile * 128;
                int rows_left = rows_b - row0;
                int rows_pos = ((rows_left < 0) ? 0 : rows_left);
                int rows_valid = ((rows_pos > 128) ? 128 : rows_pos);
                int row_base_global = q_start * num_heads + row0;
                int last_row = row0 + rows_valid - 1;
                int t_last = last_row / num_heads;
                int num = g_len - q_len_b + t_last - cp_rank;
                int vis_raw = num / cp_world + 1;
                int vis_cap = ((vis_raw > kv_len) ? kv_len : vis_raw);
                int vis_out = ((num < 0) ? 0 : vis_cap);
                int kv_end_raw = vis_out;
                int kv_end = ((rows_valid == 0) ? 0 : kv_end_raw);
                int n_tiles_total = (kv_end + 128 - 1) / 128;
                int tiles_per_split = (n_tiles_total + num_split - 1) / num_split;
                int my_start_raw = split_idx * tiles_per_split;
                int my_end_raw = my_start_raw + tiles_per_split;
                int my_end = ((my_end_raw > n_tiles_total) ? n_tiles_total : my_end_raw);
                int my_n_raw = my_end - my_start_raw;
                int empty = ((my_n_raw < 1) ? 1 : 0);
                int my_n_tiles = ((my_n_raw < 1) ? 1 : my_n_raw);
                int my_start = ((my_n_raw < 1) ? 0 : my_start_raw);
                int pt_base = b * max_pages_per_seq;
                int row_in_req = row0 + row_in_tile;
                int row_is_query = row_in_tile < rows_valid;
                int row_valid = row_is_query & (int)(empty == 0);
                int token_s = row_in_req / num_heads;
                int out_row = row_base_global + row_in_tile;
                float q_scale_row = ((row_is_query != 0) ? q_scale[out_row] : 1.0f);
                float scale_row = softmax_scale_log2 * q_scale_row;
                int num_0 = g_len - q_len_b + token_s - cp_rank;
                int vis_raw_1 = num_0 / cp_world + 1;
                int vis_cap_2 = ((vis_raw_1 > kv_len) ? kv_len : vis_raw_1);
                int vis_out_3 = ((num_0 < 0) ? 0 : vis_cap_2);
                int row_limit_raw = vis_out_3;
                int row_limit = ((row_valid != 0) ? row_limit_raw : 0);
                float row_max = -CAKE_INF;
                float row_sum = 0.0f;
                #pragma unroll 1
                for (int it_s = 0; it_s < my_n_tiles; it_s++) {
                    float sv0[32];
                    float sv1[32];
                    mbarrier_wait(s_full_addr, _phase_s_full_0);
                    _phase_s_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(sv0[0]), "=f"(sv0[1]), "=f"(sv0[2]), "=f"(sv0[3]), "=f"(sv0[4]), "=f"(sv0[5]), "=f"(sv0[6]), "=f"(sv0[7]), "=f"(sv0[8]), "=f"(sv0[9]), "=f"(sv0[10]), "=f"(sv0[11]), "=f"(sv0[12]), "=f"(sv0[13]), "=f"(sv0[14]), "=f"(sv0[15]), "=f"(sv0[16]), "=f"(sv0[17]), "=f"(sv0[18]), "=f"(sv0[19]), "=f"(sv0[20]), "=f"(sv0[21]), "=f"(sv0[22]), "=f"(sv0[23]), "=f"(sv0[24]), "=f"(sv0[25]), "=f"(sv0[26]), "=f"(sv0[27]), "=f"(sv0[28]), "=f"(sv0[29]), "=f"(sv0[30]), "=f"(sv0[31])
                        : "r"(taddr + (unsigned int)lane_base));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(sv1[0]), "=f"(sv1[1]), "=f"(sv1[2]), "=f"(sv1[3]), "=f"(sv1[4]), "=f"(sv1[5]), "=f"(sv1[6]), "=f"(sv1[7]), "=f"(sv1[8]), "=f"(sv1[9]), "=f"(sv1[10]), "=f"(sv1[11]), "=f"(sv1[12]), "=f"(sv1[13]), "=f"(sv1[14]), "=f"(sv1[15]), "=f"(sv1[16]), "=f"(sv1[17]), "=f"(sv1[18]), "=f"(sv1[19]), "=f"(sv1[20]), "=f"(sv1[21]), "=f"(sv1[22]), "=f"(sv1[23]), "=f"(sv1[24]), "=f"(sv1[25]), "=f"(sv1[26]), "=f"(sv1[27]), "=f"(sv1[28]), "=f"(sv1[29]), "=f"(sv1[30]), "=f"(sv1[31])
                        : "r"(taddr + (unsigned int)lane_base + 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(s_empty_addr), "r"(0) : "memory");
                    int tile_base_s = (my_start + it_s) * 128 + n_half * 32;
                    int _max_0 = ((row_limit - tile_base_s) > (0) ? (row_limit - tile_base_s) : (0));
                    int _min_0 = ((_max_0) < (32) ? (_max_0) : (32));
                    int valid_0 = _min_0;
                    int _max_1 = ((row_limit - tile_base_s - 64) > (0) ? (row_limit - tile_base_s - 64) : (0));
                    int _min_1 = ((_max_1) < (32) ? (_max_1) : (32));
                    int valid_1 = _min_1;
                    if (valid_0 < 32) {
                        uint32_t _slice_lo_mask_0;
                        {
                            int _lim_0 = valid_0;
                            if (_lim_0 <= 0) { _slice_lo_mask_0 = 0u; }
                            else if (_lim_0 >= 32) { _slice_lo_mask_0 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_0) : "r"(_lim_0));
                            }
                        }
                        if (!(_slice_lo_mask_0 & (1u << 0))) sv0[0] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 1))) sv0[1] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 2))) sv0[2] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 3))) sv0[3] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 4))) sv0[4] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 5))) sv0[5] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 6))) sv0[6] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 7))) sv0[7] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 8))) sv0[8] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 9))) sv0[9] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 10))) sv0[10] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 11))) sv0[11] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 12))) sv0[12] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 13))) sv0[13] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 14))) sv0[14] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 15))) sv0[15] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 16))) sv0[16] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 17))) sv0[17] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 18))) sv0[18] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 19))) sv0[19] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 20))) sv0[20] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 21))) sv0[21] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 22))) sv0[22] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 23))) sv0[23] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 24))) sv0[24] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 25))) sv0[25] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 26))) sv0[26] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 27))) sv0[27] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 28))) sv0[28] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 29))) sv0[29] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 30))) sv0[30] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 31))) sv0[31] = -CAKE_INF;
                    }
                    if (valid_1 < 32) {
                        uint32_t _slice_lo_mask_1;
                        {
                            int _lim_1 = valid_1;
                            if (_lim_1 <= 0) { _slice_lo_mask_1 = 0u; }
                            else if (_lim_1 >= 32) { _slice_lo_mask_1 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_1) : "r"(_lim_1));
                            }
                        }
                        if (!(_slice_lo_mask_1 & (1u << 0))) sv1[0] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 1))) sv1[1] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 2))) sv1[2] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 3))) sv1[3] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 4))) sv1[4] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 5))) sv1[5] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 6))) sv1[6] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 7))) sv1[7] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 8))) sv1[8] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 9))) sv1[9] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 10))) sv1[10] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 11))) sv1[11] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 12))) sv1[12] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 13))) sv1[13] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 14))) sv1[14] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 15))) sv1[15] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 16))) sv1[16] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 17))) sv1[17] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 18))) sv1[18] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 19))) sv1[19] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 20))) sv1[20] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 21))) sv1[21] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 22))) sv1[22] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 23))) sv1[23] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 24))) sv1[24] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 25))) sv1[25] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 26))) sv1[26] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 27))) sv1[27] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 28))) sv1[28] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 29))) sv1[29] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 30))) sv1[30] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 31))) sv1[31] = -CAKE_INF;
                    }
                    float2 _reg_reduce_max2_2 = {-CAKE_INF, -CAKE_INF};
                    _reg_reduce_max2_2.x = max_noftz(_reg_reduce_max2_2.x, max_noftz(sv0[0], sv0[1]));
                    _reg_reduce_max2_2.y = max_noftz(_reg_reduce_max2_2.y, max_noftz(sv0[2], sv0[3]));
                    _reg_reduce_max2_2.x = max_noftz(_reg_reduce_max2_2.x, max_noftz(sv0[4], sv0[5]));
                    _reg_reduce_max2_2.y = max_noftz(_reg_reduce_max2_2.y, max_noftz(sv0[6], sv0[7]));
                    _reg_reduce_max2_2.x = max_noftz(_reg_reduce_max2_2.x, max_noftz(sv0[8], sv0[9]));
                    _reg_reduce_max2_2.y = max_noftz(_reg_reduce_max2_2.y, max_noftz(sv0[10], sv0[11]));
                    _reg_reduce_max2_2.x = max_noftz(_reg_reduce_max2_2.x, max_noftz(sv0[12], sv0[13]));
                    _reg_reduce_max2_2.y = max_noftz(_reg_reduce_max2_2.y, max_noftz(sv0[14], sv0[15]));
                    _reg_reduce_max2_2.x = max_noftz(_reg_reduce_max2_2.x, max_noftz(sv0[16], sv0[17]));
                    _reg_reduce_max2_2.y = max_noftz(_reg_reduce_max2_2.y, max_noftz(sv0[18], sv0[19]));
                    _reg_reduce_max2_2.x = max_noftz(_reg_reduce_max2_2.x, max_noftz(sv0[20], sv0[21]));
                    _reg_reduce_max2_2.y = max_noftz(_reg_reduce_max2_2.y, max_noftz(sv0[22], sv0[23]));
                    _reg_reduce_max2_2.x = max_noftz(_reg_reduce_max2_2.x, max_noftz(sv0[24], sv0[25]));
                    _reg_reduce_max2_2.y = max_noftz(_reg_reduce_max2_2.y, max_noftz(sv0[26], sv0[27]));
                    _reg_reduce_max2_2.x = max_noftz(_reg_reduce_max2_2.x, max_noftz(sv0[28], sv0[29]));
                    _reg_reduce_max2_2.y = max_noftz(_reg_reduce_max2_2.y, max_noftz(sv0[30], sv0[31]));
                    float sv0_max = row_max_reduce(_reg_reduce_max2_2);
                    float2 _reg_reduce_max2_3 = {-CAKE_INF, -CAKE_INF};
                    _reg_reduce_max2_3.x = max_noftz(_reg_reduce_max2_3.x, max_noftz(sv1[0], sv1[1]));
                    _reg_reduce_max2_3.y = max_noftz(_reg_reduce_max2_3.y, max_noftz(sv1[2], sv1[3]));
                    _reg_reduce_max2_3.x = max_noftz(_reg_reduce_max2_3.x, max_noftz(sv1[4], sv1[5]));
                    _reg_reduce_max2_3.y = max_noftz(_reg_reduce_max2_3.y, max_noftz(sv1[6], sv1[7]));
                    _reg_reduce_max2_3.x = max_noftz(_reg_reduce_max2_3.x, max_noftz(sv1[8], sv1[9]));
                    _reg_reduce_max2_3.y = max_noftz(_reg_reduce_max2_3.y, max_noftz(sv1[10], sv1[11]));
                    _reg_reduce_max2_3.x = max_noftz(_reg_reduce_max2_3.x, max_noftz(sv1[12], sv1[13]));
                    _reg_reduce_max2_3.y = max_noftz(_reg_reduce_max2_3.y, max_noftz(sv1[14], sv1[15]));
                    _reg_reduce_max2_3.x = max_noftz(_reg_reduce_max2_3.x, max_noftz(sv1[16], sv1[17]));
                    _reg_reduce_max2_3.y = max_noftz(_reg_reduce_max2_3.y, max_noftz(sv1[18], sv1[19]));
                    _reg_reduce_max2_3.x = max_noftz(_reg_reduce_max2_3.x, max_noftz(sv1[20], sv1[21]));
                    _reg_reduce_max2_3.y = max_noftz(_reg_reduce_max2_3.y, max_noftz(sv1[22], sv1[23]));
                    _reg_reduce_max2_3.x = max_noftz(_reg_reduce_max2_3.x, max_noftz(sv1[24], sv1[25]));
                    _reg_reduce_max2_3.y = max_noftz(_reg_reduce_max2_3.y, max_noftz(sv1[26], sv1[27]));
                    _reg_reduce_max2_3.x = max_noftz(_reg_reduce_max2_3.x, max_noftz(sv1[28], sv1[29]));
                    _reg_reduce_max2_3.y = max_noftz(_reg_reduce_max2_3.y, max_noftz(sv1[30], sv1[31]));
                    float sv1_max = row_max_reduce(_reg_reduce_max2_3);
                    float _max_2 = max_noftz(sv0_max, sv1_max);
                    float tile_max = _max_2;
                    float tile_max_scaled = ((tile_max > -CAKE_INF) ? tile_max * scale_row : -CAKE_INF);
                    float _max_3 = max_noftz(row_max, tile_max_scaled);
                    float new_max = _max_3;
                    smem_xchg_max[xchg_idx] = new_max;
                    asm volatile("barrier.sync %0, 64;" :: "r"(xchg_bar) : "memory");
                    float _max_4 = max_noftz(new_max, smem_xchg_max[xchg_idx ^ 64]);
                    new_max = _max_4;
                    asm volatile("barrier.sync %0, 64;" :: "r"(xchg_bar) : "memory");
                    if (row_max > -CAKE_INF && new_max - row_max <= 2.0f) {
                        new_max = row_max;
                    }
                    float safe_max = ((new_max == -CAKE_INF) ? 0.0f : new_max);
                    float _exp2_0 = approx_exp2(row_max - safe_max);
                    float alpha = ((row_max > -CAKE_INF) ? _exp2_0 : 1.0f);
                    row_max = new_max;
                    mbarrier_wait(corr_empty_addr, _phase_corr_empty_0);
                    _phase_corr_empty_0 ^= 1;
                    row_state[row_c] = alpha;
                    mbarrier_arrive(corr_sig_addr);
                    const float2 _fma_b2_4 = {scale_row, scale_row};
                    const float2 _fma_c2_5 = {6.0f - safe_max, 6.0f - safe_max};
                    float2 _fma_pair_6 = fma_f32x2(make_float2(sv0[0], sv0[1]), _fma_b2_4, _fma_c2_5);
                    sv0[0] = _fma_pair_6.x;
                    sv0[1] = _fma_pair_6.y;
                    float2 _fma_pair_7 = fma_f32x2(make_float2(sv0[2], sv0[3]), _fma_b2_4, _fma_c2_5);
                    sv0[2] = _fma_pair_7.x;
                    sv0[3] = _fma_pair_7.y;
                    float2 _fma_pair_8 = fma_f32x2(make_float2(sv0[4], sv0[5]), _fma_b2_4, _fma_c2_5);
                    sv0[4] = _fma_pair_8.x;
                    sv0[5] = _fma_pair_8.y;
                    float2 _fma_pair_9 = fma_f32x2(make_float2(sv0[6], sv0[7]), _fma_b2_4, _fma_c2_5);
                    sv0[6] = _fma_pair_9.x;
                    sv0[7] = _fma_pair_9.y;
                    float2 _fma_pair_10 = fma_f32x2(make_float2(sv0[8], sv0[9]), _fma_b2_4, _fma_c2_5);
                    sv0[8] = _fma_pair_10.x;
                    sv0[9] = _fma_pair_10.y;
                    float2 _fma_pair_11 = fma_f32x2(make_float2(sv0[10], sv0[11]), _fma_b2_4, _fma_c2_5);
                    sv0[10] = _fma_pair_11.x;
                    sv0[11] = _fma_pair_11.y;
                    float2 _fma_pair_12 = fma_f32x2(make_float2(sv0[12], sv0[13]), _fma_b2_4, _fma_c2_5);
                    sv0[12] = _fma_pair_12.x;
                    sv0[13] = _fma_pair_12.y;
                    float2 _fma_pair_13 = fma_f32x2(make_float2(sv0[14], sv0[15]), _fma_b2_4, _fma_c2_5);
                    sv0[14] = _fma_pair_13.x;
                    sv0[15] = _fma_pair_13.y;
                    float2 _fma_pair_14 = fma_f32x2(make_float2(sv0[16], sv0[17]), _fma_b2_4, _fma_c2_5);
                    sv0[16] = _fma_pair_14.x;
                    sv0[17] = _fma_pair_14.y;
                    float2 _fma_pair_15 = fma_f32x2(make_float2(sv0[18], sv0[19]), _fma_b2_4, _fma_c2_5);
                    sv0[18] = _fma_pair_15.x;
                    sv0[19] = _fma_pair_15.y;
                    float2 _fma_pair_16 = fma_f32x2(make_float2(sv0[20], sv0[21]), _fma_b2_4, _fma_c2_5);
                    sv0[20] = _fma_pair_16.x;
                    sv0[21] = _fma_pair_16.y;
                    float2 _fma_pair_17 = fma_f32x2(make_float2(sv0[22], sv0[23]), _fma_b2_4, _fma_c2_5);
                    sv0[22] = _fma_pair_17.x;
                    sv0[23] = _fma_pair_17.y;
                    float2 _fma_pair_18 = fma_f32x2(make_float2(sv0[24], sv0[25]), _fma_b2_4, _fma_c2_5);
                    sv0[24] = _fma_pair_18.x;
                    sv0[25] = _fma_pair_18.y;
                    float2 _fma_pair_19 = fma_f32x2(make_float2(sv0[26], sv0[27]), _fma_b2_4, _fma_c2_5);
                    sv0[26] = _fma_pair_19.x;
                    sv0[27] = _fma_pair_19.y;
                    float2 _fma_pair_20 = fma_f32x2(make_float2(sv0[28], sv0[29]), _fma_b2_4, _fma_c2_5);
                    sv0[28] = _fma_pair_20.x;
                    sv0[29] = _fma_pair_20.y;
                    float2 _fma_pair_21 = fma_f32x2(make_float2(sv0[30], sv0[31]), _fma_b2_4, _fma_c2_5);
                    sv0[30] = _fma_pair_21.x;
                    sv0[31] = _fma_pair_21.y;
                    const float2 _fma_b2_22 = {scale_row, scale_row};
                    const float2 _fma_c2_23 = {6.0f - safe_max, 6.0f - safe_max};
                    float2 _fma_pair_24 = fma_f32x2(make_float2(sv1[0], sv1[1]), _fma_b2_22, _fma_c2_23);
                    sv1[0] = _fma_pair_24.x;
                    sv1[1] = _fma_pair_24.y;
                    float2 _fma_pair_25 = fma_f32x2(make_float2(sv1[2], sv1[3]), _fma_b2_22, _fma_c2_23);
                    sv1[2] = _fma_pair_25.x;
                    sv1[3] = _fma_pair_25.y;
                    float2 _fma_pair_26 = fma_f32x2(make_float2(sv1[4], sv1[5]), _fma_b2_22, _fma_c2_23);
                    sv1[4] = _fma_pair_26.x;
                    sv1[5] = _fma_pair_26.y;
                    float2 _fma_pair_27 = fma_f32x2(make_float2(sv1[6], sv1[7]), _fma_b2_22, _fma_c2_23);
                    sv1[6] = _fma_pair_27.x;
                    sv1[7] = _fma_pair_27.y;
                    float2 _fma_pair_28 = fma_f32x2(make_float2(sv1[8], sv1[9]), _fma_b2_22, _fma_c2_23);
                    sv1[8] = _fma_pair_28.x;
                    sv1[9] = _fma_pair_28.y;
                    float2 _fma_pair_29 = fma_f32x2(make_float2(sv1[10], sv1[11]), _fma_b2_22, _fma_c2_23);
                    sv1[10] = _fma_pair_29.x;
                    sv1[11] = _fma_pair_29.y;
                    float2 _fma_pair_30 = fma_f32x2(make_float2(sv1[12], sv1[13]), _fma_b2_22, _fma_c2_23);
                    sv1[12] = _fma_pair_30.x;
                    sv1[13] = _fma_pair_30.y;
                    float2 _fma_pair_31 = fma_f32x2(make_float2(sv1[14], sv1[15]), _fma_b2_22, _fma_c2_23);
                    sv1[14] = _fma_pair_31.x;
                    sv1[15] = _fma_pair_31.y;
                    float2 _fma_pair_32 = fma_f32x2(make_float2(sv1[16], sv1[17]), _fma_b2_22, _fma_c2_23);
                    sv1[16] = _fma_pair_32.x;
                    sv1[17] = _fma_pair_32.y;
                    float2 _fma_pair_33 = fma_f32x2(make_float2(sv1[18], sv1[19]), _fma_b2_22, _fma_c2_23);
                    sv1[18] = _fma_pair_33.x;
                    sv1[19] = _fma_pair_33.y;
                    float2 _fma_pair_34 = fma_f32x2(make_float2(sv1[20], sv1[21]), _fma_b2_22, _fma_c2_23);
                    sv1[20] = _fma_pair_34.x;
                    sv1[21] = _fma_pair_34.y;
                    float2 _fma_pair_35 = fma_f32x2(make_float2(sv1[22], sv1[23]), _fma_b2_22, _fma_c2_23);
                    sv1[22] = _fma_pair_35.x;
                    sv1[23] = _fma_pair_35.y;
                    float2 _fma_pair_36 = fma_f32x2(make_float2(sv1[24], sv1[25]), _fma_b2_22, _fma_c2_23);
                    sv1[24] = _fma_pair_36.x;
                    sv1[25] = _fma_pair_36.y;
                    float2 _fma_pair_37 = fma_f32x2(make_float2(sv1[26], sv1[27]), _fma_b2_22, _fma_c2_23);
                    sv1[26] = _fma_pair_37.x;
                    sv1[27] = _fma_pair_37.y;
                    float2 _fma_pair_38 = fma_f32x2(make_float2(sv1[28], sv1[29]), _fma_b2_22, _fma_c2_23);
                    sv1[28] = _fma_pair_38.x;
                    sv1[29] = _fma_pair_38.y;
                    float2 _fma_pair_39 = fma_f32x2(make_float2(sv1[30], sv1[31]), _fma_b2_22, _fma_c2_23);
                    sv1[30] = _fma_pair_39.x;
                    sv1[31] = _fma_pair_39.y;
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        sv0[_le] = approx_exp2(sv0[_le]);
                    }
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        sv1[_le] = approx_exp2(sv1[_le]);
                    }
                    float2 _reg_reduce_sum2_40 = make_float2(0.0f, 0.0f);
                    _reg_reduce_sum2_40 = add_f32x2(_reg_reduce_sum2_40, make_float2(sv0[0], sv0[1]));
                    _reg_reduce_sum2_40 = add_f32x2(_reg_reduce_sum2_40, make_float2(sv0[2], sv0[3]));
                    _reg_reduce_sum2_40 = add_f32x2(_reg_reduce_sum2_40, make_float2(sv0[4], sv0[5]));
                    _reg_reduce_sum2_40 = add_f32x2(_reg_reduce_sum2_40, make_float2(sv0[6], sv0[7]));
                    _reg_reduce_sum2_40 = add_f32x2(_reg_reduce_sum2_40, make_float2(sv0[8], sv0[9]));
                    _reg_reduce_sum2_40 = add_f32x2(_reg_reduce_sum2_40, make_float2(sv0[10], sv0[11]));
                    _reg_reduce_sum2_40 = add_f32x2(_reg_reduce_sum2_40, make_float2(sv0[12], sv0[13]));
                    _reg_reduce_sum2_40 = add_f32x2(_reg_reduce_sum2_40, make_float2(sv0[14], sv0[15]));
                    _reg_reduce_sum2_40 = add_f32x2(_reg_reduce_sum2_40, make_float2(sv0[16], sv0[17]));
                    _reg_reduce_sum2_40 = add_f32x2(_reg_reduce_sum2_40, make_float2(sv0[18], sv0[19]));
                    _reg_reduce_sum2_40 = add_f32x2(_reg_reduce_sum2_40, make_float2(sv0[20], sv0[21]));
                    _reg_reduce_sum2_40 = add_f32x2(_reg_reduce_sum2_40, make_float2(sv0[22], sv0[23]));
                    _reg_reduce_sum2_40 = add_f32x2(_reg_reduce_sum2_40, make_float2(sv0[24], sv0[25]));
                    _reg_reduce_sum2_40 = add_f32x2(_reg_reduce_sum2_40, make_float2(sv0[26], sv0[27]));
                    _reg_reduce_sum2_40 = add_f32x2(_reg_reduce_sum2_40, make_float2(sv0[28], sv0[29]));
                    _reg_reduce_sum2_40 = add_f32x2(_reg_reduce_sum2_40, make_float2(sv0[30], sv0[31]));
                    float sv0_sum = _reg_reduce_sum2_40.x + _reg_reduce_sum2_40.y;
                    float2 _reg_reduce_sum2_41 = make_float2(0.0f, 0.0f);
                    _reg_reduce_sum2_41 = add_f32x2(_reg_reduce_sum2_41, make_float2(sv1[0], sv1[1]));
                    _reg_reduce_sum2_41 = add_f32x2(_reg_reduce_sum2_41, make_float2(sv1[2], sv1[3]));
                    _reg_reduce_sum2_41 = add_f32x2(_reg_reduce_sum2_41, make_float2(sv1[4], sv1[5]));
                    _reg_reduce_sum2_41 = add_f32x2(_reg_reduce_sum2_41, make_float2(sv1[6], sv1[7]));
                    _reg_reduce_sum2_41 = add_f32x2(_reg_reduce_sum2_41, make_float2(sv1[8], sv1[9]));
                    _reg_reduce_sum2_41 = add_f32x2(_reg_reduce_sum2_41, make_float2(sv1[10], sv1[11]));
                    _reg_reduce_sum2_41 = add_f32x2(_reg_reduce_sum2_41, make_float2(sv1[12], sv1[13]));
                    _reg_reduce_sum2_41 = add_f32x2(_reg_reduce_sum2_41, make_float2(sv1[14], sv1[15]));
                    _reg_reduce_sum2_41 = add_f32x2(_reg_reduce_sum2_41, make_float2(sv1[16], sv1[17]));
                    _reg_reduce_sum2_41 = add_f32x2(_reg_reduce_sum2_41, make_float2(sv1[18], sv1[19]));
                    _reg_reduce_sum2_41 = add_f32x2(_reg_reduce_sum2_41, make_float2(sv1[20], sv1[21]));
                    _reg_reduce_sum2_41 = add_f32x2(_reg_reduce_sum2_41, make_float2(sv1[22], sv1[23]));
                    _reg_reduce_sum2_41 = add_f32x2(_reg_reduce_sum2_41, make_float2(sv1[24], sv1[25]));
                    _reg_reduce_sum2_41 = add_f32x2(_reg_reduce_sum2_41, make_float2(sv1[26], sv1[27]));
                    _reg_reduce_sum2_41 = add_f32x2(_reg_reduce_sum2_41, make_float2(sv1[28], sv1[29]));
                    _reg_reduce_sum2_41 = add_f32x2(_reg_reduce_sum2_41, make_float2(sv1[30], sv1[31]));
                    float sv1_sum = _reg_reduce_sum2_41.x + _reg_reduce_sum2_41.y;
                    float block_sum = sv0_sum + sv1_sum;
                    row_sum = row_sum * alpha + block_sum;
                    mbarrier_wait(p_empty_addr + (p_stage_s) * 8, p_phase_s);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    int p_base = smem_p_addr + (unsigned int)(p_stage_s * 8192);
                    uint32_t _fp8_0[8];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv0[0]), "f"(sv0[1]),
                                               "f"(sv0[2]), "f"(sv0[3]));
                        _fp8_0[0] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv0[4]), "f"(sv0[5]),
                                               "f"(sv0[6]), "f"(sv0[7]));
                        _fp8_0[1] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv0[8]), "f"(sv0[9]),
                                               "f"(sv0[10]), "f"(sv0[11]));
                        _fp8_0[2] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv0[12]), "f"(sv0[13]),
                                               "f"(sv0[14]), "f"(sv0[15]));
                        _fp8_0[3] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv0[16]), "f"(sv0[17]),
                                               "f"(sv0[18]), "f"(sv0[19]));
                        _fp8_0[4] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv0[20]), "f"(sv0[21]),
                                               "f"(sv0[22]), "f"(sv0[23]));
                        _fp8_0[5] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv0[24]), "f"(sv0[25]),
                                               "f"(sv0[26]), "f"(sv0[27]));
                        _fp8_0[6] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv0[28]), "f"(sv0[29]),
                                               "f"(sv0[30]), "f"(sv0[31]));
                        _fp8_0[7] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p_base + (row_c * 128 + n_half * 32 ^ (row_c * 128 + n_half * 32 >> 7 & 7) << 4))), "r"(_fp8_0[0]), "r"(_fp8_0[1]), "r"(_fp8_0[2]), "r"(_fp8_0[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p_base + (row_c * 128 + (n_half * 32 + 16) ^ (row_c * 128 + (n_half * 32 + 16) >> 7 & 7) << 4))), "r"(_fp8_0[4]), "r"(_fp8_0[5]), "r"(_fp8_0[6]), "r"(_fp8_0[7]) : "memory");
                    uint32_t _fp8_1[8];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv1[0]), "f"(sv1[1]),
                                               "f"(sv1[2]), "f"(sv1[3]));
                        _fp8_1[0] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv1[4]), "f"(sv1[5]),
                                               "f"(sv1[6]), "f"(sv1[7]));
                        _fp8_1[1] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv1[8]), "f"(sv1[9]),
                                               "f"(sv1[10]), "f"(sv1[11]));
                        _fp8_1[2] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv1[12]), "f"(sv1[13]),
                                               "f"(sv1[14]), "f"(sv1[15]));
                        _fp8_1[3] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv1[16]), "f"(sv1[17]),
                                               "f"(sv1[18]), "f"(sv1[19]));
                        _fp8_1[4] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv1[20]), "f"(sv1[21]),
                                               "f"(sv1[22]), "f"(sv1[23]));
                        _fp8_1[5] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv1[24]), "f"(sv1[25]),
                                               "f"(sv1[26]), "f"(sv1[27]));
                        _fp8_1[6] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv1[28]), "f"(sv1[29]),
                                               "f"(sv1[30]), "f"(sv1[31]));
                        _fp8_1[7] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p_base + (row_c * 128 + (64 + n_half * 32) ^ (row_c * 128 + (64 + n_half * 32) >> 7 & 7) << 4))), "r"(_fp8_1[0]), "r"(_fp8_1[1]), "r"(_fp8_1[2]), "r"(_fp8_1[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p_base + (row_c * 128 + (64 + n_half * 32 + 16) ^ (row_c * 128 + (64 + n_half * 32 + 16) >> 7 & 7) << 4))), "r"(_fp8_1[4]), "r"(_fp8_1[5]), "r"(_fp8_1[6]), "r"(_fp8_1[7]) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(p_full_addr + p_stage_s * 8), "r"(0) : "memory");
                    p_stage_s += 1;
                    if (p_stage_s == 2) { p_stage_s = 0; p_phase_s ^= 1; }
                }
                mbarrier_wait(o_full_addr, _phase_o_full_0);
                _phase_o_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                smem_xchg_sum[xchg_idx] = row_sum;
                asm volatile("barrier.sync %0, 64;" :: "r"(xchg_bar) : "memory");
                row_sum = row_sum + smem_xchg_sum[xchg_idx ^ 64];
                asm volatile("barrier.sync %0, 64;" :: "r"(xchg_bar) : "memory");
                int direct_out = num_split == 1;
                float out_scale = ((direct_out != 0) ? bmm2_scale : 1.0f);
                float _rcp_0 = approx_rcp(row_sum);
                float inv_sum = ((row_sum > 0.0f) ? _rcp_0 * (out_scale * 8.0f) : 0.0f);
                int o_elem_base = out_row * num_split + split_idx;
                int warp_row0_local = rank_s * 64 + quadrant % 2 * 32;
                int warp_bulk = rows_valid >= warp_row0_local + 32 && empty == 0;
                int row_g0 = row_base_global + warp_row0_local;
                int warp_tiles_base = smem_v30_addr + (unsigned int)(quadrant * 16384);
                float o_ra[32];
                float o_rb[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(o_ra[0]), "=f"(o_ra[1]), "=f"(o_ra[2]), "=f"(o_ra[3]), "=f"(o_ra[4]), "=f"(o_ra[5]), "=f"(o_ra[6]), "=f"(o_ra[7]), "=f"(o_ra[8]), "=f"(o_ra[9]), "=f"(o_ra[10]), "=f"(o_ra[11]), "=f"(o_ra[12]), "=f"(o_ra[13]), "=f"(o_ra[14]), "=f"(o_ra[15]), "=f"(o_ra[16]), "=f"(o_ra[17]), "=f"(o_ra[18]), "=f"(o_ra[19]), "=f"(o_ra[20]), "=f"(o_ra[21]), "=f"(o_ra[22]), "=f"(o_ra[23]), "=f"(o_ra[24]), "=f"(o_ra[25]), "=f"(o_ra[26]), "=f"(o_ra[27]), "=f"(o_ra[28]), "=f"(o_ra[29]), "=f"(o_ra[30]), "=f"(o_ra[31])
                    : "r"(taddr + (unsigned int)lane_base + 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_rb[0]), "=f"(o_rb[1]), "=f"(o_rb[2]), "=f"(o_rb[3]), "=f"(o_rb[4]), "=f"(o_rb[5]), "=f"(o_rb[6]), "=f"(o_rb[7]), "=f"(o_rb[8]), "=f"(o_rb[9]), "=f"(o_rb[10]), "=f"(o_rb[11]), "=f"(o_rb[12]), "=f"(o_rb[13]), "=f"(o_rb[14]), "=f"(o_rb[15]), "=f"(o_rb[16]), "=f"(o_rb[17]), "=f"(o_rb[18]), "=f"(o_rb[19]), "=f"(o_rb[20]), "=f"(o_rb[21]), "=f"(o_rb[22]), "=f"(o_rb[23]), "=f"(o_rb[24]), "=f"(o_rb[25]), "=f"(o_rb[26]), "=f"(o_rb[27]), "=f"(o_rb[28]), "=f"(o_rb[29]), "=f"(o_rb[30]), "=f"(o_rb[31])
                        : "r"(taddr + (unsigned int)lane_base + 64 + 32));
                }
                #if __CUDA_ARCH__ >= 1000
                const float2 _scale2_42 = {inv_sum, inv_sum};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(o_ra)[_ls], _scale2_42);
                #else
                #pragma unroll
                for (int _ls = 0; _ls < 32; _ls++) {
                    o_ra[_ls] = o_ra[_ls] * inv_sum;
                }
                #endif
                if (warp_bulk != 0) {
                    uint32_t o_ra_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_ra[_lp*2 + 0], o_ra[_lp*2+1 + 0]));
                        o_ra_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    int warp_tile = warp_tiles_base;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile + (lane * 128 ^ (lane * 128 >> 7 & 7) << 4))), "r"(o_ra_bf16[0]), "r"(o_ra_bf16[1]), "r"(o_ra_bf16[2]), "r"(o_ra_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile + (lane * 128 + 16 ^ (lane * 128 + 16 >> 7 & 7) << 4))), "r"(o_ra_bf16[4]), "r"(o_ra_bf16[5]), "r"(o_ra_bf16[6]), "r"(o_ra_bf16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile + (lane * 128 + 32 ^ (lane * 128 + 32 >> 7 & 7) << 4))), "r"(o_ra_bf16[8]), "r"(o_ra_bf16[9]), "r"(o_ra_bf16[10]), "r"(o_ra_bf16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile + (lane * 128 + 48 ^ (lane * 128 + 48 >> 7 & 7) << 4))), "r"(o_ra_bf16[12]), "r"(o_ra_bf16[13]), "r"(o_ra_bf16[14]), "r"(o_ra_bf16[15]) : "memory");
                } else if (row_valid != 0) {
                    {
                        __nv_bfloat162 _pk[8];
                        _pk[0] = __floats2bfloat162_rn(o_ra[0 + 0], o_ra[0 + 1]);
                        _pk[1] = __floats2bfloat162_rn(o_ra[0 + 2], o_ra[0 + 3]);
                        _pk[2] = __floats2bfloat162_rn(o_ra[0 + 4], o_ra[0 + 5]);
                        _pk[3] = __floats2bfloat162_rn(o_ra[0 + 6], o_ra[0 + 7]);
                        _pk[4] = __floats2bfloat162_rn(o_ra[0 + 8], o_ra[0 + 9]);
                        _pk[5] = __floats2bfloat162_rn(o_ra[0 + 10], o_ra[0 + 11]);
                        _pk[6] = __floats2bfloat162_rn(o_ra[0 + 12], o_ra[0 + 13]);
                        _pk[7] = __floats2bfloat162_rn(o_ra[0 + 14], o_ra[0 + 15]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + (o_elem_base * 512 + n_half * 128)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + (o_elem_base * 512 + n_half * 128)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                    }
                    {
                        __nv_bfloat162 _pk[8];
                        _pk[0] = __floats2bfloat162_rn(o_ra[16 + 0], o_ra[16 + 1]);
                        _pk[1] = __floats2bfloat162_rn(o_ra[16 + 2], o_ra[16 + 3]);
                        _pk[2] = __floats2bfloat162_rn(o_ra[16 + 4], o_ra[16 + 5]);
                        _pk[3] = __floats2bfloat162_rn(o_ra[16 + 6], o_ra[16 + 7]);
                        _pk[4] = __floats2bfloat162_rn(o_ra[16 + 8], o_ra[16 + 9]);
                        _pk[5] = __floats2bfloat162_rn(o_ra[16 + 10], o_ra[16 + 11]);
                        _pk[6] = __floats2bfloat162_rn(o_ra[16 + 12], o_ra[16 + 13]);
                        _pk[7] = __floats2bfloat162_rn(o_ra[16 + 14], o_ra[16 + 15]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + (o_elem_base * 512 + n_half * 128 + 16)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + (o_elem_base * 512 + n_half * 128 + 16)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_ra[0]), "=f"(o_ra[1]), "=f"(o_ra[2]), "=f"(o_ra[3]), "=f"(o_ra[4]), "=f"(o_ra[5]), "=f"(o_ra[6]), "=f"(o_ra[7]), "=f"(o_ra[8]), "=f"(o_ra[9]), "=f"(o_ra[10]), "=f"(o_ra[11]), "=f"(o_ra[12]), "=f"(o_ra[13]), "=f"(o_ra[14]), "=f"(o_ra[15]), "=f"(o_ra[16]), "=f"(o_ra[17]), "=f"(o_ra[18]), "=f"(o_ra[19]), "=f"(o_ra[20]), "=f"(o_ra[21]), "=f"(o_ra[22]), "=f"(o_ra[23]), "=f"(o_ra[24]), "=f"(o_ra[25]), "=f"(o_ra[26]), "=f"(o_ra[27]), "=f"(o_ra[28]), "=f"(o_ra[29]), "=f"(o_ra[30]), "=f"(o_ra[31])
                        : "r"(taddr + (unsigned int)lane_base + 64 + 64));
                }
                #if __CUDA_ARCH__ >= 1000
                const float2 _scale2_43 = {inv_sum, inv_sum};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(o_rb)[_ls], _scale2_43);
                #else
                #pragma unroll
                for (int _ls = 0; _ls < 32; _ls++) {
                    o_rb[_ls] = o_rb[_ls] * inv_sum;
                }
                #endif
                if (warp_bulk != 0) {
                    uint32_t o_rb_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_rb[_lp*2 + 0], o_rb[_lp*2+1 + 0]));
                        o_rb_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    int warp_tile_1 = warp_tiles_base;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_1 + (lane * 128 + 64 ^ (lane * 128 + 64 >> 7 & 7) << 4))), "r"(o_rb_bf16[0]), "r"(o_rb_bf16[1]), "r"(o_rb_bf16[2]), "r"(o_rb_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_1 + (lane * 128 + 80 ^ (lane * 128 + 80 >> 7 & 7) << 4))), "r"(o_rb_bf16[4]), "r"(o_rb_bf16[5]), "r"(o_rb_bf16[6]), "r"(o_rb_bf16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_1 + (lane * 128 + 96 ^ (lane * 128 + 96 >> 7 & 7) << 4))), "r"(o_rb_bf16[8]), "r"(o_rb_bf16[9]), "r"(o_rb_bf16[10]), "r"(o_rb_bf16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_1 + (lane * 128 + 112 ^ (lane * 128 + 112 >> 7 & 7) << 4))), "r"(o_rb_bf16[12]), "r"(o_rb_bf16[13]), "r"(o_rb_bf16[14]), "r"(o_rb_bf16[15]) : "memory");
                    {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        __syncwarp();
                        if (elect_sync()) {
                            tma_store_3d((&tmap_po), n_half * 128 + 32 - 32, split_idx, row_g0, warp_tile_1);
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                } else if (row_valid != 0) {
                    {
                        __nv_bfloat162 _pk[8];
                        _pk[0] = __floats2bfloat162_rn(o_rb[0 + 0], o_rb[0 + 1]);
                        _pk[1] = __floats2bfloat162_rn(o_rb[0 + 2], o_rb[0 + 3]);
                        _pk[2] = __floats2bfloat162_rn(o_rb[0 + 4], o_rb[0 + 5]);
                        _pk[3] = __floats2bfloat162_rn(o_rb[0 + 6], o_rb[0 + 7]);
                        _pk[4] = __floats2bfloat162_rn(o_rb[0 + 8], o_rb[0 + 9]);
                        _pk[5] = __floats2bfloat162_rn(o_rb[0 + 10], o_rb[0 + 11]);
                        _pk[6] = __floats2bfloat162_rn(o_rb[0 + 12], o_rb[0 + 13]);
                        _pk[7] = __floats2bfloat162_rn(o_rb[0 + 14], o_rb[0 + 15]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + (o_elem_base * 512 + n_half * 128 + 32)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + (o_elem_base * 512 + n_half * 128 + 32)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                    }
                    {
                        __nv_bfloat162 _pk[8];
                        _pk[0] = __floats2bfloat162_rn(o_rb[16 + 0], o_rb[16 + 1]);
                        _pk[1] = __floats2bfloat162_rn(o_rb[16 + 2], o_rb[16 + 3]);
                        _pk[2] = __floats2bfloat162_rn(o_rb[16 + 4], o_rb[16 + 5]);
                        _pk[3] = __floats2bfloat162_rn(o_rb[16 + 6], o_rb[16 + 7]);
                        _pk[4] = __floats2bfloat162_rn(o_rb[16 + 8], o_rb[16 + 9]);
                        _pk[5] = __floats2bfloat162_rn(o_rb[16 + 10], o_rb[16 + 11]);
                        _pk[6] = __floats2bfloat162_rn(o_rb[16 + 12], o_rb[16 + 13]);
                        _pk[7] = __floats2bfloat162_rn(o_rb[16 + 14], o_rb[16 + 15]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + (o_elem_base * 512 + n_half * 128 + 32 + 16)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + (o_elem_base * 512 + n_half * 128 + 32 + 16)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_rb[0]), "=f"(o_rb[1]), "=f"(o_rb[2]), "=f"(o_rb[3]), "=f"(o_rb[4]), "=f"(o_rb[5]), "=f"(o_rb[6]), "=f"(o_rb[7]), "=f"(o_rb[8]), "=f"(o_rb[9]), "=f"(o_rb[10]), "=f"(o_rb[11]), "=f"(o_rb[12]), "=f"(o_rb[13]), "=f"(o_rb[14]), "=f"(o_rb[15]), "=f"(o_rb[16]), "=f"(o_rb[17]), "=f"(o_rb[18]), "=f"(o_rb[19]), "=f"(o_rb[20]), "=f"(o_rb[21]), "=f"(o_rb[22]), "=f"(o_rb[23]), "=f"(o_rb[24]), "=f"(o_rb[25]), "=f"(o_rb[26]), "=f"(o_rb[27]), "=f"(o_rb[28]), "=f"(o_rb[29]), "=f"(o_rb[30]), "=f"(o_rb[31])
                        : "r"(taddr + (unsigned int)lane_base + 64 + 96));
                }
                #if __CUDA_ARCH__ >= 1000
                const float2 _scale2_44 = {inv_sum, inv_sum};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(o_ra)[_ls], _scale2_44);
                #else
                #pragma unroll
                for (int _ls = 0; _ls < 32; _ls++) {
                    o_ra[_ls] = o_ra[_ls] * inv_sum;
                }
                #endif
                if (warp_bulk != 0) {
                    uint32_t o_ra_bf16_1[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_ra[_lp*2 + 0], o_ra[_lp*2+1 + 0]));
                        o_ra_bf16_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    int warp_tile_2 = warp_tiles_base + 4096;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_2 + (lane * 128 ^ (lane * 128 >> 7 & 7) << 4))), "r"(o_ra_bf16_1[0]), "r"(o_ra_bf16_1[1]), "r"(o_ra_bf16_1[2]), "r"(o_ra_bf16_1[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_2 + (lane * 128 + 16 ^ (lane * 128 + 16 >> 7 & 7) << 4))), "r"(o_ra_bf16_1[4]), "r"(o_ra_bf16_1[5]), "r"(o_ra_bf16_1[6]), "r"(o_ra_bf16_1[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_2 + (lane * 128 + 32 ^ (lane * 128 + 32 >> 7 & 7) << 4))), "r"(o_ra_bf16_1[8]), "r"(o_ra_bf16_1[9]), "r"(o_ra_bf16_1[10]), "r"(o_ra_bf16_1[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_2 + (lane * 128 + 48 ^ (lane * 128 + 48 >> 7 & 7) << 4))), "r"(o_ra_bf16_1[12]), "r"(o_ra_bf16_1[13]), "r"(o_ra_bf16_1[14]), "r"(o_ra_bf16_1[15]) : "memory");
                } else if (row_valid != 0) {
                    {
                        __nv_bfloat162 _pk[8];
                        _pk[0] = __floats2bfloat162_rn(o_ra[0 + 0], o_ra[0 + 1]);
                        _pk[1] = __floats2bfloat162_rn(o_ra[0 + 2], o_ra[0 + 3]);
                        _pk[2] = __floats2bfloat162_rn(o_ra[0 + 4], o_ra[0 + 5]);
                        _pk[3] = __floats2bfloat162_rn(o_ra[0 + 6], o_ra[0 + 7]);
                        _pk[4] = __floats2bfloat162_rn(o_ra[0 + 8], o_ra[0 + 9]);
                        _pk[5] = __floats2bfloat162_rn(o_ra[0 + 10], o_ra[0 + 11]);
                        _pk[6] = __floats2bfloat162_rn(o_ra[0 + 12], o_ra[0 + 13]);
                        _pk[7] = __floats2bfloat162_rn(o_ra[0 + 14], o_ra[0 + 15]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + (o_elem_base * 512 + n_half * 128 + 64)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + (o_elem_base * 512 + n_half * 128 + 64)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                    }
                    {
                        __nv_bfloat162 _pk[8];
                        _pk[0] = __floats2bfloat162_rn(o_ra[16 + 0], o_ra[16 + 1]);
                        _pk[1] = __floats2bfloat162_rn(o_ra[16 + 2], o_ra[16 + 3]);
                        _pk[2] = __floats2bfloat162_rn(o_ra[16 + 4], o_ra[16 + 5]);
                        _pk[3] = __floats2bfloat162_rn(o_ra[16 + 6], o_ra[16 + 7]);
                        _pk[4] = __floats2bfloat162_rn(o_ra[16 + 8], o_ra[16 + 9]);
                        _pk[5] = __floats2bfloat162_rn(o_ra[16 + 10], o_ra[16 + 11]);
                        _pk[6] = __floats2bfloat162_rn(o_ra[16 + 12], o_ra[16 + 13]);
                        _pk[7] = __floats2bfloat162_rn(o_ra[16 + 14], o_ra[16 + 15]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + (o_elem_base * 512 + n_half * 128 + 64 + 16)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + (o_elem_base * 512 + n_half * 128 + 64 + 16)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_ra[0]), "=f"(o_ra[1]), "=f"(o_ra[2]), "=f"(o_ra[3]), "=f"(o_ra[4]), "=f"(o_ra[5]), "=f"(o_ra[6]), "=f"(o_ra[7]), "=f"(o_ra[8]), "=f"(o_ra[9]), "=f"(o_ra[10]), "=f"(o_ra[11]), "=f"(o_ra[12]), "=f"(o_ra[13]), "=f"(o_ra[14]), "=f"(o_ra[15]), "=f"(o_ra[16]), "=f"(o_ra[17]), "=f"(o_ra[18]), "=f"(o_ra[19]), "=f"(o_ra[20]), "=f"(o_ra[21]), "=f"(o_ra[22]), "=f"(o_ra[23]), "=f"(o_ra[24]), "=f"(o_ra[25]), "=f"(o_ra[26]), "=f"(o_ra[27]), "=f"(o_ra[28]), "=f"(o_ra[29]), "=f"(o_ra[30]), "=f"(o_ra[31])
                        : "r"(taddr + (unsigned int)lane_base + 64 + 128));
                }
                #if __CUDA_ARCH__ >= 1000
                const float2 _scale2_45 = {inv_sum, inv_sum};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(o_rb)[_ls], _scale2_45);
                #else
                #pragma unroll
                for (int _ls = 0; _ls < 32; _ls++) {
                    o_rb[_ls] = o_rb[_ls] * inv_sum;
                }
                #endif
                if (warp_bulk != 0) {
                    uint32_t o_rb_bf16_1[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_rb[_lp*2 + 0], o_rb[_lp*2+1 + 0]));
                        o_rb_bf16_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    int warp_tile_3 = warp_tiles_base + 4096;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_3 + (lane * 128 + 64 ^ (lane * 128 + 64 >> 7 & 7) << 4))), "r"(o_rb_bf16_1[0]), "r"(o_rb_bf16_1[1]), "r"(o_rb_bf16_1[2]), "r"(o_rb_bf16_1[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_3 + (lane * 128 + 80 ^ (lane * 128 + 80 >> 7 & 7) << 4))), "r"(o_rb_bf16_1[4]), "r"(o_rb_bf16_1[5]), "r"(o_rb_bf16_1[6]), "r"(o_rb_bf16_1[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_3 + (lane * 128 + 96 ^ (lane * 128 + 96 >> 7 & 7) << 4))), "r"(o_rb_bf16_1[8]), "r"(o_rb_bf16_1[9]), "r"(o_rb_bf16_1[10]), "r"(o_rb_bf16_1[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_3 + (lane * 128 + 112 ^ (lane * 128 + 112 >> 7 & 7) << 4))), "r"(o_rb_bf16_1[12]), "r"(o_rb_bf16_1[13]), "r"(o_rb_bf16_1[14]), "r"(o_rb_bf16_1[15]) : "memory");
                    {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        __syncwarp();
                        if (elect_sync()) {
                            tma_store_3d((&tmap_po), n_half * 128 + 96 - 32, split_idx, row_g0, warp_tile_3);
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                } else if (row_valid != 0) {
                    {
                        __nv_bfloat162 _pk[8];
                        _pk[0] = __floats2bfloat162_rn(o_rb[0 + 0], o_rb[0 + 1]);
                        _pk[1] = __floats2bfloat162_rn(o_rb[0 + 2], o_rb[0 + 3]);
                        _pk[2] = __floats2bfloat162_rn(o_rb[0 + 4], o_rb[0 + 5]);
                        _pk[3] = __floats2bfloat162_rn(o_rb[0 + 6], o_rb[0 + 7]);
                        _pk[4] = __floats2bfloat162_rn(o_rb[0 + 8], o_rb[0 + 9]);
                        _pk[5] = __floats2bfloat162_rn(o_rb[0 + 10], o_rb[0 + 11]);
                        _pk[6] = __floats2bfloat162_rn(o_rb[0 + 12], o_rb[0 + 13]);
                        _pk[7] = __floats2bfloat162_rn(o_rb[0 + 14], o_rb[0 + 15]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + (o_elem_base * 512 + n_half * 128 + 96)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + (o_elem_base * 512 + n_half * 128 + 96)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                    }
                    {
                        __nv_bfloat162 _pk[8];
                        _pk[0] = __floats2bfloat162_rn(o_rb[16 + 0], o_rb[16 + 1]);
                        _pk[1] = __floats2bfloat162_rn(o_rb[16 + 2], o_rb[16 + 3]);
                        _pk[2] = __floats2bfloat162_rn(o_rb[16 + 4], o_rb[16 + 5]);
                        _pk[3] = __floats2bfloat162_rn(o_rb[16 + 6], o_rb[16 + 7]);
                        _pk[4] = __floats2bfloat162_rn(o_rb[16 + 8], o_rb[16 + 9]);
                        _pk[5] = __floats2bfloat162_rn(o_rb[16 + 10], o_rb[16 + 11]);
                        _pk[6] = __floats2bfloat162_rn(o_rb[16 + 12], o_rb[16 + 13]);
                        _pk[7] = __floats2bfloat162_rn(o_rb[16 + 14], o_rb[16 + 15]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + (o_elem_base * 512 + n_half * 128 + 96 + 16)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + (o_elem_base * 512 + n_half * 128 + 96 + 16)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_rb[0]), "=f"(o_rb[1]), "=f"(o_rb[2]), "=f"(o_rb[3]), "=f"(o_rb[4]), "=f"(o_rb[5]), "=f"(o_rb[6]), "=f"(o_rb[7]), "=f"(o_rb[8]), "=f"(o_rb[9]), "=f"(o_rb[10]), "=f"(o_rb[11]), "=f"(o_rb[12]), "=f"(o_rb[13]), "=f"(o_rb[14]), "=f"(o_rb[15]), "=f"(o_rb[16]), "=f"(o_rb[17]), "=f"(o_rb[18]), "=f"(o_rb[19]), "=f"(o_rb[20]), "=f"(o_rb[21]), "=f"(o_rb[22]), "=f"(o_rb[23]), "=f"(o_rb[24]), "=f"(o_rb[25]), "=f"(o_rb[26]), "=f"(o_rb[27]), "=f"(o_rb[28]), "=f"(o_rb[29]), "=f"(o_rb[30]), "=f"(o_rb[31])
                        : "r"(taddr + (unsigned int)lane_base + 64 + 128 + 32));
                }
                #if __CUDA_ARCH__ >= 1000
                const float2 _scale2_46 = {inv_sum, inv_sum};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(o_ra)[_ls], _scale2_46);
                #else
                #pragma unroll
                for (int _ls = 0; _ls < 32; _ls++) {
                    o_ra[_ls] = o_ra[_ls] * inv_sum;
                }
                #endif
                if (warp_bulk != 0) {
                    uint32_t o_ra_bf16_2[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_ra[_lp*2 + 0], o_ra[_lp*2+1 + 0]));
                        o_ra_bf16_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    int warp_tile_4 = warp_tiles_base + 8192;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_4 + (lane * 128 ^ (lane * 128 >> 7 & 7) << 4))), "r"(o_ra_bf16_2[0]), "r"(o_ra_bf16_2[1]), "r"(o_ra_bf16_2[2]), "r"(o_ra_bf16_2[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_4 + (lane * 128 + 16 ^ (lane * 128 + 16 >> 7 & 7) << 4))), "r"(o_ra_bf16_2[4]), "r"(o_ra_bf16_2[5]), "r"(o_ra_bf16_2[6]), "r"(o_ra_bf16_2[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_4 + (lane * 128 + 32 ^ (lane * 128 + 32 >> 7 & 7) << 4))), "r"(o_ra_bf16_2[8]), "r"(o_ra_bf16_2[9]), "r"(o_ra_bf16_2[10]), "r"(o_ra_bf16_2[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_4 + (lane * 128 + 48 ^ (lane * 128 + 48 >> 7 & 7) << 4))), "r"(o_ra_bf16_2[12]), "r"(o_ra_bf16_2[13]), "r"(o_ra_bf16_2[14]), "r"(o_ra_bf16_2[15]) : "memory");
                } else if (row_valid != 0) {
                    {
                        __nv_bfloat162 _pk[8];
                        _pk[0] = __floats2bfloat162_rn(o_ra[0 + 0], o_ra[0 + 1]);
                        _pk[1] = __floats2bfloat162_rn(o_ra[0 + 2], o_ra[0 + 3]);
                        _pk[2] = __floats2bfloat162_rn(o_ra[0 + 4], o_ra[0 + 5]);
                        _pk[3] = __floats2bfloat162_rn(o_ra[0 + 6], o_ra[0 + 7]);
                        _pk[4] = __floats2bfloat162_rn(o_ra[0 + 8], o_ra[0 + 9]);
                        _pk[5] = __floats2bfloat162_rn(o_ra[0 + 10], o_ra[0 + 11]);
                        _pk[6] = __floats2bfloat162_rn(o_ra[0 + 12], o_ra[0 + 13]);
                        _pk[7] = __floats2bfloat162_rn(o_ra[0 + 14], o_ra[0 + 15]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + (o_elem_base * 512 + n_half * 128 + 256)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + (o_elem_base * 512 + n_half * 128 + 256)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                    }
                    {
                        __nv_bfloat162 _pk[8];
                        _pk[0] = __floats2bfloat162_rn(o_ra[16 + 0], o_ra[16 + 1]);
                        _pk[1] = __floats2bfloat162_rn(o_ra[16 + 2], o_ra[16 + 3]);
                        _pk[2] = __floats2bfloat162_rn(o_ra[16 + 4], o_ra[16 + 5]);
                        _pk[3] = __floats2bfloat162_rn(o_ra[16 + 6], o_ra[16 + 7]);
                        _pk[4] = __floats2bfloat162_rn(o_ra[16 + 8], o_ra[16 + 9]);
                        _pk[5] = __floats2bfloat162_rn(o_ra[16 + 10], o_ra[16 + 11]);
                        _pk[6] = __floats2bfloat162_rn(o_ra[16 + 12], o_ra[16 + 13]);
                        _pk[7] = __floats2bfloat162_rn(o_ra[16 + 14], o_ra[16 + 15]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + (o_elem_base * 512 + n_half * 128 + 256 + 16)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + (o_elem_base * 512 + n_half * 128 + 256 + 16)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_ra[0]), "=f"(o_ra[1]), "=f"(o_ra[2]), "=f"(o_ra[3]), "=f"(o_ra[4]), "=f"(o_ra[5]), "=f"(o_ra[6]), "=f"(o_ra[7]), "=f"(o_ra[8]), "=f"(o_ra[9]), "=f"(o_ra[10]), "=f"(o_ra[11]), "=f"(o_ra[12]), "=f"(o_ra[13]), "=f"(o_ra[14]), "=f"(o_ra[15]), "=f"(o_ra[16]), "=f"(o_ra[17]), "=f"(o_ra[18]), "=f"(o_ra[19]), "=f"(o_ra[20]), "=f"(o_ra[21]), "=f"(o_ra[22]), "=f"(o_ra[23]), "=f"(o_ra[24]), "=f"(o_ra[25]), "=f"(o_ra[26]), "=f"(o_ra[27]), "=f"(o_ra[28]), "=f"(o_ra[29]), "=f"(o_ra[30]), "=f"(o_ra[31])
                        : "r"(taddr + (unsigned int)lane_base + 64 + 128 + 64));
                }
                #if __CUDA_ARCH__ >= 1000
                const float2 _scale2_47 = {inv_sum, inv_sum};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(o_rb)[_ls], _scale2_47);
                #else
                #pragma unroll
                for (int _ls = 0; _ls < 32; _ls++) {
                    o_rb[_ls] = o_rb[_ls] * inv_sum;
                }
                #endif
                if (warp_bulk != 0) {
                    uint32_t o_rb_bf16_2[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_rb[_lp*2 + 0], o_rb[_lp*2+1 + 0]));
                        o_rb_bf16_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    int warp_tile_5 = warp_tiles_base + 8192;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_5 + (lane * 128 + 64 ^ (lane * 128 + 64 >> 7 & 7) << 4))), "r"(o_rb_bf16_2[0]), "r"(o_rb_bf16_2[1]), "r"(o_rb_bf16_2[2]), "r"(o_rb_bf16_2[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_5 + (lane * 128 + 80 ^ (lane * 128 + 80 >> 7 & 7) << 4))), "r"(o_rb_bf16_2[4]), "r"(o_rb_bf16_2[5]), "r"(o_rb_bf16_2[6]), "r"(o_rb_bf16_2[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_5 + (lane * 128 + 96 ^ (lane * 128 + 96 >> 7 & 7) << 4))), "r"(o_rb_bf16_2[8]), "r"(o_rb_bf16_2[9]), "r"(o_rb_bf16_2[10]), "r"(o_rb_bf16_2[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_5 + (lane * 128 + 112 ^ (lane * 128 + 112 >> 7 & 7) << 4))), "r"(o_rb_bf16_2[12]), "r"(o_rb_bf16_2[13]), "r"(o_rb_bf16_2[14]), "r"(o_rb_bf16_2[15]) : "memory");
                    {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        __syncwarp();
                        if (elect_sync()) {
                            tma_store_3d((&tmap_po), n_half * 128 + 256 + 32 - 32, split_idx, row_g0, warp_tile_5);
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                } else if (row_valid != 0) {
                    {
                        __nv_bfloat162 _pk[8];
                        _pk[0] = __floats2bfloat162_rn(o_rb[0 + 0], o_rb[0 + 1]);
                        _pk[1] = __floats2bfloat162_rn(o_rb[0 + 2], o_rb[0 + 3]);
                        _pk[2] = __floats2bfloat162_rn(o_rb[0 + 4], o_rb[0 + 5]);
                        _pk[3] = __floats2bfloat162_rn(o_rb[0 + 6], o_rb[0 + 7]);
                        _pk[4] = __floats2bfloat162_rn(o_rb[0 + 8], o_rb[0 + 9]);
                        _pk[5] = __floats2bfloat162_rn(o_rb[0 + 10], o_rb[0 + 11]);
                        _pk[6] = __floats2bfloat162_rn(o_rb[0 + 12], o_rb[0 + 13]);
                        _pk[7] = __floats2bfloat162_rn(o_rb[0 + 14], o_rb[0 + 15]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + (o_elem_base * 512 + n_half * 128 + 256 + 32)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + (o_elem_base * 512 + n_half * 128 + 256 + 32)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                    }
                    {
                        __nv_bfloat162 _pk[8];
                        _pk[0] = __floats2bfloat162_rn(o_rb[16 + 0], o_rb[16 + 1]);
                        _pk[1] = __floats2bfloat162_rn(o_rb[16 + 2], o_rb[16 + 3]);
                        _pk[2] = __floats2bfloat162_rn(o_rb[16 + 4], o_rb[16 + 5]);
                        _pk[3] = __floats2bfloat162_rn(o_rb[16 + 6], o_rb[16 + 7]);
                        _pk[4] = __floats2bfloat162_rn(o_rb[16 + 8], o_rb[16 + 9]);
                        _pk[5] = __floats2bfloat162_rn(o_rb[16 + 10], o_rb[16 + 11]);
                        _pk[6] = __floats2bfloat162_rn(o_rb[16 + 12], o_rb[16 + 13]);
                        _pk[7] = __floats2bfloat162_rn(o_rb[16 + 14], o_rb[16 + 15]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + (o_elem_base * 512 + n_half * 128 + 256 + 32 + 16)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + (o_elem_base * 512 + n_half * 128 + 256 + 32 + 16)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_rb[0]), "=f"(o_rb[1]), "=f"(o_rb[2]), "=f"(o_rb[3]), "=f"(o_rb[4]), "=f"(o_rb[5]), "=f"(o_rb[6]), "=f"(o_rb[7]), "=f"(o_rb[8]), "=f"(o_rb[9]), "=f"(o_rb[10]), "=f"(o_rb[11]), "=f"(o_rb[12]), "=f"(o_rb[13]), "=f"(o_rb[14]), "=f"(o_rb[15]), "=f"(o_rb[16]), "=f"(o_rb[17]), "=f"(o_rb[18]), "=f"(o_rb[19]), "=f"(o_rb[20]), "=f"(o_rb[21]), "=f"(o_rb[22]), "=f"(o_rb[23]), "=f"(o_rb[24]), "=f"(o_rb[25]), "=f"(o_rb[26]), "=f"(o_rb[27]), "=f"(o_rb[28]), "=f"(o_rb[29]), "=f"(o_rb[30]), "=f"(o_rb[31])
                        : "r"(taddr + (unsigned int)lane_base + 64 + 128 + 96));
                }
                #if __CUDA_ARCH__ >= 1000
                const float2 _scale2_48 = {inv_sum, inv_sum};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(o_ra)[_ls], _scale2_48);
                #else
                #pragma unroll
                for (int _ls = 0; _ls < 32; _ls++) {
                    o_ra[_ls] = o_ra[_ls] * inv_sum;
                }
                #endif
                if (warp_bulk != 0) {
                    uint32_t o_ra_bf16_3[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_ra[_lp*2 + 0], o_ra[_lp*2+1 + 0]));
                        o_ra_bf16_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    int warp_tile_6 = warp_tiles_base + 12288;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_6 + (lane * 128 ^ (lane * 128 >> 7 & 7) << 4))), "r"(o_ra_bf16_3[0]), "r"(o_ra_bf16_3[1]), "r"(o_ra_bf16_3[2]), "r"(o_ra_bf16_3[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_6 + (lane * 128 + 16 ^ (lane * 128 + 16 >> 7 & 7) << 4))), "r"(o_ra_bf16_3[4]), "r"(o_ra_bf16_3[5]), "r"(o_ra_bf16_3[6]), "r"(o_ra_bf16_3[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_6 + (lane * 128 + 32 ^ (lane * 128 + 32 >> 7 & 7) << 4))), "r"(o_ra_bf16_3[8]), "r"(o_ra_bf16_3[9]), "r"(o_ra_bf16_3[10]), "r"(o_ra_bf16_3[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_6 + (lane * 128 + 48 ^ (lane * 128 + 48 >> 7 & 7) << 4))), "r"(o_ra_bf16_3[12]), "r"(o_ra_bf16_3[13]), "r"(o_ra_bf16_3[14]), "r"(o_ra_bf16_3[15]) : "memory");
                } else if (row_valid != 0) {
                    {
                        __nv_bfloat162 _pk[8];
                        _pk[0] = __floats2bfloat162_rn(o_ra[0 + 0], o_ra[0 + 1]);
                        _pk[1] = __floats2bfloat162_rn(o_ra[0 + 2], o_ra[0 + 3]);
                        _pk[2] = __floats2bfloat162_rn(o_ra[0 + 4], o_ra[0 + 5]);
                        _pk[3] = __floats2bfloat162_rn(o_ra[0 + 6], o_ra[0 + 7]);
                        _pk[4] = __floats2bfloat162_rn(o_ra[0 + 8], o_ra[0 + 9]);
                        _pk[5] = __floats2bfloat162_rn(o_ra[0 + 10], o_ra[0 + 11]);
                        _pk[6] = __floats2bfloat162_rn(o_ra[0 + 12], o_ra[0 + 13]);
                        _pk[7] = __floats2bfloat162_rn(o_ra[0 + 14], o_ra[0 + 15]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + (o_elem_base * 512 + n_half * 128 + 256 + 64)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + (o_elem_base * 512 + n_half * 128 + 256 + 64)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                    }
                    {
                        __nv_bfloat162 _pk[8];
                        _pk[0] = __floats2bfloat162_rn(o_ra[16 + 0], o_ra[16 + 1]);
                        _pk[1] = __floats2bfloat162_rn(o_ra[16 + 2], o_ra[16 + 3]);
                        _pk[2] = __floats2bfloat162_rn(o_ra[16 + 4], o_ra[16 + 5]);
                        _pk[3] = __floats2bfloat162_rn(o_ra[16 + 6], o_ra[16 + 7]);
                        _pk[4] = __floats2bfloat162_rn(o_ra[16 + 8], o_ra[16 + 9]);
                        _pk[5] = __floats2bfloat162_rn(o_ra[16 + 10], o_ra[16 + 11]);
                        _pk[6] = __floats2bfloat162_rn(o_ra[16 + 12], o_ra[16 + 13]);
                        _pk[7] = __floats2bfloat162_rn(o_ra[16 + 14], o_ra[16 + 15]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + (o_elem_base * 512 + n_half * 128 + 256 + 64 + 16)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + (o_elem_base * 512 + n_half * 128 + 256 + 64 + 16)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(o_empty_addr), "r"(0) : "memory");
                }
                #if __CUDA_ARCH__ >= 1000
                const float2 _scale2_49 = {inv_sum, inv_sum};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(o_rb)[_ls], _scale2_49);
                #else
                #pragma unroll
                for (int _ls = 0; _ls < 32; _ls++) {
                    o_rb[_ls] = o_rb[_ls] * inv_sum;
                }
                #endif
                if (warp_bulk != 0) {
                    uint32_t o_rb_bf16_3[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_rb[_lp*2 + 0], o_rb[_lp*2+1 + 0]));
                        o_rb_bf16_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    int warp_tile_7 = warp_tiles_base + 12288;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_7 + (lane * 128 + 64 ^ (lane * 128 + 64 >> 7 & 7) << 4))), "r"(o_rb_bf16_3[0]), "r"(o_rb_bf16_3[1]), "r"(o_rb_bf16_3[2]), "r"(o_rb_bf16_3[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_7 + (lane * 128 + 80 ^ (lane * 128 + 80 >> 7 & 7) << 4))), "r"(o_rb_bf16_3[4]), "r"(o_rb_bf16_3[5]), "r"(o_rb_bf16_3[6]), "r"(o_rb_bf16_3[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_7 + (lane * 128 + 96 ^ (lane * 128 + 96 >> 7 & 7) << 4))), "r"(o_rb_bf16_3[8]), "r"(o_rb_bf16_3[9]), "r"(o_rb_bf16_3[10]), "r"(o_rb_bf16_3[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_7 + (lane * 128 + 112 ^ (lane * 128 + 112 >> 7 & 7) << 4))), "r"(o_rb_bf16_3[12]), "r"(o_rb_bf16_3[13]), "r"(o_rb_bf16_3[14]), "r"(o_rb_bf16_3[15]) : "memory");
                    {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        __syncwarp();
                        if (elect_sync()) {
                            tma_store_3d((&tmap_po), n_half * 128 + 256 + 96 - 32, split_idx, row_g0, warp_tile_7);
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                } else if (row_valid != 0) {
                    {
                        __nv_bfloat162 _pk[8];
                        _pk[0] = __floats2bfloat162_rn(o_rb[0 + 0], o_rb[0 + 1]);
                        _pk[1] = __floats2bfloat162_rn(o_rb[0 + 2], o_rb[0 + 3]);
                        _pk[2] = __floats2bfloat162_rn(o_rb[0 + 4], o_rb[0 + 5]);
                        _pk[3] = __floats2bfloat162_rn(o_rb[0 + 6], o_rb[0 + 7]);
                        _pk[4] = __floats2bfloat162_rn(o_rb[0 + 8], o_rb[0 + 9]);
                        _pk[5] = __floats2bfloat162_rn(o_rb[0 + 10], o_rb[0 + 11]);
                        _pk[6] = __floats2bfloat162_rn(o_rb[0 + 12], o_rb[0 + 13]);
                        _pk[7] = __floats2bfloat162_rn(o_rb[0 + 14], o_rb[0 + 15]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + (o_elem_base * 512 + n_half * 128 + 256 + 96)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + (o_elem_base * 512 + n_half * 128 + 256 + 96)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                    }
                    {
                        __nv_bfloat162 _pk[8];
                        _pk[0] = __floats2bfloat162_rn(o_rb[16 + 0], o_rb[16 + 1]);
                        _pk[1] = __floats2bfloat162_rn(o_rb[16 + 2], o_rb[16 + 3]);
                        _pk[2] = __floats2bfloat162_rn(o_rb[16 + 4], o_rb[16 + 5]);
                        _pk[3] = __floats2bfloat162_rn(o_rb[16 + 6], o_rb[16 + 7]);
                        _pk[4] = __floats2bfloat162_rn(o_rb[16 + 8], o_rb[16 + 9]);
                        _pk[5] = __floats2bfloat162_rn(o_rb[16 + 10], o_rb[16 + 11]);
                        _pk[6] = __floats2bfloat162_rn(o_rb[16 + 12], o_rb[16 + 13]);
                        _pk[7] = __floats2bfloat162_rn(o_rb[16 + 14], o_rb[16 + 15]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + (o_elem_base * 512 + n_half * 128 + 256 + 96 + 16)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + (o_elem_base * 512 + n_half * 128 + 256 + 96 + 16)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                    }
                }
                {
                    if (elect_sync()) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                    }
                    __syncwarp();
                    mbarrier_arrive(epi_done_addr);
                }
                int _min_2 = ((row_is_query) < (1) ? (row_is_query) : (1));
                row_is_query = _min_2;
                if ((row_is_query & (int)(n_half == 0)) != 0) {
                    int stat_off = out_row * num_split + split_idx;
                    if (direct_out != 0) {
                        if (has_lse != 0) {
                            float safe_t = ((row_sum > 0.0f) ? row_sum : 1.0f);
                            float _log2_0;
                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(safe_t));
                            float lse_l2 = row_max + _log2_0 - 6.0f;
                            float lse_v = ((row_sum > 0.0f) ? lse_l2 * 0.6931471805599453f : -CAKE_INF);
                            *(reinterpret_cast<float*>(lse + out_row) + (0)) = lse_v;
                        }
                    } else {
                        *(reinterpret_cast<float*>(partial_max + stat_off) + (0)) = row_max;
                        *(reinterpret_cast<float*>(partial_sum + stat_off) + (0)) = row_sum;
                    }
                }
            }
            if (elect_sync()) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 112;");
        { // correction_main
            int quadrant_c = make_warp_uniform(warp % 4);
            int row_cc = quadrant_c % 2 * 32 + lane;
            int lane_base_c = quadrant_c * 32 << 16;
            int p_stage_c = 0;
            int p_phase_c = 0;
            int raw_stage_c = 0;
            int raw_phase_c = 0;
            int pub_tile_c = 0;
            int cx_1 = blockIdx.x / 2;
            int n_clusters_1 = gridDim.x / 2;
            int bz_1 = blockIdx.z;
            int q_end_bz_1 = cum_seq_lens_q[bz_1 + 1];
            int q_start_bz_1 = cum_seq_lens_q[bz_1];
            int kv_len_bz_1 = seq_lens[bz_1];
            int g_len_bz_1 = kv_len_global[bz_1];
            int rows_bz_1 = (q_end_bz_1 - q_start_bz_1) * num_heads;
            int pieces_1 = (rows_bz_1 + 128 - 1) / 128 * num_split;
            int left_1 = pieces_1 - cx_1;
            int n_pieces_raw_1 = (left_1 + n_clusters_1 - 1) / n_clusters_1;
            int n_pieces_1 = ((left_1 <= 0) ? 0 : n_pieces_raw_1);
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_sf_empty_0 = 1;
            unsigned int _phase_corr_sig_0 = 0;
            unsigned int _phase_pv_done_0 = 0;
            #pragma unroll 1
            for (int pi_c = 0; pi_c < n_pieces_1; pi_c++) {
                int p_c = cx_1 + pi_c * n_clusters_1;
                int b_1 = blockIdx.z;
                int split_idx_1 = p_c % num_split;
                int rows_b0_1 = (q_end_bz_1 - q_start_bz_1) * num_heads;
                int m_tiles_b_1 = (rows_b0_1 + 128 - 1) / 128;
                int m_tile_1 = m_tiles_b_1 - 1 - p_c / num_split;
                int q_start_1 = q_start_bz_1;
                int q_len_b_1 = q_end_bz_1 - q_start_bz_1;
                int kv_len_1 = kv_len_bz_1;
                int g_len_1 = g_len_bz_1;
                int rows_b_1 = q_len_b_1 * num_heads;
                int row0_1 = m_tile_1 * 128;
                int rows_left_1 = rows_b_1 - row0_1;
                int rows_pos_1 = ((rows_left_1 < 0) ? 0 : rows_left_1);
                int rows_valid_1 = ((rows_pos_1 > 128) ? 128 : rows_pos_1);
                int row_base_global_1 = q_start_1 * num_heads + row0_1;
                int last_row_1 = row0_1 + rows_valid_1 - 1;
                int t_last_1 = last_row_1 / num_heads;
                int num_1 = g_len_1 - q_len_b_1 + t_last_1 - cp_rank;
                int vis_raw_2 = num_1 / cp_world + 1;
                int vis_cap_1 = ((vis_raw_2 > kv_len_1) ? kv_len_1 : vis_raw_2);
                int vis_out_1 = ((num_1 < 0) ? 0 : vis_cap_1);
                int kv_end_raw_1 = vis_out_1;
                int kv_end_1 = ((rows_valid_1 == 0) ? 0 : kv_end_raw_1);
                int n_tiles_total_1 = (kv_end_1 + 128 - 1) / 128;
                int tiles_per_split_1 = (n_tiles_total_1 + num_split - 1) / num_split;
                int my_start_raw_1 = split_idx_1 * tiles_per_split_1;
                int my_end_raw_1 = my_start_raw_1 + tiles_per_split_1;
                int my_end_1 = ((my_end_raw_1 > n_tiles_total_1) ? n_tiles_total_1 : my_end_raw_1);
                int my_n_raw_1 = my_end_1 - my_start_raw_1;
                int empty_1 = ((my_n_raw_1 < 1) ? 1 : 0);
                int my_n_tiles_1 = ((my_n_raw_1 < 1) ? 1 : my_n_raw_1);
                int my_start_1 = ((my_n_raw_1 < 1) ? 0 : my_start_raw_1);
                int pt_base_1 = b_1 * max_pages_per_seq;
                pub_tile_c = 0;
                mbarrier_wait(q_full_addr, _phase_q_full_0);
                _phase_q_full_0 ^= 1;
                unsigned int q_words_c[8];
                int qs_addr_c = smem_qs_addr + (unsigned int)(row_cc * 32);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_words_c[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_c[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_c[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_c[(0) + 3]))
                    : "r"(qs_addr_c));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_words_c[4])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_c[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_c[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_c[(4) + 3]))
                    : "r"(qs_addr_c + 16));
                unsigned int q_row_words_c[16];
                int q_swz_c = row_cc >> 1 & 3;
                int q_atom_addr_c = smem_v0_addr + (unsigned int)(row_cc * 64);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(0) + 3]))
                    : "r"(q_atom_addr_c + (q_swz_c ^ 0) * 16));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[4])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(4) + 3]))
                    : "r"(q_atom_addr_c + (q_swz_c ^ 1) * 16));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[8])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(8) + 3]))
                    : "r"(q_atom_addr_c + (q_swz_c ^ 2) * 16));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[12])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(12) + 3]))
                    : "r"(q_atom_addr_c + (q_swz_c ^ 3) * 16));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 416), "r"(q_row_words_c[0]), "r"(q_row_words_c[1]), "r"(q_row_words_c[2]), "r"(q_row_words_c[3]), "r"(q_row_words_c[4]), "r"(q_row_words_c[5]), "r"(q_row_words_c[6]), "r"(q_row_words_c[7]), "r"(q_row_words_c[8]), "r"(q_row_words_c[9]), "r"(q_row_words_c[10]), "r"(q_row_words_c[11]), "r"(q_row_words_c[12]), "r"(q_row_words_c[13]), "r"(q_row_words_c[14]), "r"(q_row_words_c[15]));
                int q_atom_addr_c_0 = smem_v1_addr + (unsigned int)(row_cc * 64);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(0) + 3]))
                    : "r"(q_atom_addr_c_0 + (q_swz_c ^ 0) * 16));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[4])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(4) + 3]))
                    : "r"(q_atom_addr_c_0 + (q_swz_c ^ 1) * 16));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[8])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(8) + 3]))
                    : "r"(q_atom_addr_c_0 + (q_swz_c ^ 2) * 16));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[12])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(12) + 3]))
                    : "r"(q_atom_addr_c_0 + (q_swz_c ^ 3) * 16));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 416 + 16), "r"(q_row_words_c[0]), "r"(q_row_words_c[1]), "r"(q_row_words_c[2]), "r"(q_row_words_c[3]), "r"(q_row_words_c[4]), "r"(q_row_words_c[5]), "r"(q_row_words_c[6]), "r"(q_row_words_c[7]), "r"(q_row_words_c[8]), "r"(q_row_words_c[9]), "r"(q_row_words_c[10]), "r"(q_row_words_c[11]), "r"(q_row_words_c[12]), "r"(q_row_words_c[13]), "r"(q_row_words_c[14]), "r"(q_row_words_c[15]));
                int q_atom_addr_c_1 = smem_v2_addr + (unsigned int)(row_cc * 64);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(0) + 3]))
                    : "r"(q_atom_addr_c_1 + (q_swz_c ^ 0) * 16));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[4])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(4) + 3]))
                    : "r"(q_atom_addr_c_1 + (q_swz_c ^ 1) * 16));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[8])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(8) + 3]))
                    : "r"(q_atom_addr_c_1 + (q_swz_c ^ 2) * 16));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[12])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(12) + 3]))
                    : "r"(q_atom_addr_c_1 + (q_swz_c ^ 3) * 16));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 416 + 32), "r"(q_row_words_c[0]), "r"(q_row_words_c[1]), "r"(q_row_words_c[2]), "r"(q_row_words_c[3]), "r"(q_row_words_c[4]), "r"(q_row_words_c[5]), "r"(q_row_words_c[6]), "r"(q_row_words_c[7]), "r"(q_row_words_c[8]), "r"(q_row_words_c[9]), "r"(q_row_words_c[10]), "r"(q_row_words_c[11]), "r"(q_row_words_c[12]), "r"(q_row_words_c[13]), "r"(q_row_words_c[14]), "r"(q_row_words_c[15]));
                int q_atom_addr_c_2 = smem_v3_addr + (unsigned int)(row_cc * 64);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(0) + 3]))
                    : "r"(q_atom_addr_c_2 + (q_swz_c ^ 0) * 16));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[4])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(4) + 3]))
                    : "r"(q_atom_addr_c_2 + (q_swz_c ^ 1) * 16));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[8])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(8) + 3]))
                    : "r"(q_atom_addr_c_2 + (q_swz_c ^ 2) * 16));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[12])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(12) + 3]))
                    : "r"(q_atom_addr_c_2 + (q_swz_c ^ 3) * 16));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 416 + 48), "r"(q_row_words_c[0]), "r"(q_row_words_c[1]), "r"(q_row_words_c[2]), "r"(q_row_words_c[3]), "r"(q_row_words_c[4]), "r"(q_row_words_c[5]), "r"(q_row_words_c[6]), "r"(q_row_words_c[7]), "r"(q_row_words_c[8]), "r"(q_row_words_c[9]), "r"(q_row_words_c[10]), "r"(q_row_words_c[11]), "r"(q_row_words_c[12]), "r"(q_row_words_c[13]), "r"(q_row_words_c[14]), "r"(q_row_words_c[15]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320), "r"(q_words_c[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 1), "r"(q_words_c[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 2), "r"(q_words_c[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 3), "r"(q_words_c[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 4), "r"((q_words_c + 1)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 4 + 1), "r"((q_words_c + 1)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 4 + 2), "r"((q_words_c + 1)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 4 + 3), "r"((q_words_c + 1)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 8), "r"((q_words_c + 2)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 8 + 1), "r"((q_words_c + 2)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 8 + 2), "r"((q_words_c + 2)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 8 + 3), "r"((q_words_c + 2)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 8 + 4), "r"((q_words_c + 3)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 8 + 4 + 1), "r"((q_words_c + 3)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 8 + 4 + 2), "r"((q_words_c + 3)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 8 + 4 + 3), "r"((q_words_c + 3)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 16), "r"((q_words_c + 4)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 16 + 1), "r"((q_words_c + 4)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 16 + 2), "r"((q_words_c + 4)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 16 + 3), "r"((q_words_c + 4)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 16 + 4), "r"((q_words_c + 5)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 16 + 4 + 1), "r"((q_words_c + 5)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 16 + 4 + 2), "r"((q_words_c + 5)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 16 + 4 + 3), "r"((q_words_c + 5)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 24), "r"((q_words_c + 6)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 24 + 1), "r"((q_words_c + 6)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 24 + 2), "r"((q_words_c + 6)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 24 + 3), "r"((q_words_c + 6)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 24 + 4), "r"((q_words_c + 7)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 24 + 4 + 1), "r"((q_words_c + 7)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 24 + 4 + 2), "r"((q_words_c + 7)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 24 + 4 + 3), "r"((q_words_c + 7)[0]));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(sfa_full_addr), "r"(0) : "memory");
                mbarrier_wait(raw_full_addr + (raw_stage_c) * 8, raw_phase_c);
                {
                    mbarrier_wait(sf_empty_addr, _phase_sf_empty_0);
                    _phase_sf_empty_0 ^= 1;
                }
                int sfb_base_c = taddr + (unsigned int)lane_base_c + 352;
                unsigned int k_words_c[8];
                int tok_c = quadrant_c / 2 * 32 + lane;
                int ks_addr_c = smem_ks_addr + (unsigned int)(raw_stage_c * 32768) + (unsigned int)(tok_c * 32);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c[(0) + 3]))
                    : "r"(ks_addr_c));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c[(4) + 3]))
                    : "r"(ks_addr_c + 16));
                int t_abs_c = (my_start_1 + pub_tile_c) * 128 + tok_c;
                if (t_abs_c >= kv_end_1 || empty_1 != 0) {
                    k_words_c[0] = 0;
                    k_words_c[1] = 0;
                    k_words_c[2] = 0;
                    k_words_c[3] = 0;
                    k_words_c[4] = 0;
                    k_words_c[5] = 0;
                    k_words_c[6] = 0;
                    k_words_c[7] = 0;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(sfb_base_c), "r"(k_words_c[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(sfb_base_c + 4), "r"((k_words_c + 1)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(sfb_base_c + 8), "r"((k_words_c + 2)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(sfb_base_c + 8 + 4), "r"((k_words_c + 3)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(sfb_base_c + 16), "r"((k_words_c + 4)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(sfb_base_c + 16 + 4), "r"((k_words_c + 5)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(sfb_base_c + 24), "r"((k_words_c + 6)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(sfb_base_c + 24 + 4), "r"((k_words_c + 7)[0]));
                mbarrier_wait(raw_full_addr + (raw_stage_c) * 8, raw_phase_c);
                int sfb_base_c_3 = taddr + (unsigned int)lane_base_c + 352 + 32;
                unsigned int k_words_c_4[8];
                int tok_c_5 = 64 + quadrant_c / 2 * 32 + lane;
                int ks_addr_c_6 = smem_ks_addr + (unsigned int)(raw_stage_c * 32768) + (unsigned int)(tok_c_5 * 32);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_4[(0) + 3]))
                    : "r"(ks_addr_c_6));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_4[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_4[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_4[(4) + 3]))
                    : "r"(ks_addr_c_6 + 16));
                int t_abs_c_7 = (my_start_1 + pub_tile_c) * 128 + tok_c_5;
                if (t_abs_c_7 >= kv_end_1 || empty_1 != 0) {
                    k_words_c_4[0] = 0;
                    k_words_c_4[1] = 0;
                    k_words_c_4[2] = 0;
                    k_words_c_4[3] = 0;
                    k_words_c_4[4] = 0;
                    k_words_c_4[5] = 0;
                    k_words_c_4[6] = 0;
                    k_words_c_4[7] = 0;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(sfb_base_c_3), "r"(k_words_c_4[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(sfb_base_c_3 + 4), "r"((k_words_c_4 + 1)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(sfb_base_c_3 + 8), "r"((k_words_c_4 + 2)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(sfb_base_c_3 + 8 + 4), "r"((k_words_c_4 + 3)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(sfb_base_c_3 + 16), "r"((k_words_c_4 + 4)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(sfb_base_c_3 + 16 + 4), "r"((k_words_c_4 + 5)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(sfb_base_c_3 + 24), "r"((k_words_c_4 + 6)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(sfb_base_c_3 + 24 + 4), "r"((k_words_c_4 + 7)[0]));
                {
                    mbarrier_arrive(raw_empty_addr + (raw_stage_c) * 8);
                    raw_stage_c += 1;
                    if (raw_stage_c == 3) { raw_stage_c = 0; raw_phase_c ^= 1; }
                    pub_tile_c = pub_tile_c + 1;
                }
                {
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(sf_full_addr), "r"(0) : "memory");
                }
                if (my_n_tiles_1 > 1) {
                    mbarrier_wait(raw_full_addr + (raw_stage_c) * 8, raw_phase_c);
                    {
                        mbarrier_wait(sf_empty_addr, _phase_sf_empty_0);
                        _phase_sf_empty_0 ^= 1;
                    }
                    int sfb_base_c_0 = taddr + (unsigned int)lane_base_c + 352;
                    unsigned int k_words_c_1[8];
                    int tok_c_2 = quadrant_c / 2 * 32 + lane;
                    int ks_addr_c_3 = smem_ks_addr + (unsigned int)(raw_stage_c * 32768) + (unsigned int)(tok_c_2 * 32);
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1[(0) + 3]))
                        : "r"(ks_addr_c_3));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1[(4) + 3]))
                        : "r"(ks_addr_c_3 + 16));
                    int t_abs_c_4 = (my_start_1 + pub_tile_c) * 128 + tok_c_2;
                    if (t_abs_c_4 >= kv_end_1 || empty_1 != 0) {
                        k_words_c_1[0] = 0;
                        k_words_c_1[1] = 0;
                        k_words_c_1[2] = 0;
                        k_words_c_1[3] = 0;
                        k_words_c_1[4] = 0;
                        k_words_c_1[5] = 0;
                        k_words_c_1[6] = 0;
                        k_words_c_1[7] = 0;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfb_base_c_0), "r"(k_words_c_1[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfb_base_c_0 + 4), "r"((k_words_c_1 + 1)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfb_base_c_0 + 8), "r"((k_words_c_1 + 2)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfb_base_c_0 + 8 + 4), "r"((k_words_c_1 + 3)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfb_base_c_0 + 16), "r"((k_words_c_1 + 4)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfb_base_c_0 + 16 + 4), "r"((k_words_c_1 + 5)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfb_base_c_0 + 24), "r"((k_words_c_1 + 6)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfb_base_c_0 + 24 + 4), "r"((k_words_c_1 + 7)[0]));
                    mbarrier_wait(raw_full_addr + (raw_stage_c) * 8, raw_phase_c);
                    int sfb_base_c_5 = taddr + (unsigned int)lane_base_c + 352 + 32;
                    unsigned int k_words_c_6[8];
                    int tok_c_7 = 64 + quadrant_c / 2 * 32 + lane;
                    int ks_addr_c_8 = smem_ks_addr + (unsigned int)(raw_stage_c * 32768) + (unsigned int)(tok_c_7 * 32);
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_6[(0) + 3]))
                        : "r"(ks_addr_c_8));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_6[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_6[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_6[(4) + 3]))
                        : "r"(ks_addr_c_8 + 16));
                    int t_abs_c_9 = (my_start_1 + pub_tile_c) * 128 + tok_c_7;
                    if (t_abs_c_9 >= kv_end_1 || empty_1 != 0) {
                        k_words_c_6[0] = 0;
                        k_words_c_6[1] = 0;
                        k_words_c_6[2] = 0;
                        k_words_c_6[3] = 0;
                        k_words_c_6[4] = 0;
                        k_words_c_6[5] = 0;
                        k_words_c_6[6] = 0;
                        k_words_c_6[7] = 0;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfb_base_c_5), "r"(k_words_c_6[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfb_base_c_5 + 4), "r"((k_words_c_6 + 1)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfb_base_c_5 + 8), "r"((k_words_c_6 + 2)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfb_base_c_5 + 8 + 4), "r"((k_words_c_6 + 3)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfb_base_c_5 + 16), "r"((k_words_c_6 + 4)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfb_base_c_5 + 16 + 4), "r"((k_words_c_6 + 5)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfb_base_c_5 + 24), "r"((k_words_c_6 + 6)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfb_base_c_5 + 24 + 4), "r"((k_words_c_6 + 7)[0]));
                    {
                        mbarrier_arrive(raw_empty_addr + (raw_stage_c) * 8);
                        raw_stage_c += 1;
                        if (raw_stage_c == 3) { raw_stage_c = 0; raw_phase_c ^= 1; }
                        pub_tile_c = pub_tile_c + 1;
                    }
                    {
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(sf_full_addr), "r"(0) : "memory");
                    }
                }
                #pragma unroll 1
                for (int it_c = 0; it_c < my_n_tiles_1; it_c++) {
                    mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                    _phase_corr_sig_0 ^= 1;
                    float alpha_c = row_state[row_cc];
                    mbarrier_arrive(corr_empty_addr);
                    int need_rescale = 0;
                    if (it_c > 0) {
                        int _vote_0 = __all_sync(0xFFFFFFFF, alpha_c == 1.0f);
                        need_rescale = _vote_0 == 0;
                    }
                    if (need_rescale != 0) {
                        mbarrier_wait(pv_done_addr, _phase_pv_done_0);
                        _phase_pv_done_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        #pragma unroll
                        for (int col = 0; col < 256; col += 32) {
                            int o_addr = taddr + (unsigned int)lane_base_c + 64 + (unsigned int)col;
                            float _tmem_load_0[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                                : "r"(o_addr));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_0 = {alpha_c, alpha_c};
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_ls], _scale2_0);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 32; _ls++) {
                                _tmem_load_0[_ls] = _tmem_load_0[_ls] * alpha_c;
                            }
                            #endif
                            tmem_st_x32_f32(o_addr, _tmem_load_0);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(p_full_addr + p_stage_c * 8), "r"(0) : "memory");
                    if (it_c > 0) {
                        if (need_rescale == 0) {
                            mbarrier_wait(pv_done_addr, _phase_pv_done_0);
                            _phase_pv_done_0 ^= 1;
                        }
                    }
                    p_stage_c += 1;
                    if (p_stage_c == 2) { p_stage_c = 0; p_phase_c ^= 1; }
                    if (my_n_tiles_1 > it_c + 2) {
                        mbarrier_wait(raw_full_addr + (raw_stage_c) * 8, raw_phase_c);
                        {
                            mbarrier_wait(sf_empty_addr, _phase_sf_empty_0);
                            _phase_sf_empty_0 ^= 1;
                        }
                        int sfb_base_c_0_1 = taddr + (unsigned int)lane_base_c + 352;
                        unsigned int k_words_c_1_1[8];
                        int tok_c_2_1 = quadrant_c / 2 * 32 + lane;
                        int ks_addr_c_3_1 = smem_ks_addr + (unsigned int)(raw_stage_c * 32768) + (unsigned int)(tok_c_2_1 * 32);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1_1[(0) + 3]))
                            : "r"(ks_addr_c_3_1));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1_1[(4) + 3]))
                            : "r"(ks_addr_c_3_1 + 16));
                        int t_abs_c_4_1 = (my_start_1 + pub_tile_c) * 128 + tok_c_2_1;
                        if (t_abs_c_4_1 >= kv_end_1 || empty_1 != 0) {
                            k_words_c_1_1[0] = 0;
                            k_words_c_1_1[1] = 0;
                            k_words_c_1_1[2] = 0;
                            k_words_c_1_1[3] = 0;
                            k_words_c_1_1[4] = 0;
                            k_words_c_1_1[5] = 0;
                            k_words_c_1_1[6] = 0;
                            k_words_c_1_1[7] = 0;
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c_0_1), "r"(k_words_c_1_1[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c_0_1 + 4), "r"((k_words_c_1_1 + 1)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c_0_1 + 8), "r"((k_words_c_1_1 + 2)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c_0_1 + 8 + 4), "r"((k_words_c_1_1 + 3)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c_0_1 + 16), "r"((k_words_c_1_1 + 4)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c_0_1 + 16 + 4), "r"((k_words_c_1_1 + 5)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c_0_1 + 24), "r"((k_words_c_1_1 + 6)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c_0_1 + 24 + 4), "r"((k_words_c_1_1 + 7)[0]));
                        mbarrier_wait(raw_full_addr + (raw_stage_c) * 8, raw_phase_c);
                        int sfb_base_c_5_1 = taddr + (unsigned int)lane_base_c + 352 + 32;
                        unsigned int k_words_c_6_1[8];
                        int tok_c_7_1 = 64 + quadrant_c / 2 * 32 + lane;
                        int ks_addr_c_8_1 = smem_ks_addr + (unsigned int)(raw_stage_c * 32768) + (unsigned int)(tok_c_7_1 * 32);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_6_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_6_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_6_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_6_1[(0) + 3]))
                            : "r"(ks_addr_c_8_1));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_6_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_6_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_6_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_6_1[(4) + 3]))
                            : "r"(ks_addr_c_8_1 + 16));
                        int t_abs_c_9_1 = (my_start_1 + pub_tile_c) * 128 + tok_c_7_1;
                        if (t_abs_c_9_1 >= kv_end_1 || empty_1 != 0) {
                            k_words_c_6_1[0] = 0;
                            k_words_c_6_1[1] = 0;
                            k_words_c_6_1[2] = 0;
                            k_words_c_6_1[3] = 0;
                            k_words_c_6_1[4] = 0;
                            k_words_c_6_1[5] = 0;
                            k_words_c_6_1[6] = 0;
                            k_words_c_6_1[7] = 0;
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c_5_1), "r"(k_words_c_6_1[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c_5_1 + 4), "r"((k_words_c_6_1 + 1)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c_5_1 + 8), "r"((k_words_c_6_1 + 2)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c_5_1 + 8 + 4), "r"((k_words_c_6_1 + 3)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c_5_1 + 16), "r"((k_words_c_6_1 + 4)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c_5_1 + 16 + 4), "r"((k_words_c_6_1 + 5)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c_5_1 + 24), "r"((k_words_c_6_1 + 6)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c_5_1 + 24 + 4), "r"((k_words_c_6_1 + 7)[0]));
                        {
                            mbarrier_arrive(raw_empty_addr + (raw_stage_c) * 8);
                            raw_stage_c += 1;
                            if (raw_stage_c == 3) { raw_stage_c = 0; raw_phase_c ^= 1; }
                            pub_tile_c = pub_tile_c + 1;
                        }
                        {
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                                "}"
                                :: "r"(sf_full_addr), "r"(0) : "memory");
                        }
                    }
                }
                mbarrier_wait(pv_done_addr, _phase_pv_done_0);
                _phase_pv_done_0 ^= 1;
            }
        }
    }
    // ---- Role: transform ----
    if (warp >= 8 && warp <= 13) {
        { // transform_main
            int raw_stage_t = 0;
            int raw_phase_t = 0;
            int v_stage_t = 0;
            int v_phase_t = 1;
            int role_tid = (warp - 8) * 32 + lane;
            int rank_t = cta_rank;
            int pair_r = role_tid;
            int src_r = (1 - ((pair_r / 8 / 2 * 2 + pair_r % 8 / 4 - (pair_r / 8 / 2 * 2 + pair_r % 8 / 4) / 64 * 64) / 32 ^ rank_t)) * ((pair_r / 8 % 2 * 2 + rank_t) * 4096) + ((pair_r / 8 / 2 * 2 + pair_r % 8 / 4 - (pair_r / 8 / 2 * 2 + pair_r % 8 / 4) / 64 * 64) / 32 ^ rank_t) * (16384 + pair_r / 8 % 2 * 4096) + (pair_r / 8 / 2 * 2 + pair_r % 8 / 4) / 64 * 2048 + (pair_r / 8 / 2 * 2 + pair_r % 8 / 4 - (pair_r / 8 / 2 * 2 + pair_r % 8 / 4) / 64 * 64) % 32 * 64 + (pair_r % 4 ^ (pair_r / 8 / 2 * 2 + pair_r % 8 / 4 - (pair_r / 8 / 2 * 2 + pair_r % 8 / 4) / 64 * 64) % 32 >> 1 & 3) * 16;
            int sc_r = (pair_r / 8 / 2 * 2 + pair_r % 8 / 4) * 32 + (pair_r / 8 % 2 * 2 + rank_t) * 8 + pair_r % 4 * 2;
            int v_r = pair_r / 8 % 2 * 16384 + ((pair_r / 8 / 2 * 2 + pair_r % 8 / 4) * 128 + pair_r % 4 * 32 ^ ((pair_r / 8 / 2 * 2 + pair_r % 8 / 4) * 128 + pair_r % 4 * 32 >> 7 & 7) << 4);
            int pair_r_0 = role_tid + 192;
            int src_r_1 = (1 - ((pair_r_0 / 8 / 2 * 2 + pair_r_0 % 8 / 4 - (pair_r_0 / 8 / 2 * 2 + pair_r_0 % 8 / 4) / 64 * 64) / 32 ^ rank_t)) * ((pair_r_0 / 8 % 2 * 2 + rank_t) * 4096) + ((pair_r_0 / 8 / 2 * 2 + pair_r_0 % 8 / 4 - (pair_r_0 / 8 / 2 * 2 + pair_r_0 % 8 / 4) / 64 * 64) / 32 ^ rank_t) * (16384 + pair_r_0 / 8 % 2 * 4096) + (pair_r_0 / 8 / 2 * 2 + pair_r_0 % 8 / 4) / 64 * 2048 + (pair_r_0 / 8 / 2 * 2 + pair_r_0 % 8 / 4 - (pair_r_0 / 8 / 2 * 2 + pair_r_0 % 8 / 4) / 64 * 64) % 32 * 64 + (pair_r_0 % 4 ^ (pair_r_0 / 8 / 2 * 2 + pair_r_0 % 8 / 4 - (pair_r_0 / 8 / 2 * 2 + pair_r_0 % 8 / 4) / 64 * 64) % 32 >> 1 & 3) * 16;
            int sc_r_2 = (pair_r_0 / 8 / 2 * 2 + pair_r_0 % 8 / 4) * 32 + (pair_r_0 / 8 % 2 * 2 + rank_t) * 8 + pair_r_0 % 4 * 2;
            int v_r_3 = pair_r_0 / 8 % 2 * 16384 + ((pair_r_0 / 8 / 2 * 2 + pair_r_0 % 8 / 4) * 128 + pair_r_0 % 4 * 32 ^ ((pair_r_0 / 8 / 2 * 2 + pair_r_0 % 8 / 4) * 128 + pair_r_0 % 4 * 32 >> 7 & 7) << 4);
            int pair_r_4 = role_tid + 384;
            int src_r_5 = (1 - ((pair_r_4 / 8 / 2 * 2 + pair_r_4 % 8 / 4 - (pair_r_4 / 8 / 2 * 2 + pair_r_4 % 8 / 4) / 64 * 64) / 32 ^ rank_t)) * ((pair_r_4 / 8 % 2 * 2 + rank_t) * 4096) + ((pair_r_4 / 8 / 2 * 2 + pair_r_4 % 8 / 4 - (pair_r_4 / 8 / 2 * 2 + pair_r_4 % 8 / 4) / 64 * 64) / 32 ^ rank_t) * (16384 + pair_r_4 / 8 % 2 * 4096) + (pair_r_4 / 8 / 2 * 2 + pair_r_4 % 8 / 4) / 64 * 2048 + (pair_r_4 / 8 / 2 * 2 + pair_r_4 % 8 / 4 - (pair_r_4 / 8 / 2 * 2 + pair_r_4 % 8 / 4) / 64 * 64) % 32 * 64 + (pair_r_4 % 4 ^ (pair_r_4 / 8 / 2 * 2 + pair_r_4 % 8 / 4 - (pair_r_4 / 8 / 2 * 2 + pair_r_4 % 8 / 4) / 64 * 64) % 32 >> 1 & 3) * 16;
            int sc_r_6 = (pair_r_4 / 8 / 2 * 2 + pair_r_4 % 8 / 4) * 32 + (pair_r_4 / 8 % 2 * 2 + rank_t) * 8 + pair_r_4 % 4 * 2;
            int v_r_7 = pair_r_4 / 8 % 2 * 16384 + ((pair_r_4 / 8 / 2 * 2 + pair_r_4 % 8 / 4) * 128 + pair_r_4 % 4 * 32 ^ ((pair_r_4 / 8 / 2 * 2 + pair_r_4 % 8 / 4) * 128 + pair_r_4 % 4 * 32 >> 7 & 7) << 4);
            int pair_r_8 = role_tid + 576;
            int src_r_9 = (1 - ((pair_r_8 / 8 / 2 * 2 + pair_r_8 % 8 / 4 - (pair_r_8 / 8 / 2 * 2 + pair_r_8 % 8 / 4) / 64 * 64) / 32 ^ rank_t)) * ((pair_r_8 / 8 % 2 * 2 + rank_t) * 4096) + ((pair_r_8 / 8 / 2 * 2 + pair_r_8 % 8 / 4 - (pair_r_8 / 8 / 2 * 2 + pair_r_8 % 8 / 4) / 64 * 64) / 32 ^ rank_t) * (16384 + pair_r_8 / 8 % 2 * 4096) + (pair_r_8 / 8 / 2 * 2 + pair_r_8 % 8 / 4) / 64 * 2048 + (pair_r_8 / 8 / 2 * 2 + pair_r_8 % 8 / 4 - (pair_r_8 / 8 / 2 * 2 + pair_r_8 % 8 / 4) / 64 * 64) % 32 * 64 + (pair_r_8 % 4 ^ (pair_r_8 / 8 / 2 * 2 + pair_r_8 % 8 / 4 - (pair_r_8 / 8 / 2 * 2 + pair_r_8 % 8 / 4) / 64 * 64) % 32 >> 1 & 3) * 16;
            int sc_r_10 = (pair_r_8 / 8 / 2 * 2 + pair_r_8 % 8 / 4) * 32 + (pair_r_8 / 8 % 2 * 2 + rank_t) * 8 + pair_r_8 % 4 * 2;
            int v_r_11 = pair_r_8 / 8 % 2 * 16384 + ((pair_r_8 / 8 / 2 * 2 + pair_r_8 % 8 / 4) * 128 + pair_r_8 % 4 * 32 ^ ((pair_r_8 / 8 / 2 * 2 + pair_r_8 % 8 / 4) * 128 + pair_r_8 % 4 * 32 >> 7 & 7) << 4);
            int pair_r_12 = role_tid + 768;
            int src_r_13 = (1 - ((pair_r_12 / 8 / 2 * 2 + pair_r_12 % 8 / 4 - (pair_r_12 / 8 / 2 * 2 + pair_r_12 % 8 / 4) / 64 * 64) / 32 ^ rank_t)) * ((pair_r_12 / 8 % 2 * 2 + rank_t) * 4096) + ((pair_r_12 / 8 / 2 * 2 + pair_r_12 % 8 / 4 - (pair_r_12 / 8 / 2 * 2 + pair_r_12 % 8 / 4) / 64 * 64) / 32 ^ rank_t) * (16384 + pair_r_12 / 8 % 2 * 4096) + (pair_r_12 / 8 / 2 * 2 + pair_r_12 % 8 / 4) / 64 * 2048 + (pair_r_12 / 8 / 2 * 2 + pair_r_12 % 8 / 4 - (pair_r_12 / 8 / 2 * 2 + pair_r_12 % 8 / 4) / 64 * 64) % 32 * 64 + (pair_r_12 % 4 ^ (pair_r_12 / 8 / 2 * 2 + pair_r_12 % 8 / 4 - (pair_r_12 / 8 / 2 * 2 + pair_r_12 % 8 / 4) / 64 * 64) % 32 >> 1 & 3) * 16;
            int sc_r_14 = (pair_r_12 / 8 / 2 * 2 + pair_r_12 % 8 / 4) * 32 + (pair_r_12 / 8 % 2 * 2 + rank_t) * 8 + pair_r_12 % 4 * 2;
            int v_r_15 = pair_r_12 / 8 % 2 * 16384 + ((pair_r_12 / 8 / 2 * 2 + pair_r_12 % 8 / 4) * 128 + pair_r_12 % 4 * 32 ^ ((pair_r_12 / 8 / 2 * 2 + pair_r_12 % 8 / 4) * 128 + pair_r_12 % 4 * 32 >> 7 & 7) << 4);
            int pair_tail = role_tid - 64 + 960;
            int src_tail = (1 - ((pair_tail / 8 / 2 * 2 + pair_tail % 8 / 4 - (pair_tail / 8 / 2 * 2 + pair_tail % 8 / 4) / 64 * 64) / 32 ^ rank_t)) * ((pair_tail / 8 % 2 * 2 + rank_t) * 4096) + ((pair_tail / 8 / 2 * 2 + pair_tail % 8 / 4 - (pair_tail / 8 / 2 * 2 + pair_tail % 8 / 4) / 64 * 64) / 32 ^ rank_t) * (16384 + pair_tail / 8 % 2 * 4096) + (pair_tail / 8 / 2 * 2 + pair_tail % 8 / 4) / 64 * 2048 + (pair_tail / 8 / 2 * 2 + pair_tail % 8 / 4 - (pair_tail / 8 / 2 * 2 + pair_tail % 8 / 4) / 64 * 64) % 32 * 64 + (pair_tail % 4 ^ (pair_tail / 8 / 2 * 2 + pair_tail % 8 / 4 - (pair_tail / 8 / 2 * 2 + pair_tail % 8 / 4) / 64 * 64) % 32 >> 1 & 3) * 16;
            int sc_tail = (pair_tail / 8 / 2 * 2 + pair_tail % 8 / 4) * 32 + (pair_tail / 8 % 2 * 2 + rank_t) * 8 + pair_tail % 4 * 2;
            int v_tail = pair_tail / 8 % 2 * 16384 + ((pair_tail / 8 / 2 * 2 + pair_tail % 8 / 4) * 128 + pair_tail % 4 * 32 ^ ((pair_tail / 8 / 2 * 2 + pair_tail % 8 / 4) * 128 + pair_tail % 4 * 32 >> 7 & 7) << 4);
            int cx_2 = blockIdx.x / 2;
            int n_clusters_2 = gridDim.x / 2;
            int bz_2 = blockIdx.z;
            int q_end_bz_2 = cum_seq_lens_q[bz_2 + 1];
            int q_start_bz_2 = cum_seq_lens_q[bz_2];
            int kv_len_bz_2 = seq_lens[bz_2];
            int g_len_bz_2 = kv_len_global[bz_2];
            int rows_bz_2 = (q_end_bz_2 - q_start_bz_2) * num_heads;
            int pieces_2 = (rows_bz_2 + 128 - 1) / 128 * num_split;
            int left_2 = pieces_2 - cx_2;
            int n_pieces_raw_2 = (left_2 + n_clusters_2 - 1) / n_clusters_2;
            int n_pieces_2 = ((left_2 <= 0) ? 0 : n_pieces_raw_2);
            int epi_phase_t = 0;
            #pragma unroll 1
            for (int pi_t = 0; pi_t < n_pieces_2; pi_t++) {
                int p_t = cx_2 + pi_t * n_clusters_2;
                int b_2 = blockIdx.z;
                int split_idx_2 = p_t % num_split;
                int rows_b0_2 = (q_end_bz_2 - q_start_bz_2) * num_heads;
                int m_tiles_b_2 = (rows_b0_2 + 128 - 1) / 128;
                int m_tile_2 = m_tiles_b_2 - 1 - p_t / num_split;
                int q_start_2 = q_start_bz_2;
                int q_len_b_2 = q_end_bz_2 - q_start_bz_2;
                int kv_len_2 = kv_len_bz_2;
                int g_len_2 = g_len_bz_2;
                int rows_b_2 = q_len_b_2 * num_heads;
                int row0_2 = m_tile_2 * 128;
                int rows_left_2 = rows_b_2 - row0_2;
                int rows_pos_2 = ((rows_left_2 < 0) ? 0 : rows_left_2);
                int rows_valid_2 = ((rows_pos_2 > 128) ? 128 : rows_pos_2);
                int row_base_global_2 = q_start_2 * num_heads + row0_2;
                int last_row_2 = row0_2 + rows_valid_2 - 1;
                int t_last_2 = last_row_2 / num_heads;
                int num_2 = g_len_2 - q_len_b_2 + t_last_2 - cp_rank;
                int vis_raw_3 = num_2 / cp_world + 1;
                int vis_cap_3 = ((vis_raw_3 > kv_len_2) ? kv_len_2 : vis_raw_3);
                int vis_out_2 = ((num_2 < 0) ? 0 : vis_cap_3);
                int kv_end_raw_2 = vis_out_2;
                int kv_end_2 = ((rows_valid_2 == 0) ? 0 : kv_end_raw_2);
                int n_tiles_total_2 = (kv_end_2 + 128 - 1) / 128;
                int tiles_per_split_2 = (n_tiles_total_2 + num_split - 1) / num_split;
                int my_start_raw_2 = split_idx_2 * tiles_per_split_2;
                int my_end_raw_2 = my_start_raw_2 + tiles_per_split_2;
                int my_end_2 = ((my_end_raw_2 > n_tiles_total_2) ? n_tiles_total_2 : my_end_raw_2);
                int my_n_raw_2 = my_end_2 - my_start_raw_2;
                int empty_2 = ((my_n_raw_2 < 1) ? 1 : 0);
                int my_n_tiles_2 = ((my_n_raw_2 < 1) ? 1 : my_n_raw_2);
                int my_start_2 = ((my_n_raw_2 < 1) ? 0 : my_start_raw_2);
                int pt_base_2 = b_2 * max_pages_per_seq;
                if (pi_t > 0) {
                    mbarrier_wait(epi_done_addr, epi_phase_t);
                    epi_phase_t = epi_phase_t ^ 1;
                }
                #pragma unroll 1
                for (int it_t = 0; it_t < my_n_tiles_2; it_t++) {
                    mbarrier_wait(raw_full_addr + (raw_stage_t) * 8, raw_phase_t);
                    mbarrier_wait(v_empty_addr + (v_stage_t) * 8, v_phase_t);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    int raw_base_t = smem_v6_addr + (unsigned int)(raw_stage_t * 32768);
                    int scale_off_t = raw_stage_t * 32768 + 24576;
                    int v_off_t = v_stage_t * 32768;
                    unsigned int packed[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 3]))
                        : "r"(raw_base_t + src_r));
                    unsigned int scale0 = smem_raw_flat[scale_off_t + sc_r];
                    unsigned int scale1 = smem_raw_flat[scale_off_t + sc_r + 1];
                    unsigned int packed_0[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&packed_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0[(0) + 3]))
                        : "r"(raw_base_t + src_r_1));
                    unsigned int scale0_1 = smem_raw_flat[scale_off_t + sc_r_2];
                    unsigned int scale1_2 = smem_raw_flat[scale_off_t + sc_r_2 + 1];
                    unsigned int packed_3[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&packed_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_3[(0) + 3]))
                        : "r"(raw_base_t + src_r_5));
                    unsigned int scale0_4 = smem_raw_flat[scale_off_t + sc_r_6];
                    unsigned int scale1_5 = smem_raw_flat[scale_off_t + sc_r_6 + 1];
                    unsigned int zero_u = 0;
                    unsigned int d0 = scale0 - 24;
                    unsigned int d1 = scale1 - 24;
                    unsigned int s0 = ((scale0 >= 32) ? d0 : zero_u);
                    unsigned int s1 = ((scale1 >= 32) ? d1 : zero_u);
                    unsigned int sc0 = s0 * 16843009;
                    unsigned int sc1 = s1 * 16843009;
                    unsigned int converted[8];
                    {
                        converted[0] = cake_mla_nvfp4_qmul4<5>(packed[0], sc0);
                    }
                    {
                        converted[1] = cake_mla_nvfp4_qmul4<6>(packed[0], sc0);
                    }
                    {
                        converted[2] = cake_mla_nvfp4_qmul4<5>(packed[1], sc0);
                    }
                    {
                        converted[3] = cake_mla_nvfp4_qmul4<6>(packed[1], sc0);
                    }
                    {
                        converted[4] = cake_mla_nvfp4_qmul4<5>(packed[2], sc1);
                    }
                    {
                        converted[5] = cake_mla_nvfp4_qmul4<6>(packed[2], sc1);
                    }
                    {
                        converted[6] = cake_mla_nvfp4_qmul4<5>(packed[3], sc1);
                    }
                    {
                        converted[7] = cake_mla_nvfp4_qmul4<6>(packed[3], sc1);
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v30_addr + (unsigned int)(v_off_t + v_r)), "r"(converted[0]), "r"(converted[1]), "r"(converted[2]), "r"(converted[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v30_addr + (unsigned int)(v_off_t + (v_r ^ 16))), "r"(converted[4]), "r"(converted[5]), "r"(converted[6]), "r"(converted[7]) : "memory");
                    unsigned int packed_6[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&packed_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_6[(0) + 3]))
                        : "r"(raw_base_t + src_r_9));
                    unsigned int scale0_7 = smem_raw_flat[scale_off_t + sc_r_10];
                    unsigned int scale1_8 = smem_raw_flat[scale_off_t + sc_r_10 + 1];
                    unsigned int zero_u_9 = 0;
                    unsigned int d0_10 = scale0_1 - 24;
                    unsigned int d1_11 = scale1_2 - 24;
                    unsigned int s0_12 = ((scale0_1 >= 32) ? d0_10 : zero_u_9);
                    unsigned int s1_13 = ((scale1_2 >= 32) ? d1_11 : zero_u_9);
                    unsigned int sc0_14 = s0_12 * 16843009;
                    unsigned int sc1_15 = s1_13 * 16843009;
                    unsigned int converted_16[8];
                    {
                        converted_16[0] = cake_mla_nvfp4_qmul4<5>(packed_0[0], sc0_14);
                    }
                    {
                        converted_16[1] = cake_mla_nvfp4_qmul4<6>(packed_0[0], sc0_14);
                    }
                    {
                        converted_16[2] = cake_mla_nvfp4_qmul4<5>(packed_0[1], sc0_14);
                    }
                    {
                        converted_16[3] = cake_mla_nvfp4_qmul4<6>(packed_0[1], sc0_14);
                    }
                    {
                        converted_16[4] = cake_mla_nvfp4_qmul4<5>(packed_0[2], sc1_15);
                    }
                    {
                        converted_16[5] = cake_mla_nvfp4_qmul4<6>(packed_0[2], sc1_15);
                    }
                    {
                        converted_16[6] = cake_mla_nvfp4_qmul4<5>(packed_0[3], sc1_15);
                    }
                    {
                        converted_16[7] = cake_mla_nvfp4_qmul4<6>(packed_0[3], sc1_15);
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v30_addr + (unsigned int)(v_off_t + v_r_3)), "r"(converted_16[0]), "r"(converted_16[1]), "r"(converted_16[2]), "r"(converted_16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v30_addr + (unsigned int)(v_off_t + (v_r_3 ^ 16))), "r"(converted_16[4]), "r"(converted_16[5]), "r"(converted_16[6]), "r"(converted_16[7]) : "memory");
                    unsigned int packed_17[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&packed_17[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_17[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_17[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_17[(0) + 3]))
                        : "r"(raw_base_t + src_r_13));
                    unsigned int scale0_18 = smem_raw_flat[scale_off_t + sc_r_14];
                    unsigned int scale1_19 = smem_raw_flat[scale_off_t + sc_r_14 + 1];
                    unsigned int zero_u_20 = 0;
                    unsigned int d0_21 = scale0_4 - 24;
                    unsigned int d1_22 = scale1_5 - 24;
                    unsigned int s0_23 = ((scale0_4 >= 32) ? d0_21 : zero_u_20);
                    unsigned int s1_24 = ((scale1_5 >= 32) ? d1_22 : zero_u_20);
                    unsigned int sc0_25 = s0_23 * 16843009;
                    unsigned int sc1_26 = s1_24 * 16843009;
                    unsigned int converted_27[8];
                    {
                        converted_27[0] = cake_mla_nvfp4_qmul4<5>(packed_3[0], sc0_25);
                    }
                    {
                        converted_27[1] = cake_mla_nvfp4_qmul4<6>(packed_3[0], sc0_25);
                    }
                    {
                        converted_27[2] = cake_mla_nvfp4_qmul4<5>(packed_3[1], sc0_25);
                    }
                    {
                        converted_27[3] = cake_mla_nvfp4_qmul4<6>(packed_3[1], sc0_25);
                    }
                    {
                        converted_27[4] = cake_mla_nvfp4_qmul4<5>(packed_3[2], sc1_26);
                    }
                    {
                        converted_27[5] = cake_mla_nvfp4_qmul4<6>(packed_3[2], sc1_26);
                    }
                    {
                        converted_27[6] = cake_mla_nvfp4_qmul4<5>(packed_3[3], sc1_26);
                    }
                    {
                        converted_27[7] = cake_mla_nvfp4_qmul4<6>(packed_3[3], sc1_26);
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v30_addr + (unsigned int)(v_off_t + v_r_7)), "r"(converted_27[0]), "r"(converted_27[1]), "r"(converted_27[2]), "r"(converted_27[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v30_addr + (unsigned int)(v_off_t + (v_r_7 ^ 16))), "r"(converted_27[4]), "r"(converted_27[5]), "r"(converted_27[6]), "r"(converted_27[7]) : "memory");
                    unsigned int zero_u_28 = 0;
                    unsigned int d0_29 = scale0_7 - 24;
                    unsigned int d1_30 = scale1_8 - 24;
                    unsigned int s0_31 = ((scale0_7 >= 32) ? d0_29 : zero_u_28);
                    unsigned int s1_32 = ((scale1_8 >= 32) ? d1_30 : zero_u_28);
                    unsigned int sc0_33 = s0_31 * 16843009;
                    unsigned int sc1_34 = s1_32 * 16843009;
                    unsigned int converted_35[8];
                    {
                        converted_35[0] = cake_mla_nvfp4_qmul4<5>(packed_6[0], sc0_33);
                    }
                    {
                        converted_35[1] = cake_mla_nvfp4_qmul4<6>(packed_6[0], sc0_33);
                    }
                    {
                        converted_35[2] = cake_mla_nvfp4_qmul4<5>(packed_6[1], sc0_33);
                    }
                    {
                        converted_35[3] = cake_mla_nvfp4_qmul4<6>(packed_6[1], sc0_33);
                    }
                    {
                        converted_35[4] = cake_mla_nvfp4_qmul4<5>(packed_6[2], sc1_34);
                    }
                    {
                        converted_35[5] = cake_mla_nvfp4_qmul4<6>(packed_6[2], sc1_34);
                    }
                    {
                        converted_35[6] = cake_mla_nvfp4_qmul4<5>(packed_6[3], sc1_34);
                    }
                    {
                        converted_35[7] = cake_mla_nvfp4_qmul4<6>(packed_6[3], sc1_34);
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v30_addr + (unsigned int)(v_off_t + v_r_11)), "r"(converted_35[0]), "r"(converted_35[1]), "r"(converted_35[2]), "r"(converted_35[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v30_addr + (unsigned int)(v_off_t + (v_r_11 ^ 16))), "r"(converted_35[4]), "r"(converted_35[5]), "r"(converted_35[6]), "r"(converted_35[7]) : "memory");
                    if (role_tid >= 64 && role_tid < 128) {
                        unsigned int packed_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1[(0) + 3]))
                            : "r"(raw_base_t + src_tail));
                        unsigned int scale0_2 = smem_raw_flat[scale_off_t + sc_tail];
                        unsigned int scale1_3 = smem_raw_flat[scale_off_t + sc_tail + 1];
                        unsigned int zero_u_4 = 0;
                        unsigned int d0_5 = scale0_2 - 24;
                        unsigned int d1_6 = scale1_3 - 24;
                        unsigned int s0_7 = ((scale0_2 >= 32) ? d0_5 : zero_u_4);
                        unsigned int s1_8 = ((scale1_3 >= 32) ? d1_6 : zero_u_4);
                        unsigned int sc0_9 = s0_7 * 16843009;
                        unsigned int sc1_10 = s1_8 * 16843009;
                        unsigned int converted_11[8];
                        {
                            converted_11[0] = cake_mla_nvfp4_qmul4<5>(packed_1[0], sc0_9);
                        }
                        {
                            converted_11[1] = cake_mla_nvfp4_qmul4<6>(packed_1[0], sc0_9);
                        }
                        {
                            converted_11[2] = cake_mla_nvfp4_qmul4<5>(packed_1[1], sc0_9);
                        }
                        {
                            converted_11[3] = cake_mla_nvfp4_qmul4<6>(packed_1[1], sc0_9);
                        }
                        {
                            converted_11[4] = cake_mla_nvfp4_qmul4<5>(packed_1[2], sc1_10);
                        }
                        {
                            converted_11[5] = cake_mla_nvfp4_qmul4<6>(packed_1[2], sc1_10);
                        }
                        {
                            converted_11[6] = cake_mla_nvfp4_qmul4<5>(packed_1[3], sc1_10);
                        }
                        {
                            converted_11[7] = cake_mla_nvfp4_qmul4<6>(packed_1[3], sc1_10);
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v30_addr + (unsigned int)(v_off_t + v_tail)), "r"(converted_11[0]), "r"(converted_11[1]), "r"(converted_11[2]), "r"(converted_11[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v30_addr + (unsigned int)(v_off_t + (v_tail ^ 16))), "r"(converted_11[4]), "r"(converted_11[5]), "r"(converted_11[6]), "r"(converted_11[7]) : "memory");
                    }
                    unsigned int zero_u_36 = 0;
                    unsigned int d0_37 = scale0_18 - 24;
                    unsigned int d1_38 = scale1_19 - 24;
                    unsigned int s0_39 = ((scale0_18 >= 32) ? d0_37 : zero_u_36);
                    unsigned int s1_40 = ((scale1_19 >= 32) ? d1_38 : zero_u_36);
                    unsigned int sc0_41 = s0_39 * 16843009;
                    unsigned int sc1_42 = s1_40 * 16843009;
                    unsigned int converted_43[8];
                    {
                        converted_43[0] = cake_mla_nvfp4_qmul4<5>(packed_17[0], sc0_41);
                    }
                    {
                        converted_43[1] = cake_mla_nvfp4_qmul4<6>(packed_17[0], sc0_41);
                    }
                    {
                        converted_43[2] = cake_mla_nvfp4_qmul4<5>(packed_17[1], sc0_41);
                    }
                    {
                        converted_43[3] = cake_mla_nvfp4_qmul4<6>(packed_17[1], sc0_41);
                    }
                    {
                        converted_43[4] = cake_mla_nvfp4_qmul4<5>(packed_17[2], sc1_42);
                    }
                    {
                        converted_43[5] = cake_mla_nvfp4_qmul4<6>(packed_17[2], sc1_42);
                    }
                    {
                        converted_43[6] = cake_mla_nvfp4_qmul4<5>(packed_17[3], sc1_42);
                    }
                    {
                        converted_43[7] = cake_mla_nvfp4_qmul4<6>(packed_17[3], sc1_42);
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v30_addr + (unsigned int)(v_off_t + v_r_15)), "r"(converted_43[0]), "r"(converted_43[1]), "r"(converted_43[2]), "r"(converted_43[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v30_addr + (unsigned int)(v_off_t + (v_r_15 ^ 16))), "r"(converted_43[4]), "r"(converted_43[5]), "r"(converted_43[6]), "r"(converted_43[7]) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(v_full_addr + v_stage_t * 8), "r"(0) : "memory");
                    mbarrier_arrive(raw_empty_addr + (raw_stage_t) * 8);
                    raw_stage_t += 1;
                    if (raw_stage_t == 3) { raw_stage_t = 0; raw_phase_t ^= 1; }
                    v_stage_t += 1;
                    if (v_stage_t == 2) { v_stage_t = 0; v_phase_t ^= 1; }
                }
            }
        }
    }
    // ---- Role: load ----
    if (warp == 14) {
        { // load_main
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            int rank_l = cta_rank;
            int raw_stage_l = 0;
            int raw_phase_l = 1;
            int cx_3 = blockIdx.x / 2;
            int n_clusters_3 = gridDim.x / 2;
            int bz_3 = blockIdx.z;
            int q_end_bz_3 = cum_seq_lens_q[bz_3 + 1];
            int q_start_bz_3 = cum_seq_lens_q[bz_3];
            int kv_len_bz_3 = seq_lens[bz_3];
            int g_len_bz_3 = kv_len_global[bz_3];
            int rows_bz_3 = (q_end_bz_3 - q_start_bz_3) * num_heads;
            int pieces_3 = (rows_bz_3 + 128 - 1) / 128 * num_split;
            int left_3 = pieces_3 - cx_3;
            int n_pieces_raw_3 = (left_3 + n_clusters_3 - 1) / n_clusters_3;
            int n_pieces_3 = ((left_3 <= 0) ? 0 : n_pieces_raw_3);
            unsigned int _phase_q_empty_0 = 1;
            #pragma unroll 1
            for (int pi_l = 0; pi_l < n_pieces_3; pi_l++) {
                int p_l = cx_3 + pi_l * n_clusters_3;
                int b_3 = blockIdx.z;
                int split_idx_3 = p_l % num_split;
                int rows_b0_3 = (q_end_bz_3 - q_start_bz_3) * num_heads;
                int m_tiles_b_3 = (rows_b0_3 + 128 - 1) / 128;
                int m_tile_3 = m_tiles_b_3 - 1 - p_l / num_split;
                int q_start_3 = q_start_bz_3;
                int q_len_b_3 = q_end_bz_3 - q_start_bz_3;
                int kv_len_3 = kv_len_bz_3;
                int g_len_3 = g_len_bz_3;
                int rows_b_3 = q_len_b_3 * num_heads;
                int row0_3 = m_tile_3 * 128;
                int rows_left_3 = rows_b_3 - row0_3;
                int rows_pos_3 = ((rows_left_3 < 0) ? 0 : rows_left_3);
                int rows_valid_3 = ((rows_pos_3 > 128) ? 128 : rows_pos_3);
                int row_base_global_3 = q_start_3 * num_heads + row0_3;
                int last_row_3 = row0_3 + rows_valid_3 - 1;
                int t_last_3 = last_row_3 / num_heads;
                int num_3 = g_len_3 - q_len_b_3 + t_last_3 - cp_rank;
                int vis_raw_4 = num_3 / cp_world + 1;
                int vis_cap_4 = ((vis_raw_4 > kv_len_3) ? kv_len_3 : vis_raw_4);
                int vis_out_4 = ((num_3 < 0) ? 0 : vis_cap_4);
                int kv_end_raw_3 = vis_out_4;
                int kv_end_3 = ((rows_valid_3 == 0) ? 0 : kv_end_raw_3);
                int n_tiles_total_3 = (kv_end_3 + 128 - 1) / 128;
                int tiles_per_split_3 = (n_tiles_total_3 + num_split - 1) / num_split;
                int my_start_raw_3 = split_idx_3 * tiles_per_split_3;
                int my_end_raw_3 = my_start_raw_3 + tiles_per_split_3;
                int my_end_3 = ((my_end_raw_3 > n_tiles_total_3) ? n_tiles_total_3 : my_end_raw_3);
                int my_n_raw_3 = my_end_3 - my_start_raw_3;
                int empty_3 = ((my_n_raw_3 < 1) ? 1 : 0);
                int my_n_tiles_3 = ((my_n_raw_3 < 1) ? 1 : my_n_raw_3);
                int my_start_3 = ((my_n_raw_3 < 1) ? 0 : my_start_raw_3);
                int pt_base_3 = b_3 * max_pages_per_seq;
                int q_row0 = row_base_global_3 + rank_l * 64;
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                _phase_q_empty_0 ^= 1;
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr, 22528);
                    tma_2d_gmem2smem(smem_v0_addr, (&tmap_qn), 0, q_row0, q_full_addr);
                    tma_2d_gmem2smem(smem_v1_addr, (&tmap_qn), 64, q_row0, q_full_addr);
                    tma_2d_gmem2smem(smem_v2_addr, (&tmap_qn), 128, q_row0, q_full_addr);
                    tma_2d_gmem2smem(smem_v3_addr, (&tmap_qn), 192, q_row0, q_full_addr);
                    tma_2d_gmem2smem(smem_qs_addr, (&tmap_qs), 0, q_row0, q_full_addr);
                    tma_2d_gmem2smem(smem_qr_addr, (&tmap_qr), 0, q_row0, q_full_addr);
                }
                #pragma unroll 1
                for (int it_l = 0; it_l < my_n_tiles_3; it_l++) {
                    mbarrier_wait(raw_empty_addr + (raw_stage_l) * 8, raw_phase_l);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(raw_full_addr + (raw_stage_l) * 8, 32768);
                        if (rank_l == 0) {
                            int tok_raw = (my_start_3 + it_l) * 128;
                            int tok = ((tok_raw >= kv_end_3 || empty_3 != 0) ? 0 : tok_raw);
                            int pidx = tok >> page_shift;
                            int off = tok - (pidx << page_shift);
                            int g_raw = page_table[pt_base_3 + pidx];
                            int g = ((g_raw < 0) ? 0 : g_raw);
                            int tok_raw_0 = (my_start_3 + it_l) * 128 + 32;
                            int tok_1 = ((tok_raw_0 >= kv_end_3 || empty_3 != 0) ? 0 : tok_raw_0);
                            int pidx_2 = tok_1 >> page_shift;
                            int off_3 = tok_1 - (pidx_2 << page_shift);
                            int g_raw_4 = page_table[pt_base_3 + pidx_2];
                            int g_5 = ((g_raw_4 < 0) ? 0 : g_raw_4);
                            tma_3d_gmem2smem(smem_v6_addr + (unsigned int)(raw_stage_l * 32768), (&tmap_k), 0, off, g, raw_full_addr + (raw_stage_l) * 8);
                            tma_3d_gmem2smem(smem_v7_addr + (unsigned int)(raw_stage_l * 32768), (&tmap_k), 64, off, g, raw_full_addr + (raw_stage_l) * 8);
                            tma_3d_gmem2smem(smem_v8_addr + (unsigned int)(raw_stage_l * 32768), (&tmap_k), 128, off, g, raw_full_addr + (raw_stage_l) * 8);
                            tma_3d_gmem2smem(smem_v9_addr + (unsigned int)(raw_stage_l * 32768), (&tmap_k), 192, off, g, raw_full_addr + (raw_stage_l) * 8);
                            tma_3d_gmem2smem(smem_v22_addr + (unsigned int)(raw_stage_l * 32768), (&tmap_k), 0, off_3, g_5, raw_full_addr + (raw_stage_l) * 8);
                            tma_3d_gmem2smem(smem_v23_addr + (unsigned int)(raw_stage_l * 32768), (&tmap_k), 128, off_3, g_5, raw_full_addr + (raw_stage_l) * 8);
                            tma_3d_gmem2smem(smem_ks_addr + (unsigned int)(raw_stage_l * 32768), (&tmap_ks), 0, off, g, raw_full_addr + (raw_stage_l) * 8);
                            tma_3d_gmem2smem(smem_ks_addr + (unsigned int)(raw_stage_l * 32768) + 1024, (&tmap_ks), 0, off_3, g_5, raw_full_addr + (raw_stage_l) * 8);
                            tma_3d_gmem2smem(smem_v28_addr + (unsigned int)(raw_stage_l * 32768), (&tmap_kr), 0, off, g, raw_full_addr + (raw_stage_l) * 8);
                        }
                        if (rank_l == 1) {
                            int tok_raw_1 = (my_start_3 + it_l) * 128 + 32;
                            int tok_2 = ((tok_raw_1 >= kv_end_3 || empty_3 != 0) ? 0 : tok_raw_1);
                            int pidx_1 = tok_2 >> page_shift;
                            int off_1 = tok_2 - (pidx_1 << page_shift);
                            int g_raw_1 = page_table[pt_base_3 + pidx_1];
                            int g_1 = ((g_raw_1 < 0) ? 0 : g_raw_1);
                            int tok_raw_0_1 = (my_start_3 + it_l) * 128;
                            int tok_1_1 = ((tok_raw_0_1 >= kv_end_3 || empty_3 != 0) ? 0 : tok_raw_0_1);
                            int pidx_2_1 = tok_1_1 >> page_shift;
                            int off_3_1 = tok_1_1 - (pidx_2_1 << page_shift);
                            int g_raw_4_1 = page_table[pt_base_3 + pidx_2_1];
                            int g_5_1 = ((g_raw_4_1 < 0) ? 0 : g_raw_4_1);
                            tma_3d_gmem2smem(smem_v6_addr + (unsigned int)(raw_stage_l * 32768), (&tmap_k), 0, off_1, g_1, raw_full_addr + (raw_stage_l) * 8);
                            tma_3d_gmem2smem(smem_v7_addr + (unsigned int)(raw_stage_l * 32768), (&tmap_k), 64, off_1, g_1, raw_full_addr + (raw_stage_l) * 8);
                            tma_3d_gmem2smem(smem_v8_addr + (unsigned int)(raw_stage_l * 32768), (&tmap_k), 128, off_1, g_1, raw_full_addr + (raw_stage_l) * 8);
                            tma_3d_gmem2smem(smem_v9_addr + (unsigned int)(raw_stage_l * 32768), (&tmap_k), 192, off_1, g_1, raw_full_addr + (raw_stage_l) * 8);
                            tma_3d_gmem2smem(smem_v22_addr + (unsigned int)(raw_stage_l * 32768), (&tmap_k), 64, off_3_1, g_5_1, raw_full_addr + (raw_stage_l) * 8);
                            tma_3d_gmem2smem(smem_v23_addr + (unsigned int)(raw_stage_l * 32768), (&tmap_k), 192, off_3_1, g_5_1, raw_full_addr + (raw_stage_l) * 8);
                            tma_3d_gmem2smem(smem_ks_addr + (unsigned int)(raw_stage_l * 32768) + 1024, (&tmap_ks), 0, off_1, g_1, raw_full_addr + (raw_stage_l) * 8);
                            tma_3d_gmem2smem(smem_ks_addr + (unsigned int)(raw_stage_l * 32768), (&tmap_ks), 0, off_3_1, g_5_1, raw_full_addr + (raw_stage_l) * 8);
                            tma_3d_gmem2smem(smem_v28_addr + (unsigned int)(raw_stage_l * 32768), (&tmap_kr), 0, off_1, g_1, raw_full_addr + (raw_stage_l) * 8);
                        }
                        if (rank_l == 0) {
                            int tok_raw_2 = (my_start_3 + it_l) * 128 + 64;
                            int tok_3 = ((tok_raw_2 >= kv_end_3 || empty_3 != 0) ? 0 : tok_raw_2);
                            int pidx_3 = tok_3 >> page_shift;
                            int off_2 = tok_3 - (pidx_3 << page_shift);
                            int g_raw_2 = page_table[pt_base_3 + pidx_3];
                            int g_2 = ((g_raw_2 < 0) ? 0 : g_raw_2);
                            int tok_raw_0_2 = (my_start_3 + it_l) * 128 + 96;
                            int tok_1_2 = ((tok_raw_0_2 >= kv_end_3 || empty_3 != 0) ? 0 : tok_raw_0_2);
                            int pidx_2_2 = tok_1_2 >> page_shift;
                            int off_3_2 = tok_1_2 - (pidx_2_2 << page_shift);
                            int g_raw_4_2 = page_table[pt_base_3 + pidx_2_2];
                            int g_5_2 = ((g_raw_4_2 < 0) ? 0 : g_raw_4_2);
                            tma_3d_gmem2smem(smem_v10_addr + (unsigned int)(raw_stage_l * 32768), (&tmap_k), 0, off_2, g_2, raw_full_addr + (raw_stage_l) * 8);
                            tma_3d_gmem2smem(smem_v11_addr + (unsigned int)(raw_stage_l * 32768), (&tmap_k), 64, off_2, g_2, raw_full_addr + (raw_stage_l) * 8);
                            tma_3d_gmem2smem(smem_v12_addr + (unsigned int)(raw_stage_l * 32768), (&tmap_k), 128, off_2, g_2, raw_full_addr + (raw_stage_l) * 8);
                            tma_3d_gmem2smem(smem_v13_addr + (unsigned int)(raw_stage_l * 32768), (&tmap_k), 192, off_2, g_2, raw_full_addr + (raw_stage_l) * 8);
                            tma_3d_gmem2smem(smem_v24_addr + (unsigned int)(raw_stage_l * 32768), (&tmap_k), 0, off_3_2, g_5_2, raw_full_addr + (raw_stage_l) * 8);
                            tma_3d_gmem2smem(smem_v25_addr + (unsigned int)(raw_stage_l * 32768), (&tmap_k), 128, off_3_2, g_5_2, raw_full_addr + (raw_stage_l) * 8);
                            tma_3d_gmem2smem(smem_ks_addr + (unsigned int)(raw_stage_l * 32768) + 2048, (&tmap_ks), 0, off_2, g_2, raw_full_addr + (raw_stage_l) * 8);
                            tma_3d_gmem2smem(smem_ks_addr + (unsigned int)(raw_stage_l * 32768) + 3072, (&tmap_ks), 0, off_3_2, g_5_2, raw_full_addr + (raw_stage_l) * 8);
                            tma_3d_gmem2smem(smem_v29_addr + (unsigned int)(raw_stage_l * 32768), (&tmap_kr), 0, off_2, g_2, raw_full_addr + (raw_stage_l) * 8);
                        }
                        if (rank_l == 1) {
                            int tok_raw_3 = (my_start_3 + it_l) * 128 + 96;
                            int tok_4 = ((tok_raw_3 >= kv_end_3 || empty_3 != 0) ? 0 : tok_raw_3);
                            int pidx_4 = tok_4 >> page_shift;
                            int off_4 = tok_4 - (pidx_4 << page_shift);
                            int g_raw_3 = page_table[pt_base_3 + pidx_4];
                            int g_3 = ((g_raw_3 < 0) ? 0 : g_raw_3);
                            int tok_raw_0_3 = (my_start_3 + it_l) * 128 + 64;
                            int tok_1_3 = ((tok_raw_0_3 >= kv_end_3 || empty_3 != 0) ? 0 : tok_raw_0_3);
                            int pidx_2_3 = tok_1_3 >> page_shift;
                            int off_3_3 = tok_1_3 - (pidx_2_3 << page_shift);
                            int g_raw_4_3 = page_table[pt_base_3 + pidx_2_3];
                            int g_5_3 = ((g_raw_4_3 < 0) ? 0 : g_raw_4_3);
                            tma_3d_gmem2smem(smem_v10_addr + (unsigned int)(raw_stage_l * 32768), (&tmap_k), 0, off_4, g_3, raw_full_addr + (raw_stage_l) * 8);
                            tma_3d_gmem2smem(smem_v11_addr + (unsigned int)(raw_stage_l * 32768), (&tmap_k), 64, off_4, g_3, raw_full_addr + (raw_stage_l) * 8);
                            tma_3d_gmem2smem(smem_v12_addr + (unsigned int)(raw_stage_l * 32768), (&tmap_k), 128, off_4, g_3, raw_full_addr + (raw_stage_l) * 8);
                            tma_3d_gmem2smem(smem_v13_addr + (unsigned int)(raw_stage_l * 32768), (&tmap_k), 192, off_4, g_3, raw_full_addr + (raw_stage_l) * 8);
                            tma_3d_gmem2smem(smem_v24_addr + (unsigned int)(raw_stage_l * 32768), (&tmap_k), 64, off_3_3, g_5_3, raw_full_addr + (raw_stage_l) * 8);
                            tma_3d_gmem2smem(smem_v25_addr + (unsigned int)(raw_stage_l * 32768), (&tmap_k), 192, off_3_3, g_5_3, raw_full_addr + (raw_stage_l) * 8);
                            tma_3d_gmem2smem(smem_ks_addr + (unsigned int)(raw_stage_l * 32768) + 3072, (&tmap_ks), 0, off_4, g_3, raw_full_addr + (raw_stage_l) * 8);
                            tma_3d_gmem2smem(smem_ks_addr + (unsigned int)(raw_stage_l * 32768) + 2048, (&tmap_ks), 0, off_3_3, g_5_3, raw_full_addr + (raw_stage_l) * 8);
                            tma_3d_gmem2smem(smem_v29_addr + (unsigned int)(raw_stage_l * 32768), (&tmap_kr), 0, off_4, g_3, raw_full_addr + (raw_stage_l) * 8);
                        }
                    }
                    raw_stage_l += 1;
                    if (raw_stage_l == 3) { raw_stage_l = 0; raw_phase_l ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 15) {
        { // mma_main
            int raw_stage_m = 0;
            int raw_phase_m = 0;
            int v_stage_m = 0;
            int v_phase_m = 0;
            int p_stage_m = 0;
            int p_phase_m = 0;
            unsigned int _phase_q_full_0_1 = 0;
            unsigned int _phase_sfa_full_0 = 0;
            unsigned int _phase_sf_full_0 = 0;
            unsigned int _phase_s_empty_0 = 1;
            unsigned int _phase_o_empty_0 = 1;
            if (cta_rank == 0) {
                if (elect_sync()) {
                    int cx_4 = blockIdx.x / 2;
                    int n_clusters_4 = gridDim.x / 2;
                    int bz_4 = blockIdx.z;
                    int q_end_bz_4 = cum_seq_lens_q[bz_4 + 1];
                    int q_start_bz_4 = cum_seq_lens_q[bz_4];
                    int kv_len_bz_4 = seq_lens[bz_4];
                    int g_len_bz_4 = kv_len_global[bz_4];
                    int rows_bz_4 = (q_end_bz_4 - q_start_bz_4) * num_heads;
                    int pieces_4 = (rows_bz_4 + 128 - 1) / 128 * num_split;
                    int left_4 = pieces_4 - cx_4;
                    int n_pieces_raw_4 = (left_4 + n_clusters_4 - 1) / n_clusters_4;
                    int n_pieces_4 = ((left_4 <= 0) ? 0 : n_pieces_raw_4);
                    #pragma unroll 1
                    for (int pi_m = 0; pi_m < n_pieces_4; pi_m++) {
                        int p_m = cx_4 + pi_m * n_clusters_4;
                        int b_4 = blockIdx.z;
                        int split_idx_4 = p_m % num_split;
                        int rows_b0_4 = (q_end_bz_4 - q_start_bz_4) * num_heads;
                        int m_tiles_b_4 = (rows_b0_4 + 128 - 1) / 128;
                        int m_tile_4 = m_tiles_b_4 - 1 - p_m / num_split;
                        int q_start_4 = q_start_bz_4;
                        int q_len_b_4 = q_end_bz_4 - q_start_bz_4;
                        int kv_len_4 = kv_len_bz_4;
                        int g_len_4 = g_len_bz_4;
                        int rows_b_4 = q_len_b_4 * num_heads;
                        int row0_4 = m_tile_4 * 128;
                        int rows_left_4 = rows_b_4 - row0_4;
                        int rows_pos_4 = ((rows_left_4 < 0) ? 0 : rows_left_4);
                        int rows_valid_4 = ((rows_pos_4 > 128) ? 128 : rows_pos_4);
                        int row_base_global_4 = q_start_4 * num_heads + row0_4;
                        int last_row_4 = row0_4 + rows_valid_4 - 1;
                        int t_last_4 = last_row_4 / num_heads;
                        int num_4 = g_len_4 - q_len_b_4 + t_last_4 - cp_rank;
                        int vis_raw_5 = num_4 / cp_world + 1;
                        int vis_cap_5 = ((vis_raw_5 > kv_len_4) ? kv_len_4 : vis_raw_5);
                        int vis_out_5 = ((num_4 < 0) ? 0 : vis_cap_5);
                        int kv_end_raw_4 = vis_out_5;
                        int kv_end_4 = ((rows_valid_4 == 0) ? 0 : kv_end_raw_4);
                        int n_tiles_total_4 = (kv_end_4 + 128 - 1) / 128;
                        int tiles_per_split_4 = (n_tiles_total_4 + num_split - 1) / num_split;
                        int my_start_raw_4 = split_idx_4 * tiles_per_split_4;
                        int my_end_raw_4 = my_start_raw_4 + tiles_per_split_4;
                        int my_end_4 = ((my_end_raw_4 > n_tiles_total_4) ? n_tiles_total_4 : my_end_raw_4);
                        int my_n_raw_4 = my_end_4 - my_start_raw_4;
                        int empty_4 = ((my_n_raw_4 < 1) ? 1 : 0);
                        int my_n_tiles_4 = ((my_n_raw_4 < 1) ? 1 : my_n_raw_4);
                        int my_start_4 = ((my_n_raw_4 < 1) ? 0 : my_start_raw_4);
                        int pt_base_4 = b_4 * max_pages_per_seq;
                        mbarrier_wait(q_full_addr, _phase_q_full_0_1);
                        _phase_q_full_0_1 ^= 1;
                        mbarrier_wait(sfa_full_addr, _phase_sfa_full_0);
                        _phase_sfa_full_0 ^= 1;
                        #pragma unroll 1
                        for (int it_m = 0; it_m < my_n_tiles_4; it_m++) {
                            {
                                mbarrier_wait(raw_full_addr + (raw_stage_m) * 8, raw_phase_m);
                                mbarrier_wait(sf_full_addr, _phase_sf_full_0);
                                _phase_sf_full_0 ^= 1;
                                mbarrier_wait(s_empty_addr, _phase_s_empty_0);
                                _phase_s_empty_0 ^= 1;
                                asm volatile("tcgen05.fence::after_thread_sync;");
                            }
                            int _mma_b_lo_0 = (((smem_v14_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 2048;
                            {
                                uint64_t b_desc = ((uint64_t)_mma_b_lo_0) | ((uint64_t)0x80004020 << 32);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_acc), "r"(tmem_tmem_q), "l"(b_desc), "r"(0x8100480U), "r"(tmem_sfa0), "r"(tmem_sfb0_0), "r"(0) : "memory");
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_acc), "r"((tmem_tmem_q + 8)), "l"((b_desc + 2)), "r"(0x8100480U), "r"((tmem_sfa0 + 4)), "r"((tmem_sfb0_0 + 4)), "r"(1) : "memory");
                            }
                            int _mma_b_lo_1 = (((smem_v15_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 2048;
                            {
                                uint64_t b_desc = ((uint64_t)_mma_b_lo_1) | ((uint64_t)0x80004020 << 32);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_acc), "r"(tmem_tmem_q + 16), "l"(b_desc), "r"(0x8100480U), "r"(tmem_sfa1), "r"(tmem_sfb0_1), "r"(1) : "memory");
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_acc), "r"((tmem_tmem_q + 16 + 8)), "l"((b_desc + 2)), "r"(0x8100480U), "r"((tmem_sfa1 + 4)), "r"((tmem_sfb0_1 + 4)), "r"(1) : "memory");
                            }
                            int _mma_b_lo_2 = (((smem_v16_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 2048;
                            {
                                uint64_t b_desc = ((uint64_t)_mma_b_lo_2) | ((uint64_t)0x80004020 << 32);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_acc), "r"(tmem_tmem_q + 32), "l"(b_desc), "r"(0x8100480U), "r"(tmem_sfa2), "r"(tmem_sfb0_2), "r"(1) : "memory");
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_acc), "r"((tmem_tmem_q + 32 + 8)), "l"((b_desc + 2)), "r"(0x8100480U), "r"((tmem_sfa2 + 4)), "r"((tmem_sfb0_2 + 4)), "r"(1) : "memory");
                            }
                            int _mma_b_lo_3 = (((smem_v17_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 2048;
                            {
                                uint64_t b_desc = ((uint64_t)_mma_b_lo_3) | ((uint64_t)0x80004020 << 32);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_acc), "r"(tmem_tmem_q + 48), "l"(b_desc), "r"(0x8100480U), "r"(tmem_sfa3), "r"(tmem_sfb0_3), "r"(1) : "memory");
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_acc), "r"((tmem_tmem_q + 48 + 8)), "l"((b_desc + 2)), "r"(0x8100480U), "r"((tmem_sfa3 + 4)), "r"((tmem_sfb0_3 + 4)), "r"(1) : "memory");
                            }
                            int _mma_a_lo_4 = ((smem_qr_addr) >> 4) & 0x3FFF;
                            int _mma_b_lo_4 = (((smem_v28_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 2048;
                            asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 135266320;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"(tmem_tmem_acc), "r"(1));
                            int _mma_b_lo_5 = (((smem_v18_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 2048;
                            {
                                uint64_t b_desc = ((uint64_t)_mma_b_lo_5) | ((uint64_t)0x80004020 << 32);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_acc + (32))), "r"(tmem_tmem_q), "l"(b_desc), "r"(0x8100480U), "r"(tmem_sfa0), "r"(tmem_sfb1_0), "r"(0) : "memory");
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_acc + (32))), "r"((tmem_tmem_q + 8)), "l"((b_desc + 2)), "r"(0x8100480U), "r"((tmem_sfa0 + 4)), "r"((tmem_sfb1_0 + 4)), "r"(1) : "memory");
                            }
                            int _mma_b_lo_6 = (((smem_v19_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 2048;
                            {
                                uint64_t b_desc = ((uint64_t)_mma_b_lo_6) | ((uint64_t)0x80004020 << 32);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_acc + (32))), "r"(tmem_tmem_q + 16), "l"(b_desc), "r"(0x8100480U), "r"(tmem_sfa1), "r"(tmem_sfb1_1), "r"(1) : "memory");
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_acc + (32))), "r"((tmem_tmem_q + 16 + 8)), "l"((b_desc + 2)), "r"(0x8100480U), "r"((tmem_sfa1 + 4)), "r"((tmem_sfb1_1 + 4)), "r"(1) : "memory");
                            }
                            int _mma_b_lo_7 = (((smem_v20_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 2048;
                            {
                                uint64_t b_desc = ((uint64_t)_mma_b_lo_7) | ((uint64_t)0x80004020 << 32);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_acc + (32))), "r"(tmem_tmem_q + 32), "l"(b_desc), "r"(0x8100480U), "r"(tmem_sfa2), "r"(tmem_sfb1_2), "r"(1) : "memory");
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_acc + (32))), "r"((tmem_tmem_q + 32 + 8)), "l"((b_desc + 2)), "r"(0x8100480U), "r"((tmem_sfa2 + 4)), "r"((tmem_sfb1_2 + 4)), "r"(1) : "memory");
                            }
                            int _mma_b_lo_8 = (((smem_v21_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 2048;
                            {
                                uint64_t b_desc = ((uint64_t)_mma_b_lo_8) | ((uint64_t)0x80004020 << 32);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_acc + (32))), "r"(tmem_tmem_q + 48), "l"(b_desc), "r"(0x8100480U), "r"(tmem_sfa3), "r"(tmem_sfb1_3), "r"(1) : "memory");
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_acc + (32))), "r"((tmem_tmem_q + 48 + 8)), "l"((b_desc + 2)), "r"(0x8100480U), "r"((tmem_sfa3 + 4)), "r"((tmem_sfb1_3 + 4)), "r"(1) : "memory");
                            }
                            int _mma_b_lo_9 = (((smem_v29_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 2048;
                            asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 135266320;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_9), "r"((tmem_tmem_acc + (32))), "r"(1));
                            {
                                tcgen05_commit_cg2_multicast(s_full_addr, (uint16_t)(3));
                                tcgen05_commit_cg2_multicast(sf_empty_addr, (uint16_t)(3));
                                tcgen05_commit_cg2_multicast(raw_empty_addr + (raw_stage_m) * 8, (uint16_t)(3));
                                raw_stage_m += 1;
                                if (raw_stage_m == 3) { raw_stage_m = 0; raw_phase_m ^= 1; }
                            }
                            if (it_m >= 1) {
                                if (it_m - 1 == 0) {
                                    mbarrier_wait(o_empty_addr, _phase_o_empty_0);
                                    _phase_o_empty_0 ^= 1;
                                }
                                int first_pv_flag = ((it_m - 1 == 0) ? 1 : 0);
                                mbarrier_wait(v_full_addr + (v_stage_m) * 8, v_phase_m);
                                mbarrier_wait(p_full_addr + (p_stage_m) * 8, p_phase_m);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                int _mma_a_lo_10 = (((smem_p_addr) >> 4) & 0x3FFF) + (p_stage_m) * 512;
                                int _mma_b_lo_10 = ((((smem_v30_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage_m) * 2048;
                                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 138477584;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_10), "r"(_mma_b_lo_10), "r"((tmem_tmem_acc + (64))), "r"(((first_pv_flag) ? 0 : 1)));
                                int _mma_a_lo_11 = (((smem_p_addr) >> 4) & 0x3FFF) + (p_stage_m) * 512;
                                int _mma_b_lo_11 = ((((smem_v31_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage_m) * 2048;
                                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 138477584;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_11), "r"(_mma_b_lo_11), "r"((tmem_tmem_acc + (192))), "r"(((first_pv_flag) ? 0 : 1)));
                                tcgen05_commit_cg2_multicast(pv_done_addr, (uint16_t)(3));
                                tcgen05_commit_cg2_multicast(p_empty_addr + (p_stage_m) * 8, (uint16_t)(3));
                                tcgen05_commit_cg2_multicast(v_empty_addr + (v_stage_m) * 8, (uint16_t)(3));
                                p_stage_m += 1;
                                if (p_stage_m == 2) { p_stage_m = 0; p_phase_m ^= 1; }
                                v_stage_m += 1;
                                if (v_stage_m == 2) { v_stage_m = 0; v_phase_m ^= 1; }
                                if (it_m - 1 + 1 == my_n_tiles_4) {
                                    tcgen05_commit_cg2_multicast(o_full_addr, (uint16_t)(3));
                                }
                            }
                        }
                        tcgen05_commit_cg2_multicast(q_empty_addr, (uint16_t)(3));
                        if (my_n_tiles_4 - 1 == 0) {
                            mbarrier_wait(o_empty_addr, _phase_o_empty_0);
                            _phase_o_empty_0 ^= 1;
                        }
                        int first_pv_flag_1 = ((my_n_tiles_4 - 1 == 0) ? 1 : 0);
                        mbarrier_wait(v_full_addr + (v_stage_m) * 8, v_phase_m);
                        mbarrier_wait(p_full_addr + (p_stage_m) * 8, p_phase_m);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_12 = (((smem_p_addr) >> 4) & 0x3FFF) + (p_stage_m) * 512;
                        int _mma_b_lo_12 = ((((smem_v30_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage_m) * 2048;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 138477584;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_12), "r"(_mma_b_lo_12), "r"((tmem_tmem_acc + (64))), "r"(((first_pv_flag_1) ? 0 : 1)));
                        int _mma_a_lo_13 = (((smem_p_addr) >> 4) & 0x3FFF) + (p_stage_m) * 512;
                        int _mma_b_lo_13 = ((((smem_v31_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage_m) * 2048;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 138477584;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_13), "r"(_mma_b_lo_13), "r"((tmem_tmem_acc + (192))), "r"(((first_pv_flag_1) ? 0 : 1)));
                        tcgen05_commit_cg2_multicast(pv_done_addr, (uint16_t)(3));
                        tcgen05_commit_cg2_multicast(p_empty_addr + (p_stage_m) * 8, (uint16_t)(3));
                        tcgen05_commit_cg2_multicast(v_empty_addr + (v_stage_m) * 8, (uint16_t)(3));
                        p_stage_m += 1;
                        if (p_stage_m == 2) { p_stage_m = 0; p_phase_m ^= 1; }
                        v_stage_m += 1;
                        if (v_stage_m == 2) { v_stage_m = 0; v_phase_m ^= 1; }
                        if (my_n_tiles_4 - 1 + 1 == my_n_tiles_4) {
                            tcgen05_commit_cg2_multicast(o_full_addr, (uint16_t)(3));
                        }
                    }
                }
            }
        }
    }

    // Kernel teardown ops
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
