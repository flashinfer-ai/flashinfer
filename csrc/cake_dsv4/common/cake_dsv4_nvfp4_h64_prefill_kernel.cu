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

// Common preamble (typedefs, tensor-map ABI) shared by this export's prefill kernels.
#include "cake_dsv4_nvfp4_prefill_device_common.cuh"


#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_TMEM_SCRATCH_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 384
#define TMEM_TMEM_SFB_OFFSET 416
#define NUM_INDEX_PIPE_STAGES 6
#define NUM_WORK_PIPE_STAGES 2
#define NUM_THROTTLE_PIPE_STAGES 2
#define SMEM_SMEM_Q_A_OFF 1024
#define SMEM_SMEM_Q_A_STAGE_BYTES 16384
#define SMEM_SMEM_Q_A_STRIDE 16384
#define SMEM_SMEM_Q_B_OFF 17408
#define SMEM_SMEM_Q_B_STAGE_BYTES 16384
#define SMEM_SMEM_Q_B_STRIDE 16384
#define SMEM_SMEM_Q_ROPE_OFF 33792
#define SMEM_SMEM_Q_ROPE_STAGE_BYTES 16384
#define SMEM_SMEM_Q_ROPE_STRIDE 16384
#define SMEM_SMEM_Q_SF_OFF 50176
#define SMEM_SMEM_Q_SF_STAGE_BYTES 2048
#define SMEM_SMEM_Q_SF_STRIDE 2048
#define SMEM_SMEM_K_A_OFF 52224
#define SMEM_SMEM_K_A_STAGE_BYTES 8192
#define SMEM_SMEM_K_A_STRIDE 26624
#define SMEM_SMEM_K_B_OFF 60416
#define SMEM_SMEM_K_B_STAGE_BYTES 8192
#define SMEM_SMEM_K_B_STRIDE 26624
#define SMEM_SMEM_K_C_OFF 68608
#define SMEM_SMEM_K_C_STAGE_BYTES 8192
#define SMEM_SMEM_K_C_STRIDE 26624
#define SMEM_SMEM_K_SF_OFF 76800
#define SMEM_SMEM_K_SF_STAGE_BYTES 2048
#define SMEM_SMEM_K_SF_STRIDE 26624
#define SMEM_SMEM_V_OFF 132096
#define SMEM_SMEM_V_STAGE_BYTES 32768
#define SMEM_SMEM_V_STRIDE 32768
#define SMEM_SMEM_P_OFF 197632
#define SMEM_SMEM_P_STAGE_BYTES 4096
#define SMEM_SMEM_P_STRIDE 4096
#define SMEM_SMEM_INDICES_OFF 205824
#define SMEM_SMEM_INDICES_STAGE_BYTES 1024
#define SMEM_SMEM_INDICES_STRIDE 1024
#define SMEM_SMEM_SCALE_OFF 211968
#define SMEM_SMEM_SCALE_STAGE_BYTES 1024
#define SMEM_SMEM_SCALE_STRIDE 1024
#define SMEM_SMEM_EXCH_OFF 212992
#define SMEM_SMEM_EXCH_STAGE_BYTES 1024
#define SMEM_SMEM_EXCH_STRIDE 1024
#define SMEM_SMEM_SUM_OFF 214016
#define SMEM_SMEM_SUM_STAGE_BYTES 512
#define SMEM_SMEM_SUM_STRIDE 512
#define SMEM_SMEM_FMAX_OFF 214528
#define SMEM_SMEM_FMAX_STAGE_BYTES 512
#define SMEM_SMEM_FMAX_STRIDE 512
#define SMEM_SMEM_KEXP_OFF 215040
#define SMEM_SMEM_KEXP_STAGE_BYTES 32
#define SMEM_SMEM_KEXP_STRIDE 32
#define SMEM_WORK_RESPONSE_OFF 215072
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_TOTAL 215168
#define THREADS 768
#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>
#include "cake_device_helpers/cake_elect_sync_1ee8cd91025d8932.cuh"
#include "cake_device_helpers/cake_mbarrier_init_15e581aef85ee586.cuh"
#include "cake_device_helpers/cake_mbarrier_wait_c93bf7e39b218733.cuh"
#include "cake_device_helpers/cake_tcgen05_mma_mxf4nvf4_bs_82eb5079722601c2.cuh"
#include "cake_device_helpers/cake_tcgen05_mma_f16_03fde8c1cbf058c8.cuh"
union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};
#include "cake_device_helpers/cake_incr_smem_desc_lo_4429b045244f74bc.cuh"
#include "cake_device_helpers/cake_elect_commit_8131fdc67daf4d23.cuh"
#include "cake_device_helpers/cake_mbarrier_arrive_71ef360c3ac78eb7.cuh"
#include "cake_device_helpers/cake_mbarrier_arrive_expect_tx_54ce59d489f69526.cuh"
#include "cake_device_helpers/cake_tmem_ld_x16_a9ed61940fdba296.cuh"
#include "cake_device_helpers/cake_tmem_st_x16_f32_44948ad68a2365fa.cuh"
#include "cake_device_helpers/cake_approx_exp2_04487d25211b6cf4.cuh"
#include "cake_device_helpers/cake_approx_rcp_8010e011dc029034.cuh"
#include "cake_device_helpers/cake_max_noftz_a41d70352450a2c9.cuh"
#include "cake_device_helpers/cake_row_max_reduce_7d57055e9b222006.cuh"
#include "cake_device_helpers/cake_mul_f32x2_noftz_inplace_8c326b4fb12f27f0.cuh"
#include "cake_device_helpers/cake_add_f32x2_74578aa6314ab32b.cuh"
#include "cake_device_helpers/cake_fma_f32x2_35fce702230b2007.cuh"
#include "cake_device_helpers/cake_tma_3d_gmem2smem_7df1e07811f83fc6.cuh"
#include "cake_device_helpers/cake_tma_gather4_gmem2smem_3b819e678c50bc52.cuh"
#include "cake_device_helpers/cake_make_warp_uniform_26e432f3ff129648.cuh"
#include "cake_device_helpers/cake_qmul4_broadcast_346d0cff0a3e7d7b.cuh"
#include "cake_device_helpers/cake_qmul4_lower4_064647e7d3b6ba3a.cuh"
#include "cake_device_helpers/cake_qmul4_higher4_6a55113d4db2f308.cuh"

extern "C" {

__global__ __launch_bounds__(768, LAUNCH_MIN_BLOCKS) void
kernel_cake_dsv4_nvfp4_h64_prefill(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_q_sf, const __grid_constant__ CUtensorMap tmap_swa_kv, const __grid_constant__ CUtensorMap tmap_compressed_kv, const __grid_constant__ CUtensorMap tmap_swa_sf, const __grid_constant__ CUtensorMap tmap_compressed_sf, __nv_bfloat16* __restrict__ O, float* __restrict__ LSE, int* __restrict__ swa_indices, int* __restrict__ compressed_indices, int* __restrict__ sparse_topk_lens, float* __restrict__ sinks, float* __restrict__ bmm1_scale, float* __restrict__ bmm2_scale, int num_heads, int swa_index_stride, int compressed_index_stride, int sparse_topk_lens_offset, int num_query_tokens, int has_sinks, int swa_page_log2, int swa_pitch_units, int swa_footer_units, int compressed_page_log2, int compressed_pitch_units, int compressed_footer_units, int swa_width, int compressed_width, int* __restrict__ extra_topk_lens)
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
    #define q_load_full_addr (mbar_base + 0)
    #define q_full_addr (mbar_base + 8)
    #define q_empty_addr (mbar_base + 16)
    #define index_full_addr (mbar_base + 24)
    #define index_empty_addr (mbar_base + 72)
    #define k_full_addr (mbar_base + 120)
    #define k_empty_addr (mbar_base + 144)
    #define kstage_free_addr (mbar_base + 168)
    #define sf_full_addr (mbar_base + 192)
    #define kexp_full_addr (mbar_base + 216)
    #define v_full_addr (mbar_base + 232)
    #define v_empty_addr (mbar_base + 248)
    #define s_full_addr (mbar_base + 264)
    #define s_empty_addr (mbar_base + 280)
    #define p_full_addr (mbar_base + 296)
    #define p_recycle_addr (mbar_base + 312)
    #define stats_addr (mbar_base + 328)
    #define o_empty_addr (mbar_base + 344)
    #define o_full_addr (mbar_base + 360)
    #define sum_ready_addr (mbar_base + 376)
    #define sum_empty_addr (mbar_base + 384)
    #define tmem_dealloc_addr (mbar_base + 392)
    #define work_full_addr (mbar_base + 400)
    #define work_empty_addr (mbar_base + 416)
    #define throttle_full_addr (mbar_base + 432)
    #define throttle_empty_addr (mbar_base + 448)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* smem_q_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_q_a_addr = smem + 1024;
    uint8_t* smem_q_b = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_q_b_addr = smem + 17408;
    __nv_bfloat16* smem_q_rope = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int smem_q_rope_addr = smem + 33792;
    unsigned int* smem_q_sf = reinterpret_cast<unsigned int*>(smem_raw + 50176);
    const int smem_q_sf_addr = smem + 50176;
    uint8_t* smem_k_a = reinterpret_cast<uint8_t*>(smem_raw + 52224);
    const int smem_k_a_addr = smem + 52224;
    uint8_t* smem_k_b = reinterpret_cast<uint8_t*>(smem_raw + 60416);
    const int smem_k_b_addr = smem + 60416;
    __nv_bfloat16* smem_k_c = reinterpret_cast<__nv_bfloat16*>(smem_raw + 68608);
    const int smem_k_c_addr = smem + 68608;
    unsigned int* smem_k_sf = reinterpret_cast<unsigned int*>(smem_raw + 76800);
    const int smem_k_sf_addr = smem + 76800;
    uint8_t* smem_v = reinterpret_cast<uint8_t*>(smem_raw + 132096);
    const int smem_v_addr = smem + 132096;
    uint8_t* smem_p = reinterpret_cast<uint8_t*>(smem_raw + 197632);
    const int smem_p_addr = smem + 197632;
    int* smem_indices = reinterpret_cast<int*>(smem_raw + 205824);
    const int smem_indices_addr = smem + 205824;
    float* smem_scale = reinterpret_cast<float*>(smem_raw + 211968);
    const int smem_scale_addr = smem + 211968;
    float* smem_exch = reinterpret_cast<float*>(smem_raw + 212992);
    const int smem_exch_addr = smem + 212992;
    float* smem_sum = reinterpret_cast<float*>(smem_raw + 214016);
    const int smem_sum_addr = smem + 214016;
    float* smem_fmax = reinterpret_cast<float*>(smem_raw + 214528);
    const int smem_fmax_addr = smem + 214528;
    unsigned int* smem_kexp = reinterpret_cast<unsigned int*>(smem_raw + 215040);
    const int smem_kexp_addr = smem + 215040;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 215072);
    const int work_response_addr = smem + 215072;

    // Mbarrier init (26 pipeline groups, 0 ordered-sequence groups, 58 barriers)
    // Mbarriers at smem_raw[0..464)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_load_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            // --- pipeline 'index_pipe' ---
            // index_full: 6 barriers, init_count=32
            mbarrier_init(smem + 24, 32);
            mbarrier_init(smem + 32, 32);
            mbarrier_init(smem + 40, 32);
            mbarrier_init(smem + 48, 32);
            mbarrier_init(smem + 56, 32);
            mbarrier_init(smem + 64, 32);
            // index_empty: 6 barriers, init_count=256
            mbarrier_init(smem + 72, 256);
            mbarrier_init(smem + 80, 256);
            mbarrier_init(smem + 88, 256);
            mbarrier_init(smem + 96, 256);
            mbarrier_init(smem + 104, 256);
            mbarrier_init(smem + 112, 256);
            // k_full: 3 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // k_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            // kstage_free: 3 barriers, init_count=2
            mbarrier_init(smem + 168, 2);
            mbarrier_init(smem + 176, 2);
            mbarrier_init(smem + 184, 2);
            // sf_full: 3 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            // kexp_full: 2 barriers, init_count=1
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            // v_full: 2 barriers, init_count=2
            mbarrier_init(smem + 232, 2);
            mbarrier_init(smem + 240, 2);
            // v_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 248, 1);
            mbarrier_init(smem + 256, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            // s_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 280, 128);
            mbarrier_init(smem + 288, 128);
            // p_full: 2 barriers, init_count=128
            mbarrier_init(smem + 296, 128);
            mbarrier_init(smem + 304, 128);
            // p_recycle: 2 barriers, init_count=1
            mbarrier_init(smem + 312, 1);
            mbarrier_init(smem + 320, 1);
            // stats: 2 barriers, init_count=128
            mbarrier_init(smem + 328, 128);
            mbarrier_init(smem + 336, 128);
            // o_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 344, 128);
            mbarrier_init(smem + 352, 128);
            // o_full: 2 barriers, init_count=1
            mbarrier_init(smem + 360, 1);
            mbarrier_init(smem + 368, 1);
            // sum_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 376, 128);
            // sum_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 384, 128);
            // tmem_dealloc: 1 barriers, init_count=448
            mbarrier_init(smem + 392, 448);
            // --- pipeline 'work_pipe' ---
            // work_full: 2 barriers, init_count=1
            mbarrier_init(smem + 400, 1);
            mbarrier_init(smem + 408, 1);
            // work_empty: 2 barriers, init_count=736
            mbarrier_init(smem + 416, 736);
            mbarrier_init(smem + 424, 736);
            // --- pipeline 'throttle_pipe' ---
            // throttle_full: 2 barriers, init_count=128
            mbarrier_init(smem + 432, 128);
            mbarrier_init(smem + 440, 128);
            // throttle_empty: 2 barriers, init_count=32
            mbarrier_init(smem + 448, 32);
            mbarrier_init(smem + 456, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 464);
    if (warp == 0) {
        int _tmem_hold = smem + 464;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_scratch = taddr;
    const int tmem_tmem_sfa = taddr + 384;
    const int tmem_tmem_sfb = taddr + 416;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    }

    // ---- Role: softmax_wg ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 120;");
        { // softmax_wg_main
            float softmax_scale_log2 = bmm1_scale[0] * 1.4426950408889634f;
            const int sm_row_base = warp % 4 * 32;
            const int sm_thread = sm_row_base + lane;
            const int sm_head = sm_thread & 63;
            const int sm_half = sm_thread >> 6;
            int sm_tile_cursor = 0;
            int sm_item = 0;
            unsigned int sm_index_stage = 0;
            unsigned int sm_work_stage = 0;
            int sm_query = blockIdx.x;
            unsigned int _phase_index_full = 0;
            unsigned int _phase_work_full = 0;
            #pragma unroll 1
            for (unsigned int sm_work = 0; sm_work < 1048576; sm_work++) {
                if (sm_query < num_query_tokens) {
                    int _max_18 = ((sparse_topk_lens[sm_query] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[sm_query] + sparse_topk_lens_offset) : (0));
                    int _min_12 = ((_max_18) < (swa_width) ? (_max_18) : (swa_width));
                    int sm_main_active = _min_12;
                    int sm_extra_active = 0;
                    if (compressed_width > 0) {
                        int _max_19 = ((extra_topk_lens[sm_query]) > (0) ? (extra_topk_lens[sm_query]) : (0));
                        int _min_13 = ((_max_19) < (compressed_width) ? (_max_19) : (compressed_width));
                        sm_extra_active = _min_13;
                    }
                    int sm_n_main = (sm_main_active + 64 - 1) / 64;
                    int sm_n_extra = (sm_extra_active + 64 - 1) / 64;
                    int _max_20 = ((sm_n_main + sm_n_extra) > (1) ? (sm_n_main + sm_n_extra) : (1));
                    int sm_tiles = _max_20;
                    int q_load_full_stage = 0;
                    int q_load_full_phase = sm_item & 1;
                    mbarrier_wait(q_load_full_addr + (q_load_full_stage) * 8, q_load_full_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    unsigned int sfa_regs[32];
                    #pragma unroll
                    for (int sfa_s = 0; sfa_s < 7; sfa_s++) {
                        sfa_regs[4 * sfa_s] = smem_q_sf[lane * 8 + sfa_s];
                        sfa_regs[4 * sfa_s + 1] = smem_q_sf[(lane + 32) * 8 + sfa_s];
                        sfa_regs[4 * sfa_s + 2] = sfa_regs[4 * sfa_s];
                        sfa_regs[4 * sfa_s + 3] = sfa_regs[4 * sfa_s + 1];
                    }
                    #pragma unroll
                    for (int sfa_pad = 28; sfa_pad < 32; sfa_pad++) {
                        sfa_regs[sfa_pad] = 0;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(taddr + 384 + (unsigned int)(sm_row_base << 16)), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[0])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[1])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[2])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[3])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[4])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[5])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[6])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[7])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[8])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[9])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[10])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[11])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[12])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[13])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[14])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[15])));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(taddr + 384 + 16 + (unsigned int)(sm_row_base << 16)), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[0])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[1])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[2])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[3])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[4])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[5])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[6])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[7])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[8])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[9])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[10])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[11])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[12])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[13])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[14])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[15])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    if (warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive(q_full_addr);
                        }
                    }
                    float row_max_val = -CUDART_INF_F;
                    float row_sum_val = 0.0f;
                    if (has_sinks != 0 && sm_head < num_heads) {
                        row_max_val = sinks[sm_head] * 1.4426950408889634f / softmax_scale_log2;
                        row_sum_val = ((sm_half == 0) ? 1.0f : 0.0f);
                    }
                    #pragma unroll 1
                    for (int sm_tile = 0; sm_tile < sm_tiles; sm_tile++) {
                        int st = sm_tile_cursor + sm_tile;
                        int s_stage = st & 1;
                        int s_phase = st >> 1 & 1;
                        int sm_slot_tile = sm_tile & 3;
                        if (sm_slot_tile == 0) {
                            mbarrier_wait(index_full_addr + (sm_index_stage) * 8, _phase_index_full);
                        }
                        int seg_in_main = sm_n_main > sm_tile || sm_n_extra == 0;
                        int seg_active = ((seg_in_main != 0) ? sm_main_active : sm_extra_active);
                        int seg_tile = ((seg_in_main != 0) ? sm_tile : sm_tile - sm_n_main);
                        int _max_21 = ((seg_active - seg_tile * 64 - sm_half * 32) > (0) ? (seg_active - seg_tile * 64 - sm_half * 32) : (0));
                        int _min_14 = ((_max_21) < (32) ? (_max_21) : (32));
                        int valid_cols = _min_14;
                        int staged_index[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&staged_index[0])) : "r"(smem_indices_addr + sm_index_stage * 1024 + (unsigned int)((sm_slot_tile * 64 + sm_half * 32 + lane) * 4)));
                        unsigned int _vote_3 = __ballot_sync(0xFFFFFFFF, staged_index[0] < 0);
                        unsigned int invalid_cols = _vote_3;
                        if (sm_slot_tile == 3 || sm_tile + 1 == sm_tiles) {
                            mbarrier_arrive(index_empty_addr + (sm_index_stage) * 8);
                            sm_index_stage += 1;
                            if (sm_index_stage == 6) { sm_index_stage = 0; _phase_index_full ^= 1; }
                        }
                        mbarrier_wait(s_full_addr + (s_stage) * 8, s_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        float sv[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(sv[0]), "=f"(sv[1]), "=f"(sv[2]), "=f"(sv[3]), "=f"(sv[4]), "=f"(sv[5]), "=f"(sv[6]), "=f"(sv[7]), "=f"(sv[8]), "=f"(sv[9]), "=f"(sv[10]), "=f"(sv[11]), "=f"(sv[12]), "=f"(sv[13]), "=f"(sv[14]), "=f"(sv[15]), "=f"(sv[16]), "=f"(sv[17]), "=f"(sv[18]), "=f"(sv[19]), "=f"(sv[20]), "=f"(sv[21]), "=f"(sv[22]), "=f"(sv[23]), "=f"(sv[24]), "=f"(sv[25]), "=f"(sv[26]), "=f"(sv[27]), "=f"(sv[28]), "=f"(sv[29]), "=f"(sv[30]), "=f"(sv[31])
                            : "r"(taddr + 256 + (unsigned int)(s_stage * 64) + (unsigned int)(sm_half * 32) + (unsigned int)(sm_row_base << 16)));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(s_empty_addr + (s_stage) * 8);
                        if (valid_cols < 32) {
                            uint32_t _slice_lo_mask_0;
                            {
                                int _lim_0 = valid_cols;
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
                            if (!(_slice_lo_mask_0 & (1u << 0))) sv[0] = -CUDART_INF_F;
                            if (!(_slice_lo_mask_0 & (1u << 1))) sv[1] = -CUDART_INF_F;
                            if (!(_slice_lo_mask_0 & (1u << 2))) sv[2] = -CUDART_INF_F;
                            if (!(_slice_lo_mask_0 & (1u << 3))) sv[3] = -CUDART_INF_F;
                            if (!(_slice_lo_mask_0 & (1u << 4))) sv[4] = -CUDART_INF_F;
                            if (!(_slice_lo_mask_0 & (1u << 5))) sv[5] = -CUDART_INF_F;
                            if (!(_slice_lo_mask_0 & (1u << 6))) sv[6] = -CUDART_INF_F;
                            if (!(_slice_lo_mask_0 & (1u << 7))) sv[7] = -CUDART_INF_F;
                            if (!(_slice_lo_mask_0 & (1u << 8))) sv[8] = -CUDART_INF_F;
                            if (!(_slice_lo_mask_0 & (1u << 9))) sv[9] = -CUDART_INF_F;
                            if (!(_slice_lo_mask_0 & (1u << 10))) sv[10] = -CUDART_INF_F;
                            if (!(_slice_lo_mask_0 & (1u << 11))) sv[11] = -CUDART_INF_F;
                            if (!(_slice_lo_mask_0 & (1u << 12))) sv[12] = -CUDART_INF_F;
                            if (!(_slice_lo_mask_0 & (1u << 13))) sv[13] = -CUDART_INF_F;
                            if (!(_slice_lo_mask_0 & (1u << 14))) sv[14] = -CUDART_INF_F;
                            if (!(_slice_lo_mask_0 & (1u << 15))) sv[15] = -CUDART_INF_F;
                            if (!(_slice_lo_mask_0 & (1u << 16))) sv[16] = -CUDART_INF_F;
                            if (!(_slice_lo_mask_0 & (1u << 17))) sv[17] = -CUDART_INF_F;
                            if (!(_slice_lo_mask_0 & (1u << 18))) sv[18] = -CUDART_INF_F;
                            if (!(_slice_lo_mask_0 & (1u << 19))) sv[19] = -CUDART_INF_F;
                            if (!(_slice_lo_mask_0 & (1u << 20))) sv[20] = -CUDART_INF_F;
                            if (!(_slice_lo_mask_0 & (1u << 21))) sv[21] = -CUDART_INF_F;
                            if (!(_slice_lo_mask_0 & (1u << 22))) sv[22] = -CUDART_INF_F;
                            if (!(_slice_lo_mask_0 & (1u << 23))) sv[23] = -CUDART_INF_F;
                            if (!(_slice_lo_mask_0 & (1u << 24))) sv[24] = -CUDART_INF_F;
                            if (!(_slice_lo_mask_0 & (1u << 25))) sv[25] = -CUDART_INF_F;
                            if (!(_slice_lo_mask_0 & (1u << 26))) sv[26] = -CUDART_INF_F;
                            if (!(_slice_lo_mask_0 & (1u << 27))) sv[27] = -CUDART_INF_F;
                            if (!(_slice_lo_mask_0 & (1u << 28))) sv[28] = -CUDART_INF_F;
                            if (!(_slice_lo_mask_0 & (1u << 29))) sv[29] = -CUDART_INF_F;
                            if (!(_slice_lo_mask_0 & (1u << 30))) sv[30] = -CUDART_INF_F;
                            if (!(_slice_lo_mask_0 & (1u << 31))) sv[31] = -CUDART_INF_F;
                        }
                        if (invalid_cols != 0) {
                            sv[0] = (((invalid_cols & 1) != 0) ? -CUDART_INF_F : sv[0]);
                            sv[1] = (((invalid_cols >> 1 & 1) != 0) ? -CUDART_INF_F : sv[1]);
                            sv[2] = (((invalid_cols >> 2 & 1) != 0) ? -CUDART_INF_F : sv[2]);
                            sv[3] = (((invalid_cols >> 3 & 1) != 0) ? -CUDART_INF_F : sv[3]);
                            sv[4] = (((invalid_cols >> 4 & 1) != 0) ? -CUDART_INF_F : sv[4]);
                            sv[5] = (((invalid_cols >> 5 & 1) != 0) ? -CUDART_INF_F : sv[5]);
                            sv[6] = (((invalid_cols >> 6 & 1) != 0) ? -CUDART_INF_F : sv[6]);
                            sv[7] = (((invalid_cols >> 7 & 1) != 0) ? -CUDART_INF_F : sv[7]);
                            sv[8] = (((invalid_cols >> 8 & 1) != 0) ? -CUDART_INF_F : sv[8]);
                            sv[9] = (((invalid_cols >> 9 & 1) != 0) ? -CUDART_INF_F : sv[9]);
                            sv[10] = (((invalid_cols >> 10 & 1) != 0) ? -CUDART_INF_F : sv[10]);
                            sv[11] = (((invalid_cols >> 11 & 1) != 0) ? -CUDART_INF_F : sv[11]);
                            sv[12] = (((invalid_cols >> 12 & 1) != 0) ? -CUDART_INF_F : sv[12]);
                            sv[13] = (((invalid_cols >> 13 & 1) != 0) ? -CUDART_INF_F : sv[13]);
                            sv[14] = (((invalid_cols >> 14 & 1) != 0) ? -CUDART_INF_F : sv[14]);
                            sv[15] = (((invalid_cols >> 15 & 1) != 0) ? -CUDART_INF_F : sv[15]);
                            sv[16] = (((invalid_cols >> 16 & 1) != 0) ? -CUDART_INF_F : sv[16]);
                            sv[17] = (((invalid_cols >> 17 & 1) != 0) ? -CUDART_INF_F : sv[17]);
                            sv[18] = (((invalid_cols >> 18 & 1) != 0) ? -CUDART_INF_F : sv[18]);
                            sv[19] = (((invalid_cols >> 19 & 1) != 0) ? -CUDART_INF_F : sv[19]);
                            sv[20] = (((invalid_cols >> 20 & 1) != 0) ? -CUDART_INF_F : sv[20]);
                            sv[21] = (((invalid_cols >> 21 & 1) != 0) ? -CUDART_INF_F : sv[21]);
                            sv[22] = (((invalid_cols >> 22 & 1) != 0) ? -CUDART_INF_F : sv[22]);
                            sv[23] = (((invalid_cols >> 23 & 1) != 0) ? -CUDART_INF_F : sv[23]);
                            sv[24] = (((invalid_cols >> 24 & 1) != 0) ? -CUDART_INF_F : sv[24]);
                            sv[25] = (((invalid_cols >> 25 & 1) != 0) ? -CUDART_INF_F : sv[25]);
                            sv[26] = (((invalid_cols >> 26 & 1) != 0) ? -CUDART_INF_F : sv[26]);
                            sv[27] = (((invalid_cols >> 27 & 1) != 0) ? -CUDART_INF_F : sv[27]);
                            sv[28] = (((invalid_cols >> 28 & 1) != 0) ? -CUDART_INF_F : sv[28]);
                            sv[29] = (((invalid_cols >> 29 & 1) != 0) ? -CUDART_INF_F : sv[29]);
                            sv[30] = (((invalid_cols >> 30 & 1) != 0) ? -CUDART_INF_F : sv[30]);
                            sv[31] = (((invalid_cols >> 31 & 1) != 0) ? -CUDART_INF_F : sv[31]);
                        }
                        float2 _reg_reduce_max2_1 = {-CAKE_INF, -CAKE_INF};
                        _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv[0], sv[1]));
                        _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv[2], sv[3]));
                        _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv[4], sv[5]));
                        _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv[6], sv[7]));
                        _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv[8], sv[9]));
                        _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv[10], sv[11]));
                        _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv[12], sv[13]));
                        _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv[14], sv[15]));
                        _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv[16], sv[17]));
                        _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv[18], sv[19]));
                        _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv[20], sv[21]));
                        _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv[22], sv[23]));
                        _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv[24], sv[25]));
                        _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv[26], sv[27]));
                        _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv[28], sv[29]));
                        _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv[30], sv[31]));
                        float sv_max = row_max_reduce(_reg_reduce_max2_1);
                        float local_max = sv_max;
                        smem_exch[s_stage * 128 + sm_thread] = local_max;
                        asm volatile("barrier.sync 9, 128;" ::: "memory");
                        float _max_22 = max_noftz(local_max, smem_exch[s_stage * 128 + (sm_thread ^ 64)]);
                        float _max_23 = max_noftz(_max_22, row_max_val);
                        float new_max = _max_23;
                        if (row_max_val > -CUDART_INF_F && (new_max - row_max_val) * softmax_scale_log2 <= 5.0f) {
                            new_max = row_max_val;
                        }
                        float _fma_0 = __fmaf_rn(row_max_val, softmax_scale_log2, (-new_max) * softmax_scale_log2);
                        float delta = _fma_0;
                        float _exp2_0 = approx_exp2(delta);
                        float acc_scale = ((row_max_val > -CUDART_INF_F) ? _exp2_0 : 1.0f);
                        int recycle_stage = st & 1;
                        int recycle_phase = st >> 1 & 1 ^ 1;
                        mbarrier_wait(p_recycle_addr + (recycle_stage) * 8, recycle_phase);
                        smem_scale[s_stage * 128 + sm_thread] = acc_scale;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(stats_addr + (s_stage) * 8);
                        row_max_val = new_max;
                        float safe_max = ((new_max == -CUDART_INF_F) ? 0.0f : new_max);
                        float max_scaled = safe_max * softmax_scale_log2;
                        mbarrier_wait(kexp_full_addr + (s_stage) * 8, s_phase);
                        unsigned int kexp_bits = smem_kexp[s_stage * 4];
                        float kexp_f = (float)kexp_bits;
                        float kexp_inv = __uint_as_float(127 - kexp_bits << 23);
                        const float2 _fma_b2_2 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_3 = {kexp_f - max_scaled, kexp_f - max_scaled};
                        float2 _fma_pair_4 = fma_f32x2(make_float2(sv[0], sv[1]), _fma_b2_2, _fma_c2_3);
                        sv[0] = _fma_pair_4.x;
                        sv[1] = _fma_pair_4.y;
                        float2 _fma_pair_5 = fma_f32x2(make_float2(sv[2], sv[3]), _fma_b2_2, _fma_c2_3);
                        sv[2] = _fma_pair_5.x;
                        sv[3] = _fma_pair_5.y;
                        float2 _fma_pair_6 = fma_f32x2(make_float2(sv[4], sv[5]), _fma_b2_2, _fma_c2_3);
                        sv[4] = _fma_pair_6.x;
                        sv[5] = _fma_pair_6.y;
                        float2 _fma_pair_7 = fma_f32x2(make_float2(sv[6], sv[7]), _fma_b2_2, _fma_c2_3);
                        sv[6] = _fma_pair_7.x;
                        sv[7] = _fma_pair_7.y;
                        float2 _fma_pair_8 = fma_f32x2(make_float2(sv[8], sv[9]), _fma_b2_2, _fma_c2_3);
                        sv[8] = _fma_pair_8.x;
                        sv[9] = _fma_pair_8.y;
                        float2 _fma_pair_9 = fma_f32x2(make_float2(sv[10], sv[11]), _fma_b2_2, _fma_c2_3);
                        sv[10] = _fma_pair_9.x;
                        sv[11] = _fma_pair_9.y;
                        float2 _fma_pair_10 = fma_f32x2(make_float2(sv[12], sv[13]), _fma_b2_2, _fma_c2_3);
                        sv[12] = _fma_pair_10.x;
                        sv[13] = _fma_pair_10.y;
                        float2 _fma_pair_11 = fma_f32x2(make_float2(sv[14], sv[15]), _fma_b2_2, _fma_c2_3);
                        sv[14] = _fma_pair_11.x;
                        sv[15] = _fma_pair_11.y;
                        float2 _fma_pair_12 = fma_f32x2(make_float2(sv[16], sv[17]), _fma_b2_2, _fma_c2_3);
                        sv[16] = _fma_pair_12.x;
                        sv[17] = _fma_pair_12.y;
                        float2 _fma_pair_13 = fma_f32x2(make_float2(sv[18], sv[19]), _fma_b2_2, _fma_c2_3);
                        sv[18] = _fma_pair_13.x;
                        sv[19] = _fma_pair_13.y;
                        float2 _fma_pair_14 = fma_f32x2(make_float2(sv[20], sv[21]), _fma_b2_2, _fma_c2_3);
                        sv[20] = _fma_pair_14.x;
                        sv[21] = _fma_pair_14.y;
                        float2 _fma_pair_15 = fma_f32x2(make_float2(sv[22], sv[23]), _fma_b2_2, _fma_c2_3);
                        sv[22] = _fma_pair_15.x;
                        sv[23] = _fma_pair_15.y;
                        float2 _fma_pair_16 = fma_f32x2(make_float2(sv[24], sv[25]), _fma_b2_2, _fma_c2_3);
                        sv[24] = _fma_pair_16.x;
                        sv[25] = _fma_pair_16.y;
                        float2 _fma_pair_17 = fma_f32x2(make_float2(sv[26], sv[27]), _fma_b2_2, _fma_c2_3);
                        sv[26] = _fma_pair_17.x;
                        sv[27] = _fma_pair_17.y;
                        float2 _fma_pair_18 = fma_f32x2(make_float2(sv[28], sv[29]), _fma_b2_2, _fma_c2_3);
                        sv[28] = _fma_pair_18.x;
                        sv[29] = _fma_pair_18.y;
                        float2 _fma_pair_19 = fma_f32x2(make_float2(sv[30], sv[31]), _fma_b2_2, _fma_c2_3);
                        sv[30] = _fma_pair_19.x;
                        sv[31] = _fma_pair_19.y;
                        #pragma unroll
                        for (int _le = 0; _le < 32; _le++) {
                            sv[_le] = approx_exp2(sv[_le]);
                        }
                        unsigned int _fp8_0[8];
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(sv[0]), "f"(sv[1]),
                                                   "f"(sv[2]), "f"(sv[3]));
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
                                : "=r"(_packed) : "f"(sv[4]), "f"(sv[5]),
                                                   "f"(sv[6]), "f"(sv[7]));
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
                                : "=r"(_packed) : "f"(sv[8]), "f"(sv[9]),
                                                   "f"(sv[10]), "f"(sv[11]));
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
                                : "=r"(_packed) : "f"(sv[12]), "f"(sv[13]),
                                                   "f"(sv[14]), "f"(sv[15]));
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
                                : "=r"(_packed) : "f"(sv[16]), "f"(sv[17]),
                                                   "f"(sv[18]), "f"(sv[19]));
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
                                : "=r"(_packed) : "f"(sv[20]), "f"(sv[21]),
                                                   "f"(sv[22]), "f"(sv[23]));
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
                                : "=r"(_packed) : "f"(sv[24]), "f"(sv[25]),
                                                   "f"(sv[26]), "f"(sv[27]));
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
                                : "=r"(_packed) : "f"(sv[28]), "f"(sv[29]),
                                                   "f"(sv[30]), "f"(sv[31]));
                            _fp8_0[7] = _packed;
                        }
                        int p_stage_base = smem_p_addr + (unsigned int)(s_stage * 4096);
                        #pragma unroll
                        for (int p_vec = 0; p_vec < 2; p_vec++) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p_stage_base + (sm_head * 64 + (sm_half * 32 + p_vec * 16) ^ (sm_head * 64 + (sm_half * 32 + p_vec * 16) >> 7 & 3) << 4))), "r"(_fp8_0[p_vec * 4]), "r"(_fp8_0[p_vec * 4 + 1]), "r"(_fp8_0[p_vec * 4 + 2]), "r"(_fp8_0[p_vec * 4 + 3]) : "memory");
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(p_full_addr + (s_stage) * 8);
                        float2 _reg_reduce_sum2_20 = make_float2(0.0f, 0.0f);
                        _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[0], sv[1]));
                        _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[2], sv[3]));
                        _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[4], sv[5]));
                        _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[6], sv[7]));
                        _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[8], sv[9]));
                        _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[10], sv[11]));
                        _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[12], sv[13]));
                        _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[14], sv[15]));
                        _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[16], sv[17]));
                        _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[18], sv[19]));
                        _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[20], sv[21]));
                        _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[22], sv[23]));
                        _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[24], sv[25]));
                        _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[26], sv[27]));
                        _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[28], sv[29]));
                        _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[30], sv[31]));
                        float sv_sum = _reg_reduce_sum2_20.x + _reg_reduce_sum2_20.y;
                        float block_sum = sv_sum * kexp_inv;
                        float _fma_1 = __fmaf_rn(row_sum_val, acc_scale, block_sum);
                        row_sum_val = _fma_1;
                    }
                    int sum_empty_stage = 0;
                    int sum_empty_phase = sm_item & 1 ^ 1;
                    mbarrier_wait(sum_empty_addr + (sum_empty_stage) * 8, sum_empty_phase);
                    smem_sum[sm_thread] = row_sum_val;
                    smem_fmax[sm_thread] = row_max_val;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(sum_ready_addr);
                    sm_tile_cursor = sm_tile_cursor + sm_tiles;
                    sm_item = sm_item + 1;
                }
                mbarrier_wait(work_full_addr + (sm_work_stage) * 8, _phase_work_full);
                uint32_t _clc_valid_6 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_6)
                    : "r"(work_response_addr + sm_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_6 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_6)
                    : "r"(work_response_addr + sm_work_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (sm_work_stage) * 8);
                sm_work_stage += 1;
                if (sm_work_stage == 2) { sm_work_stage = 0; _phase_work_full ^= 1; }
                if (_clc_valid_6 == 0) {
                    break;
                }
                sm_query = _clc_ctaid_6;
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: correction_wg ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 112;");
        { // correction_wg_main
            float softmax_scale_log2_1 = bmm1_scale[0] * 1.4426950408889634f;
            float output_scale = bmm2_scale[0];
            const int co_row_base = warp % 4 * 32;
            const int co_thread = co_row_base + lane;
            const int co_head = co_thread & 63;
            const int co_half = co_thread >> 6;
            const int co_lane_addr = co_row_base << 16;
            int co_tile_cursor = 0;
            int co_item = 0;
            unsigned int co_work_stage = 0;
            int co_query = blockIdx.x;
            mbarrier_arrive(o_empty_addr);
            unsigned int _phase_work_full_1 = 0;
            #pragma unroll 1
            for (unsigned int co_work = 0; co_work < 1048576; co_work++) {
                if (co_query < num_query_tokens) {
                    int _max_24 = ((sparse_topk_lens[co_query] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[co_query] + sparse_topk_lens_offset) : (0));
                    int _min_15 = ((_max_24) < (swa_width) ? (_max_24) : (swa_width));
                    int co_main_active = _min_15;
                    int co_extra_active = 0;
                    if (compressed_width > 0) {
                        int _max_25 = ((extra_topk_lens[co_query]) > (0) ? (extra_topk_lens[co_query]) : (0));
                        int _min_16 = ((_max_25) < (compressed_width) ? (_max_25) : (compressed_width));
                        co_extra_active = _min_16;
                    }
                    int co_n_main = (co_main_active + 64 - 1) / 64;
                    int co_n_extra = (co_extra_active + 64 - 1) / 64;
                    int _max_26 = ((co_n_main + co_n_extra) > (1) ? (co_n_main + co_n_extra) : (1));
                    int co_tiles = _max_26;
                    #pragma unroll 1
                    for (int co_tile = 0; co_tile < co_tiles; co_tile++) {
                        int ct = co_tile_cursor + co_tile;
                        int c_stage = ct & 1;
                        int c_phase = ct >> 1 & 1;
                        mbarrier_wait(stats_addr + (c_stage) * 8, c_phase);
                        float acc_scale_1 = smem_scale[c_stage * 128 + co_thread];
                        int _vote_4 = __any_sync(0xFFFFFFFF, acc_scale_1 < 1.0f);
                        int any_rescale = _vote_4;
                        int o_full_phase = ct - 1 & 1;
                        if (co_tile > 0) {
                            mbarrier_wait(o_full_addr, o_full_phase);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (any_rescale != 0) {
                                int o_base = taddr + (unsigned int)co_lane_addr;
                                #pragma unroll
                                for (int c = 0; c < 128; c += 16) {
                                    float _tmem_load_0[16];
                                    tmem_ld_x16(&_tmem_load_0[0], o_base + c);
                                    #if __CUDA_ARCH__ >= 1000
                                    const float2 _scale2_0 = {acc_scale_1, acc_scale_1};
                                    #pragma unroll
                                    for (int _ls = 0; _ls < 8; _ls++)
                                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_ls], _scale2_0);
                                    #else
                                    #pragma unroll
                                    for (int _ls = 0; _ls < 16; _ls++) {
                                        _tmem_load_0[_ls] = _tmem_load_0[_ls] * acc_scale_1;
                                    }
                                    #endif
                                    tmem_st_x16_f32(o_base + c, _tmem_load_0);
                                }
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive(o_empty_addr);
                        }
                        if (co_tile > 0) {
                            mbarrier_wait(o_full_addr + 8, o_full_phase);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (any_rescale != 0) {
                                int o_base_1 = taddr + 128 + (unsigned int)co_lane_addr;
                                #pragma unroll
                                for (int c_1 = 0; c_1 < 128; c_1 += 16) {
                                    float _tmem_load_1[16];
                                    tmem_ld_x16(&_tmem_load_1[0], o_base_1 + c_1);
                                    #if __CUDA_ARCH__ >= 1000
                                    const float2 _scale2_1 = {acc_scale_1, acc_scale_1};
                                    #pragma unroll
                                    for (int _ls = 0; _ls < 8; _ls++)
                                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_1);
                                    #else
                                    #pragma unroll
                                    for (int _ls = 0; _ls < 16; _ls++) {
                                        _tmem_load_1[_ls] = _tmem_load_1[_ls] * acc_scale_1;
                                    }
                                    #endif
                                    tmem_st_x16_f32(o_base_1 + c_1, _tmem_load_1);
                                }
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                            asm volatile("tcgen05.fence::before_thread_sync;");
                        }
                        mbarrier_arrive(o_empty_addr + 8);
                    }
                    int o_last_phase = co_tile_cursor + co_tiles - 1 & 1;
                    mbarrier_wait(o_full_addr, o_last_phase);
                    mbarrier_wait(o_full_addr + 8, o_last_phase);
                    int sum_ready_stage = 0;
                    int sum_ready_phase = co_item & 1;
                    mbarrier_wait(sum_ready_addr + (sum_ready_stage) * 8, sum_ready_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float total_sum = smem_sum[co_thread] + smem_sum[co_thread ^ 64];
                    float final_max = smem_fmax[co_thread];
                    const int ep_lane0 = co_row_base + lane % 16;
                    const int ep_lane1 = ep_lane0 + 16;
                    float ep_sum0 = smem_sum[ep_lane0] + smem_sum[ep_lane0 ^ 64];
                    float ep_sum1 = smem_sum[ep_lane1] + smem_sum[ep_lane1 ^ 64];
                    mbarrier_arrive(sum_empty_addr);
                    if (co_half == 0 && co_head < num_heads) {
                        float _log2_0;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(total_sum));
                        LSE[co_query * num_heads + co_head] = ((total_sum > 0.0f) ? __fmaf_rn(final_max, softmax_scale_log2_1, _log2_0) : -CUDART_INF_F);
                    }
                    float _rcp_0 = approx_rcp(ep_sum0);
                    float ep_scale0 = ((ep_sum0 > 0.0f) ? _rcp_0 : 0.0f) * output_scale;
                    float _rcp_1 = approx_rcp(ep_sum1);
                    float ep_scale1 = ((ep_sum1 > 0.0f) ? _rcp_1 : 0.0f) * output_scale;
                    int ep_shuffle = num_heads > 32;
                    const int ep_src_lane = (lane >> 1) + (lane & 1) * 16;
                    int ep_dst_sub = ((ep_shuffle != 0) ? lane >> 1 : lane % 16);
                    int ep_dst_half = ((ep_shuffle != 0) ? lane & 1 : lane / 16);
                    int ep_head0 = co_row_base + ep_dst_sub & 63;
                    int ep_head1 = ep_head0 + 16;
                    int ep_col = co_half * 128 + ep_dst_half * 16;
                    int ep_row0 = (co_query * num_heads + ep_head0) * 512 + ep_col;
                    int ep_row1 = (co_query * num_heads + ep_head1) * 512 + ep_col;
                    const int ep_group0 = co_row_base & 63;
                    float ep_buf0[16];
                    float ep_buf1[16];
                    float ep_buf2[16];
                    float ep_buf3[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[3])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[5])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[6])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[7])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[9])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[10])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[11])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[13])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[14])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[15]))
                        : "r"(taddr + (unsigned int)co_lane_addr));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[3])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[5])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[6])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[7])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[9])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[10])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[11])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[13])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[14])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[15]))
                        : "r"(taddr + (unsigned int)co_lane_addr + 1048576));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[3])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[5])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[6])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[7])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[9])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[10])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[11])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[13])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[14])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[15]))
                        : "r"(taddr + (unsigned int)co_lane_addr + 32));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[3])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[5])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[6])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[7])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[9])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[10])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[11])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[13])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[14])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[15]))
                        : "r"(taddr + (unsigned int)co_lane_addr + 1048576 + 32));
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_2 = {ep_scale0, ep_scale0};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf0)[_ls], _scale2_2);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        ep_buf0[_ls] = ep_buf0[_ls] * ep_scale0;
                    }
                    #endif
                    unsigned int ep_pack0[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf0[_lp*2 + 0], ep_buf0[_lp*2+1 + 0]));
                        ep_pack0[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (ep_shuffle != 0) {
                        #pragma unroll
                        for (int w = 0; w < 8; w++) {
                            unsigned int _shfl_0 = __shfl_sync(0xFFFFFFFF, ep_pack0[w], ep_src_lane);
                            ep_pack0[w] = _shfl_0;
                        }
                    }
                    if (ep_head0 < num_heads) {
                        {
                            const unsigned* _raw_stv8_3 = reinterpret_cast<const unsigned*>(ep_pack0);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(O + ep_row0)), "r"(_raw_stv8_3[0]), "r"(_raw_stv8_3[1]), "r"(_raw_stv8_3[2]), "r"(_raw_stv8_3[3]), "r"(_raw_stv8_3[4]), "r"(_raw_stv8_3[5]), "r"(_raw_stv8_3[6]), "r"(_raw_stv8_3[7]) : "memory");
                        }
                    }
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[3])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[5])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[6])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[7])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[9])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[10])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[11])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[13])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[14])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[15]))
                        : "r"(taddr + (unsigned int)co_lane_addr + 64));
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_4 = {ep_scale1, ep_scale1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf1)[_ls], _scale2_4);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        ep_buf1[_ls] = ep_buf1[_ls] * ep_scale1;
                    }
                    #endif
                    unsigned int ep_pack1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf1[_lp*2 + 0], ep_buf1[_lp*2+1 + 0]));
                        ep_pack1[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (ep_shuffle != 0) {
                        #pragma unroll
                        for (int w_1 = 0; w_1 < 8; w_1++) {
                            unsigned int _shfl_1 = __shfl_sync(0xFFFFFFFF, ep_pack1[w_1], ep_src_lane);
                            ep_pack1[w_1] = _shfl_1;
                        }
                    }
                    if (ep_head1 < num_heads) {
                        {
                            const unsigned* _raw_stv8_5 = reinterpret_cast<const unsigned*>(ep_pack1);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(O + ep_row1)), "r"(_raw_stv8_5[0]), "r"(_raw_stv8_5[1]), "r"(_raw_stv8_5[2]), "r"(_raw_stv8_5[3]), "r"(_raw_stv8_5[4]), "r"(_raw_stv8_5[5]), "r"(_raw_stv8_5[6]), "r"(_raw_stv8_5[7]) : "memory");
                        }
                    }
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[3])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[5])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[6])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[7])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[9])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[10])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[11])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[13])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[14])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[15]))
                        : "r"(taddr + (unsigned int)co_lane_addr + 1048576 + 64));
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_6 = {ep_scale0, ep_scale0};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf2)[_ls], _scale2_6);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        ep_buf2[_ls] = ep_buf2[_ls] * ep_scale0;
                    }
                    #endif
                    unsigned int ep_pack2[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf2[_lp*2 + 0], ep_buf2[_lp*2+1 + 0]));
                        ep_pack2[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (ep_shuffle != 0) {
                        #pragma unroll
                        for (int w_2 = 0; w_2 < 8; w_2++) {
                            unsigned int _shfl_2 = __shfl_sync(0xFFFFFFFF, ep_pack2[w_2], ep_src_lane);
                            ep_pack2[w_2] = _shfl_2;
                        }
                    }
                    if (ep_head0 < num_heads) {
                        {
                            const unsigned* _raw_stv8_7 = reinterpret_cast<const unsigned*>(ep_pack2);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(O + (ep_row0 + 32))), "r"(_raw_stv8_7[0]), "r"(_raw_stv8_7[1]), "r"(_raw_stv8_7[2]), "r"(_raw_stv8_7[3]), "r"(_raw_stv8_7[4]), "r"(_raw_stv8_7[5]), "r"(_raw_stv8_7[6]), "r"(_raw_stv8_7[7]) : "memory");
                        }
                    }
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[3])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[5])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[6])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[7])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[9])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[10])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[11])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[13])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[14])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[15]))
                        : "r"(taddr + (unsigned int)co_lane_addr + 96));
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_8 = {ep_scale1, ep_scale1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf3)[_ls], _scale2_8);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        ep_buf3[_ls] = ep_buf3[_ls] * ep_scale1;
                    }
                    #endif
                    unsigned int ep_pack3[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf3[_lp*2 + 0], ep_buf3[_lp*2+1 + 0]));
                        ep_pack3[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (ep_shuffle != 0) {
                        #pragma unroll
                        for (int w_3 = 0; w_3 < 8; w_3++) {
                            unsigned int _shfl_3 = __shfl_sync(0xFFFFFFFF, ep_pack3[w_3], ep_src_lane);
                            ep_pack3[w_3] = _shfl_3;
                        }
                    }
                    if (ep_head1 < num_heads) {
                        {
                            const unsigned* _raw_stv8_9 = reinterpret_cast<const unsigned*>(ep_pack3);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(O + (ep_row1 + 32))), "r"(_raw_stv8_9[0]), "r"(_raw_stv8_9[1]), "r"(_raw_stv8_9[2]), "r"(_raw_stv8_9[3]), "r"(_raw_stv8_9[4]), "r"(_raw_stv8_9[5]), "r"(_raw_stv8_9[6]), "r"(_raw_stv8_9[7]) : "memory");
                        }
                    }
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[3])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[5])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[6])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[7])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[9])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[10])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[11])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[13])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[14])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[15]))
                        : "r"(taddr + (unsigned int)co_lane_addr + 1048576 + 96));
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_10 = {ep_scale0, ep_scale0};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf0)[_ls], _scale2_10);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        ep_buf0[_ls] = ep_buf0[_ls] * ep_scale0;
                    }
                    #endif
                    unsigned int ep_pack4[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf0[_lp*2 + 0], ep_buf0[_lp*2+1 + 0]));
                        ep_pack4[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (ep_shuffle != 0) {
                        #pragma unroll
                        for (int w_4 = 0; w_4 < 8; w_4++) {
                            unsigned int _shfl_4 = __shfl_sync(0xFFFFFFFF, ep_pack4[w_4], ep_src_lane);
                            ep_pack4[w_4] = _shfl_4;
                        }
                    }
                    if (ep_head0 < num_heads) {
                        {
                            const unsigned* _raw_stv8_11 = reinterpret_cast<const unsigned*>(ep_pack4);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(O + (ep_row0 + 64))), "r"(_raw_stv8_11[0]), "r"(_raw_stv8_11[1]), "r"(_raw_stv8_11[2]), "r"(_raw_stv8_11[3]), "r"(_raw_stv8_11[4]), "r"(_raw_stv8_11[5]), "r"(_raw_stv8_11[6]), "r"(_raw_stv8_11[7]) : "memory");
                        }
                    }
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[3])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[5])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[6])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[7])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[9])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[10])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[11])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[13])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[14])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[15]))
                        : "r"(taddr + (unsigned int)co_lane_addr + 128));
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_12 = {ep_scale1, ep_scale1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf1)[_ls], _scale2_12);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        ep_buf1[_ls] = ep_buf1[_ls] * ep_scale1;
                    }
                    #endif
                    unsigned int ep_pack5[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf1[_lp*2 + 0], ep_buf1[_lp*2+1 + 0]));
                        ep_pack5[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (ep_shuffle != 0) {
                        #pragma unroll
                        for (int w_5 = 0; w_5 < 8; w_5++) {
                            unsigned int _shfl_5 = __shfl_sync(0xFFFFFFFF, ep_pack5[w_5], ep_src_lane);
                            ep_pack5[w_5] = _shfl_5;
                        }
                    }
                    if (ep_head1 < num_heads) {
                        {
                            const unsigned* _raw_stv8_13 = reinterpret_cast<const unsigned*>(ep_pack5);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(O + (ep_row1 + 64))), "r"(_raw_stv8_13[0]), "r"(_raw_stv8_13[1]), "r"(_raw_stv8_13[2]), "r"(_raw_stv8_13[3]), "r"(_raw_stv8_13[4]), "r"(_raw_stv8_13[5]), "r"(_raw_stv8_13[6]), "r"(_raw_stv8_13[7]) : "memory");
                        }
                    }
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[3])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[5])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[6])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[7])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[9])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[10])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[11])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[13])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[14])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[15]))
                        : "r"(taddr + (unsigned int)co_lane_addr + 128 + 1048576));
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_14 = {ep_scale0, ep_scale0};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf2)[_ls], _scale2_14);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        ep_buf2[_ls] = ep_buf2[_ls] * ep_scale0;
                    }
                    #endif
                    unsigned int ep_pack6[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf2[_lp*2 + 0], ep_buf2[_lp*2+1 + 0]));
                        ep_pack6[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (ep_shuffle != 0) {
                        #pragma unroll
                        for (int w_6 = 0; w_6 < 8; w_6++) {
                            unsigned int _shfl_6 = __shfl_sync(0xFFFFFFFF, ep_pack6[w_6], ep_src_lane);
                            ep_pack6[w_6] = _shfl_6;
                        }
                    }
                    if (ep_head0 < num_heads) {
                        {
                            const unsigned* _raw_stv8_15 = reinterpret_cast<const unsigned*>(ep_pack6);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(O + (ep_row0 + 96))), "r"(_raw_stv8_15[0]), "r"(_raw_stv8_15[1]), "r"(_raw_stv8_15[2]), "r"(_raw_stv8_15[3]), "r"(_raw_stv8_15[4]), "r"(_raw_stv8_15[5]), "r"(_raw_stv8_15[6]), "r"(_raw_stv8_15[7]) : "memory");
                        }
                    }
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[3])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[5])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[6])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[7])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[9])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[10])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[11])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[13])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[14])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[15]))
                        : "r"(taddr + (unsigned int)co_lane_addr + 128 + 32));
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_16 = {ep_scale1, ep_scale1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf3)[_ls], _scale2_16);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        ep_buf3[_ls] = ep_buf3[_ls] * ep_scale1;
                    }
                    #endif
                    unsigned int ep_pack7[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf3[_lp*2 + 0], ep_buf3[_lp*2+1 + 0]));
                        ep_pack7[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (ep_shuffle != 0) {
                        #pragma unroll
                        for (int w_7 = 0; w_7 < 8; w_7++) {
                            unsigned int _shfl_7 = __shfl_sync(0xFFFFFFFF, ep_pack7[w_7], ep_src_lane);
                            ep_pack7[w_7] = _shfl_7;
                        }
                    }
                    if (ep_head1 < num_heads) {
                        {
                            const unsigned* _raw_stv8_17 = reinterpret_cast<const unsigned*>(ep_pack7);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(O + (ep_row1 + 96))), "r"(_raw_stv8_17[0]), "r"(_raw_stv8_17[1]), "r"(_raw_stv8_17[2]), "r"(_raw_stv8_17[3]), "r"(_raw_stv8_17[4]), "r"(_raw_stv8_17[5]), "r"(_raw_stv8_17[6]), "r"(_raw_stv8_17[7]) : "memory");
                        }
                    }
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[3])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[5])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[6])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[7])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[9])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[10])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[11])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[13])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[14])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[15]))
                        : "r"(taddr + (unsigned int)co_lane_addr + 128 + 1048576 + 32));
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(o_empty_addr);
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_18 = {ep_scale0, ep_scale0};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf0)[_ls], _scale2_18);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        ep_buf0[_ls] = ep_buf0[_ls] * ep_scale0;
                    }
                    #endif
                    unsigned int ep_pack8[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf0[_lp*2 + 0], ep_buf0[_lp*2+1 + 0]));
                        ep_pack8[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (ep_shuffle != 0) {
                        #pragma unroll
                        for (int w_8 = 0; w_8 < 8; w_8++) {
                            unsigned int _shfl_8 = __shfl_sync(0xFFFFFFFF, ep_pack8[w_8], ep_src_lane);
                            ep_pack8[w_8] = _shfl_8;
                        }
                    }
                    if (ep_head0 < num_heads) {
                        {
                            const unsigned* _raw_stv8_19 = reinterpret_cast<const unsigned*>(ep_pack8);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(O + (ep_row0 + 256))), "r"(_raw_stv8_19[0]), "r"(_raw_stv8_19[1]), "r"(_raw_stv8_19[2]), "r"(_raw_stv8_19[3]), "r"(_raw_stv8_19[4]), "r"(_raw_stv8_19[5]), "r"(_raw_stv8_19[6]), "r"(_raw_stv8_19[7]) : "memory");
                        }
                    }
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[3])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[5])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[6])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[7])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[9])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[10])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[11])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[13])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[14])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[15]))
                        : "r"(taddr + (unsigned int)co_lane_addr + 128 + 64));
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_20 = {ep_scale1, ep_scale1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf1)[_ls], _scale2_20);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        ep_buf1[_ls] = ep_buf1[_ls] * ep_scale1;
                    }
                    #endif
                    unsigned int ep_pack9[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf1[_lp*2 + 0], ep_buf1[_lp*2+1 + 0]));
                        ep_pack9[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (ep_shuffle != 0) {
                        #pragma unroll
                        for (int w_9 = 0; w_9 < 8; w_9++) {
                            unsigned int _shfl_9 = __shfl_sync(0xFFFFFFFF, ep_pack9[w_9], ep_src_lane);
                            ep_pack9[w_9] = _shfl_9;
                        }
                    }
                    if (ep_head1 < num_heads) {
                        {
                            const unsigned* _raw_stv8_21 = reinterpret_cast<const unsigned*>(ep_pack9);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(O + (ep_row1 + 256))), "r"(_raw_stv8_21[0]), "r"(_raw_stv8_21[1]), "r"(_raw_stv8_21[2]), "r"(_raw_stv8_21[3]), "r"(_raw_stv8_21[4]), "r"(_raw_stv8_21[5]), "r"(_raw_stv8_21[6]), "r"(_raw_stv8_21[7]) : "memory");
                        }
                    }
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[3])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[5])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[6])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[7])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[9])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[10])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[11])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[13])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[14])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[15]))
                        : "r"(taddr + (unsigned int)co_lane_addr + 128 + 1048576 + 64));
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_22 = {ep_scale0, ep_scale0};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf2)[_ls], _scale2_22);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        ep_buf2[_ls] = ep_buf2[_ls] * ep_scale0;
                    }
                    #endif
                    unsigned int ep_pack10[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf2[_lp*2 + 0], ep_buf2[_lp*2+1 + 0]));
                        ep_pack10[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (ep_shuffle != 0) {
                        #pragma unroll
                        for (int w_10 = 0; w_10 < 8; w_10++) {
                            unsigned int _shfl_10 = __shfl_sync(0xFFFFFFFF, ep_pack10[w_10], ep_src_lane);
                            ep_pack10[w_10] = _shfl_10;
                        }
                    }
                    if (ep_head0 < num_heads) {
                        {
                            const unsigned* _raw_stv8_23 = reinterpret_cast<const unsigned*>(ep_pack10);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(O + (ep_row0 + 256 + 32))), "r"(_raw_stv8_23[0]), "r"(_raw_stv8_23[1]), "r"(_raw_stv8_23[2]), "r"(_raw_stv8_23[3]), "r"(_raw_stv8_23[4]), "r"(_raw_stv8_23[5]), "r"(_raw_stv8_23[6]), "r"(_raw_stv8_23[7]) : "memory");
                        }
                    }
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[3])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[5])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[6])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[7])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[9])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[10])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[11])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[13])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[14])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[15]))
                        : "r"(taddr + (unsigned int)co_lane_addr + 128 + 96));
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_24 = {ep_scale1, ep_scale1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf3)[_ls], _scale2_24);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        ep_buf3[_ls] = ep_buf3[_ls] * ep_scale1;
                    }
                    #endif
                    unsigned int ep_pack11[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf3[_lp*2 + 0], ep_buf3[_lp*2+1 + 0]));
                        ep_pack11[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (ep_shuffle != 0) {
                        #pragma unroll
                        for (int w_11 = 0; w_11 < 8; w_11++) {
                            unsigned int _shfl_11 = __shfl_sync(0xFFFFFFFF, ep_pack11[w_11], ep_src_lane);
                            ep_pack11[w_11] = _shfl_11;
                        }
                    }
                    if (ep_head1 < num_heads) {
                        {
                            const unsigned* _raw_stv8_25 = reinterpret_cast<const unsigned*>(ep_pack11);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(O + (ep_row1 + 256 + 32))), "r"(_raw_stv8_25[0]), "r"(_raw_stv8_25[1]), "r"(_raw_stv8_25[2]), "r"(_raw_stv8_25[3]), "r"(_raw_stv8_25[4]), "r"(_raw_stv8_25[5]), "r"(_raw_stv8_25[6]), "r"(_raw_stv8_25[7]) : "memory");
                        }
                    }
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[3])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[5])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[6])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[7])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[9])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[10])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[11])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[13])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[14])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[15]))
                        : "r"(taddr + (unsigned int)co_lane_addr + 128 + 1048576 + 96));
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_26 = {ep_scale0, ep_scale0};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf0)[_ls], _scale2_26);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        ep_buf0[_ls] = ep_buf0[_ls] * ep_scale0;
                    }
                    #endif
                    unsigned int ep_pack12[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf0[_lp*2 + 0], ep_buf0[_lp*2+1 + 0]));
                        ep_pack12[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (ep_shuffle != 0) {
                        #pragma unroll
                        for (int w_12 = 0; w_12 < 8; w_12++) {
                            unsigned int _shfl_12 = __shfl_sync(0xFFFFFFFF, ep_pack12[w_12], ep_src_lane);
                            ep_pack12[w_12] = _shfl_12;
                        }
                    }
                    if (ep_head0 < num_heads) {
                        {
                            const unsigned* _raw_stv8_27 = reinterpret_cast<const unsigned*>(ep_pack12);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(O + (ep_row0 + 256 + 64))), "r"(_raw_stv8_27[0]), "r"(_raw_stv8_27[1]), "r"(_raw_stv8_27[2]), "r"(_raw_stv8_27[3]), "r"(_raw_stv8_27[4]), "r"(_raw_stv8_27[5]), "r"(_raw_stv8_27[6]), "r"(_raw_stv8_27[7]) : "memory");
                        }
                    }
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_28 = {ep_scale1, ep_scale1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf1)[_ls], _scale2_28);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        ep_buf1[_ls] = ep_buf1[_ls] * ep_scale1;
                    }
                    #endif
                    unsigned int ep_pack13[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf1[_lp*2 + 0], ep_buf1[_lp*2+1 + 0]));
                        ep_pack13[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (ep_shuffle != 0) {
                        #pragma unroll
                        for (int w_13 = 0; w_13 < 8; w_13++) {
                            unsigned int _shfl_13 = __shfl_sync(0xFFFFFFFF, ep_pack13[w_13], ep_src_lane);
                            ep_pack13[w_13] = _shfl_13;
                        }
                    }
                    if (ep_head1 < num_heads) {
                        {
                            const unsigned* _raw_stv8_29 = reinterpret_cast<const unsigned*>(ep_pack13);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(O + (ep_row1 + 256 + 64))), "r"(_raw_stv8_29[0]), "r"(_raw_stv8_29[1]), "r"(_raw_stv8_29[2]), "r"(_raw_stv8_29[3]), "r"(_raw_stv8_29[4]), "r"(_raw_stv8_29[5]), "r"(_raw_stv8_29[6]), "r"(_raw_stv8_29[7]) : "memory");
                        }
                    }
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_30 = {ep_scale0, ep_scale0};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf2)[_ls], _scale2_30);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        ep_buf2[_ls] = ep_buf2[_ls] * ep_scale0;
                    }
                    #endif
                    unsigned int ep_pack14[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf2[_lp*2 + 0], ep_buf2[_lp*2+1 + 0]));
                        ep_pack14[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (ep_shuffle != 0) {
                        #pragma unroll
                        for (int w_14 = 0; w_14 < 8; w_14++) {
                            unsigned int _shfl_14 = __shfl_sync(0xFFFFFFFF, ep_pack14[w_14], ep_src_lane);
                            ep_pack14[w_14] = _shfl_14;
                        }
                    }
                    if (ep_head0 < num_heads) {
                        {
                            const unsigned* _raw_stv8_31 = reinterpret_cast<const unsigned*>(ep_pack14);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(O + (ep_row0 + 256 + 96))), "r"(_raw_stv8_31[0]), "r"(_raw_stv8_31[1]), "r"(_raw_stv8_31[2]), "r"(_raw_stv8_31[3]), "r"(_raw_stv8_31[4]), "r"(_raw_stv8_31[5]), "r"(_raw_stv8_31[6]), "r"(_raw_stv8_31[7]) : "memory");
                        }
                    }
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_32 = {ep_scale1, ep_scale1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf3)[_ls], _scale2_32);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        ep_buf3[_ls] = ep_buf3[_ls] * ep_scale1;
                    }
                    #endif
                    unsigned int ep_pack15[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf3[_lp*2 + 0], ep_buf3[_lp*2+1 + 0]));
                        ep_pack15[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (ep_shuffle != 0) {
                        #pragma unroll
                        for (int w_15 = 0; w_15 < 8; w_15++) {
                            unsigned int _shfl_15 = __shfl_sync(0xFFFFFFFF, ep_pack15[w_15], ep_src_lane);
                            ep_pack15[w_15] = _shfl_15;
                        }
                    }
                    if (ep_head1 < num_heads) {
                        {
                            const unsigned* _raw_stv8_33 = reinterpret_cast<const unsigned*>(ep_pack15);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(O + (ep_row1 + 256 + 96))), "r"(_raw_stv8_33[0]), "r"(_raw_stv8_33[1]), "r"(_raw_stv8_33[2]), "r"(_raw_stv8_33[3]), "r"(_raw_stv8_33[4]), "r"(_raw_stv8_33[5]), "r"(_raw_stv8_33[6]), "r"(_raw_stv8_33[7]) : "memory");
                        }
                    }
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    co_tile_cursor = co_tile_cursor + co_tiles;
                    co_item = co_item + 1;
                }
                mbarrier_wait(work_full_addr + (co_work_stage) * 8, _phase_work_full_1);
                uint32_t _clc_valid_7 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_7)
                    : "r"(work_response_addr + co_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_7 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_7)
                    : "r"(work_response_addr + co_work_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (co_work_stage) * 8);
                co_work_stage += 1;
                if (co_work_stage == 2) { co_work_stage = 0; _phase_work_full_1 ^= 1; }
                if (_clc_valid_7 == 0) {
                    break;
                }
                co_query = _clc_ctaid_7;
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 8) {
        { // mma_warp_main
            int mma_tile_cursor = 0;
            int mma_item = 0;
            unsigned int mma_work_stage = 0;
            int mma_query = blockIdx.x;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int mma_work = 0; mma_work < 1048576; mma_work++) {
                if (mma_query < num_query_tokens) {
                    int _max_12 = ((sparse_topk_lens[mma_query] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[mma_query] + sparse_topk_lens_offset) : (0));
                    int _min_8 = ((_max_12) < (swa_width) ? (_max_12) : (swa_width));
                    int mma_main_active = _min_8;
                    int mma_extra_active = 0;
                    if (compressed_width > 0) {
                        int _max_13 = ((extra_topk_lens[mma_query]) > (0) ? (extra_topk_lens[mma_query]) : (0));
                        int _min_9 = ((_max_13) < (compressed_width) ? (_max_13) : (compressed_width));
                        mma_extra_active = _min_9;
                    }
                    int mma_n_main = (mma_main_active + 64 - 1) / 64;
                    int mma_n_extra = (mma_extra_active + 64 - 1) / 64;
                    int _max_14 = ((mma_n_main + mma_n_extra) > (1) ? (mma_n_main + mma_n_extra) : (1));
                    int mma_tiles = _max_14;
                    int q_empty_stage = 0;
                    int q_empty_phase = mma_item & 1 ^ 1;
                    mbarrier_wait(q_empty_addr + (q_empty_stage) * 8, q_empty_phase);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(q_load_full_addr, 51200);
                        #pragma unroll
                        for (int q_copy = 0; q_copy < 2; q_copy++) {
                            tma_3d_gmem2smem(smem_q_a_addr + (unsigned int)(q_copy * 8192), (&tmap_q), 0, 0, mma_query, q_load_full_addr);
                            tma_3d_gmem2smem(smem_q_b_addr + (unsigned int)(q_copy * 8192), (&tmap_q), 128, 0, mma_query, q_load_full_addr);
                            tma_3d_gmem2smem(smem_q_rope_addr + (unsigned int)(q_copy * 8192), (&tmap_q), 224, 0, mma_query, q_load_full_addr);
                        }
                        tma_3d_gmem2smem(smem_q_sf_addr, (&tmap_q_sf), 352, 0, mma_query, q_load_full_addr);
                    }
                    int q_full_stage = 0;
                    int q_full_phase = mma_item & 1;
                    mbarrier_wait(q_full_addr + (q_full_stage) * 8, q_full_phase);
                    #pragma unroll 1
                    for (int mma_tile = 0; mma_tile < mma_tiles; mma_tile++) {
                        int mt = mma_tile_cursor + mma_tile;
                        int m_stage = mt & 1;
                        int mk_stage = mt % 3;
                        int mk_phase = mt / 3 & 1;
                        mbarrier_wait(sf_full_addr + (mk_stage) * 8, mk_phase);
                        int s_free_stage = mt & 1;
                        int s_free_phase = mt >> 1 & 1 ^ 1;
                        mbarrier_wait(s_empty_addr + (s_free_stage) * 8, s_free_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_0 = make_warp_uniform((((smem_q_a_addr) >> 4) & 0x3FFF) + (0) * 1024);
                        int _mma_b_lo_0 = make_warp_uniform((((smem_k_a_addr) >> 4) & 0x3FFF) + (mk_stage) * 1664);
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4nvf4_bs((tmem_tmem_scratch + (256 + m_stage * 64)), a_desc + 0, b_desc + 0,
                                    0x8100480U, tmem_tmem_sfa + 0, tmem_tmem_sfb + mk_stage * 32 + 0, 0);
                                tcgen05_mma_mxf4nvf4_bs((tmem_tmem_scratch + (256 + m_stage * 64)), a_desc + 2, b_desc + 2,
                                    0x8100480U, tmem_tmem_sfa + 4, tmem_tmem_sfb + mk_stage * 32 + 4, 1);
                                tcgen05_mma_mxf4nvf4_bs((tmem_tmem_scratch + (256 + m_stage * 64)), a_desc + 4, b_desc + 4,
                                    0x8100480U, tmem_tmem_sfa + 8, tmem_tmem_sfb + mk_stage * 32 + 8, 1);
                                tcgen05_mma_mxf4nvf4_bs((tmem_tmem_scratch + (256 + m_stage * 64)), a_desc + 6, b_desc + 6,
                                    0x8100480U, tmem_tmem_sfa + 12, tmem_tmem_sfb + mk_stage * 32 + 12, 1);
                            }
                        }
                        int _mma_a_lo_1 = make_warp_uniform((((smem_q_b_addr) >> 4) & 0x3FFF) + (0) * 1024);
                        int _mma_b_lo_1 = make_warp_uniform((((smem_k_b_addr) >> 4) & 0x3FFF) + (mk_stage) * 1664);
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4nvf4_bs((tmem_tmem_scratch + (256 + m_stage * 64)), a_desc + 0, b_desc + 0,
                                    0x8100480U, tmem_tmem_sfa + 16 + 0, tmem_tmem_sfb + (mk_stage * 32 + 16) + 0, 1);
                                tcgen05_mma_mxf4nvf4_bs((tmem_tmem_scratch + (256 + m_stage * 64)), a_desc + 2, b_desc + 2,
                                    0x8100480U, tmem_tmem_sfa + 16 + 4, tmem_tmem_sfb + (mk_stage * 32 + 16) + 4, 1);
                                tcgen05_mma_mxf4nvf4_bs((tmem_tmem_scratch + (256 + m_stage * 64)), a_desc + 4, b_desc + 4,
                                    0x8100480U, tmem_tmem_sfa + 16 + 8, tmem_tmem_sfb + (mk_stage * 32 + 16) + 8, 1);
                            }
                        }
                        int _mma_a_lo_2 = make_warp_uniform((((smem_q_rope_addr) >> 4) & 0x3FFF) + (0) * 1024);
                        int _mma_b_lo_2 = make_warp_uniform((((smem_k_c_addr) >> 4) & 0x3FFF) + (mk_stage) * 1664);
                        {
                            uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                            uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_scratch + (256 + m_stage * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135267472, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_scratch + (256 + m_stage * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135267472, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_scratch + (256 + m_stage * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135267472, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_scratch + (256 + m_stage * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135267472, 1);
                            }
                        }
                        elect_commit(k_empty_addr + (mk_stage) * 8);
                        elect_commit(s_full_addr + (m_stage) * 8);
                    }
                    elect_commit(q_empty_addr);
                    mma_tile_cursor = mma_tile_cursor + mma_tiles;
                    mma_item = mma_item + 1;
                }
                mbarrier_wait(work_full_addr + (mma_work_stage) * 8, _phase_work_full_2);
                uint32_t _clc_valid_4 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_4)
                    : "r"(work_response_addr + mma_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_4 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_4)
                    : "r"(work_response_addr + mma_work_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(work_empty_addr + (mma_work_stage) * 8);
                mma_work_stage += 1;
                if (mma_work_stage == 2) { mma_work_stage = 0; _phase_work_full_2 ^= 1; }
                if (_clc_valid_4 == 0) {
                    break;
                }
                mma_query = _clc_ctaid_4;
            }
            mbarrier_arrive(tmem_dealloc_addr);
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait(tmem_dealloc_addr, _phase_tmem_dealloc_0);
            _phase_tmem_dealloc_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
        }
    }
    // ---- Role: scheduler ----
    if (warp == 9) {
        { // scheduler_main
            unsigned int sc_work_stage = 0;
            unsigned int sc_throttle_stage = 0;
            unsigned int _phase_throttle_full = 0;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_work_full_3 = 0;
            #pragma unroll 1
            for (unsigned int sc_work = 0; sc_work < 1048576; sc_work++) {
                mbarrier_wait(throttle_full_addr + (sc_throttle_stage) * 8, _phase_throttle_full);
                mbarrier_arrive(throttle_empty_addr + (sc_throttle_stage) * 8);
                sc_throttle_stage += 1;
                if (sc_throttle_stage == 2) { sc_throttle_stage = 0; _phase_throttle_full ^= 1; }
                mbarrier_wait(work_empty_addr + (sc_work_stage) * 8, _phase_work_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(work_full_addr + (sc_work_stage) * 8, 16);
                    asm volatile(
                        "fence.proxy.async.shared::cta;\n\t"
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.b128"
                            " [%0], [%1];"
                        :: "r"(work_response_addr + sc_work_stage * 16 + 0 * 16), "r"(work_full_addr + sc_work_stage * 8)
                        : "memory");
                }
                mbarrier_wait(work_full_addr + (sc_work_stage) * 8, _phase_work_full_3);
                uint32_t _clc_valid_8 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_8)
                    : "r"(work_response_addr + sc_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_8 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_8)
                    : "r"(work_response_addr + sc_work_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                sc_work_stage += 1;
                if (sc_work_stage == 2) { sc_work_stage = 0; _phase_work_empty ^= 1; _phase_work_full_3 ^= 1; }
                if (_clc_valid_8 == 0) {
                    break;
                }
            }
            #pragma unroll
            for (unsigned int sc_tail = 0; sc_tail < 2; sc_tail++) {
                mbarrier_wait(work_empty_addr + (sc_work_stage) * 8, _phase_work_empty);
                sc_work_stage += 1;
                if (sc_work_stage == 2) { sc_work_stage = 0; _phase_work_empty ^= 1; _phase_work_full_3 ^= 1; }
            }
        }
    }
    // ---- Role: index_warp ----
    if (warp == 10) {
        { // index_warp_main
            unsigned int ix_stage = 0;
            unsigned int ix_work_stage = 0;
            int ix_query = blockIdx.x;
            unsigned int _phase_index_empty = 1;
            unsigned int _phase_work_full_4 = 0;
            #pragma unroll 1
            for (unsigned int ix_work = 0; ix_work < 1048576; ix_work++) {
                if (ix_query < num_query_tokens) {
                    int _max_0 = ((sparse_topk_lens[ix_query] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[ix_query] + sparse_topk_lens_offset) : (0));
                    int _min_0 = ((_max_0) < (swa_width) ? (_max_0) : (swa_width));
                    int ix_main_active = _min_0;
                    int ix_extra_active = 0;
                    if (compressed_width > 0) {
                        int _max_1 = ((extra_topk_lens[ix_query]) > (0) ? (extra_topk_lens[ix_query]) : (0));
                        int _min_1 = ((_max_1) < (compressed_width) ? (_max_1) : (compressed_width));
                        ix_extra_active = _min_1;
                    }
                    int ix_n_main = (ix_main_active + 64 - 1) / 64;
                    int ix_n_extra = (ix_extra_active + 64 - 1) / 64;
                    int _max_2 = ((ix_n_main + ix_n_extra) > (1) ? (ix_n_main + ix_n_extra) : (1));
                    int ix_tiles = _max_2;
                    int ix_passes = (ix_tiles + 4 - 1) / 4;
                    #pragma unroll 1
                    for (int ix_pass = 0; ix_pass < ix_passes; ix_pass++) {
                        mbarrier_wait(index_empty_addr + (ix_stage) * 8, _phase_index_empty);
                        int ix_base = smem_indices_addr + ix_stage * 1024;
                        #pragma unroll
                        for (int ix_half = 0; ix_half < 2; ix_half++) {
                            int ix_slot = ix_half * 128 + lane * 4;
                            int ix_tile = ix_pass * 4 + (ix_slot >> 6);
                            int ix_in_main = ix_tile < ix_n_main || ix_n_extra == 0;
                            int ix_seg_tile = ((ix_in_main != 0) ? ix_tile : ix_tile - ix_n_main);
                            int ix_width = ((ix_in_main != 0) ? swa_width : compressed_width);
                            int ix_col = ix_seg_tile * 64 + (ix_slot & 63);
                            int* ix_row_ptr = ((ix_in_main != 0) ? (swa_indices + (ix_query * swa_index_stride)) : (compressed_indices + (ix_query * compressed_index_stride)));
                            int ix_vals[4];
                            #pragma unroll
                            for (int ix_j = 0; ix_j < 4; ix_j++) {
                                ix_vals[ix_j] = -1;
                            }
                            #pragma unroll
                            for (int ix_j_1 = 0; ix_j_1 < 4; ix_j_1++) {
                                if (ix_width > ix_col + ix_j_1) {
                                    ix_vals[ix_j_1] = ix_row_ptr[ix_col + ix_j_1];
                                }
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_indices_addr + ((unsigned int)ix_base - smem_indices_addr + (unsigned int)(ix_slot * 4))), "r"(ix_vals[0]), "r"(ix_vals[1]), "r"(ix_vals[2]), "r"(ix_vals[3]) : "memory");
                        }
                        mbarrier_arrive(index_full_addr + (ix_stage) * 8);
                        ix_stage += 1;
                        if (ix_stage == 6) { ix_stage = 0; _phase_index_empty ^= 1; }
                    }
                }
                mbarrier_wait(work_full_addr + (ix_work_stage) * 8, _phase_work_full_4);
                uint32_t _clc_valid_0 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_0)
                    : "r"(work_response_addr + ix_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_0 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_0)
                    : "r"(work_response_addr + ix_work_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (ix_work_stage) * 8);
                ix_work_stage += 1;
                if (ix_work_stage == 2) { ix_work_stage = 0; _phase_work_full_4 ^= 1; }
                if (_clc_valid_0 == 0) {
                    break;
                }
                ix_query = _clc_ctaid_0;
            }
        }
    }
    // ---- Role: pv_issuer ----
    if (warp == 11) {
        { // pv_issuer_main
            int pv_tile_cursor = 0;
            unsigned int pv_work_stage = 0;
            int pv_query = blockIdx.x;
            unsigned int _phase_work_full_5 = 0;
            #pragma unroll 1
            for (unsigned int pv_work = 0; pv_work < 1048576; pv_work++) {
                if (pv_query < num_query_tokens) {
                    int _max_15 = ((sparse_topk_lens[pv_query] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[pv_query] + sparse_topk_lens_offset) : (0));
                    int _min_10 = ((_max_15) < (swa_width) ? (_max_15) : (swa_width));
                    int pv_main_active = _min_10;
                    int pv_extra_active = 0;
                    if (compressed_width > 0) {
                        int _max_16 = ((extra_topk_lens[pv_query]) > (0) ? (extra_topk_lens[pv_query]) : (0));
                        int _min_11 = ((_max_16) < (compressed_width) ? (_max_16) : (compressed_width));
                        pv_extra_active = _min_11;
                    }
                    int pv_n_main = (pv_main_active + 64 - 1) / 64;
                    int pv_n_extra = (pv_extra_active + 64 - 1) / 64;
                    int _max_17 = ((pv_n_main + pv_n_extra) > (1) ? (pv_n_main + pv_n_extra) : (1));
                    int pv_tiles = _max_17;
                    int pv_first = 1;
                    #pragma unroll 1
                    for (int pv_tile = 0; pv_tile < pv_tiles; pv_tile++) {
                        int pvt = pv_tile_cursor + pv_tile;
                        int pv_stage = pvt & 1;
                        int pv_phase = pvt >> 1 & 1;
                        mbarrier_wait(p_full_addr + (pv_stage) * 8, pv_phase);
                        mbarrier_wait(v_full_addr + (pv_stage) * 8, pv_phase);
                        int o_empty_phase = pvt & 1;
                        mbarrier_wait(o_empty_addr, o_empty_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_3 = make_warp_uniform((((smem_p_addr) >> 4) & 0x3FFF) + (pv_stage) * 256);
                        int _mma_b_lo_3 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (pv_stage) * 2048);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 71368720;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"(tmem_tmem_scratch), "r"(((pv_first) ? 0 : 1)));
                        elect_commit(o_full_addr);
                        mbarrier_wait(o_empty_addr + 8, o_empty_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_4 = make_warp_uniform((((smem_p_addr) >> 4) & 0x3FFF) + (pv_stage) * 256);
                        int _mma_b_lo_4 = make_warp_uniform(((((smem_v_addr + 16384) >> 4) & 0x3FFF) | 0x2000000) + (pv_stage) * 2048);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 71368720;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"((tmem_tmem_scratch + (128))), "r"(((pv_first) ? 0 : 1)));
                        elect_commit(o_full_addr + 8);
                        elect_commit(v_empty_addr + (pv_stage) * 8);
                        elect_commit(p_recycle_addr + (pv_stage) * 8);
                        pv_first = 0;
                    }
                    pv_tile_cursor = pv_tile_cursor + pv_tiles;
                }
                mbarrier_wait(work_full_addr + (pv_work_stage) * 8, _phase_work_full_5);
                uint32_t _clc_valid_5 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_5)
                    : "r"(work_response_addr + pv_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_5 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_5)
                    : "r"(work_response_addr + pv_work_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(work_empty_addr + (pv_work_stage) * 8);
                pv_work_stage += 1;
                if (pv_work_stage == 2) { pv_work_stage = 0; _phase_work_full_5 ^= 1; }
                if (_clc_valid_5 == 0) {
                    break;
                }
                pv_query = _clc_ctaid_5;
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: load_warp ----
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // load_warp_main
            const int ld_warp = warp - 12;
            const int ld_thread = ld_warp * 32 + lane;
            const int ld_row_base = ld_warp * 32;
            const int v_key = ld_thread & 63;
            const int v_half = ld_thread >> 6;
            int ld_tile_cursor = 0;
            unsigned int ld_work_stage = 0;
            unsigned int ld_throttle_stage = 0;
            int ld_query = blockIdx.x;
            unsigned int _phase_throttle_empty = 1;
            unsigned int _phase_work_full_6 = 0;
            #pragma unroll 1
            for (unsigned int ld_work = 0; ld_work < 1048576; ld_work++) {
                mbarrier_wait(throttle_empty_addr + (ld_throttle_stage) * 8, _phase_throttle_empty);
                mbarrier_arrive(throttle_full_addr + (ld_throttle_stage) * 8);
                ld_throttle_stage += 1;
                if (ld_throttle_stage == 2) { ld_throttle_stage = 0; _phase_throttle_empty ^= 1; }
                if (ld_query < num_query_tokens) {
                    int _max_6 = ((sparse_topk_lens[ld_query] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[ld_query] + sparse_topk_lens_offset) : (0));
                    int _min_4 = ((_max_6) < (swa_width) ? (_max_6) : (swa_width));
                    int ld_main_active = _min_4;
                    int ld_extra_active = 0;
                    if (compressed_width > 0) {
                        int _max_7 = ((extra_topk_lens[ld_query]) > (0) ? (extra_topk_lens[ld_query]) : (0));
                        int _min_5 = ((_max_7) < (compressed_width) ? (_max_7) : (compressed_width));
                        ld_extra_active = _min_5;
                    }
                    int ld_n_main = (ld_main_active + 64 - 1) / 64;
                    int ld_n_extra = (ld_extra_active + 64 - 1) / 64;
                    int _max_8 = ((ld_n_main + ld_n_extra) > (1) ? (ld_n_main + ld_n_extra) : (1));
                    int ld_tiles = _max_8;
                    int ld_item_on = ld_work < 48;
                    #pragma unroll 1
                    for (int ld_tile = 0; ld_tile < ld_tiles; ld_tile++) {
                        int pt = ld_tile_cursor + ld_tile;
                        int p_stage = pt & 1;
                        int k_stage = pt % 3;
                        int k_phase = pt / 3 & 1;
                        int lt_on = pt < 320;
                        mbarrier_wait(k_full_addr + (k_stage) * 8, k_phase);
                        int sf_base = smem_k_sf_addr + (unsigned int)(k_stage * 26624);
                        unsigned int sf_a[8];
                        unsigned int sf_b[8];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&sf_a[0])), "=r"(*reinterpret_cast<uint32_t*>(&sf_a[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sf_a[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sf_a[(0) + 3]))
                            : "r"(sf_base + lane * 32));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&sf_a[4])), "=r"(*reinterpret_cast<uint32_t*>(&sf_a[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sf_a[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sf_a[(4) + 3]))
                            : "r"(sf_base + lane * 32 + 16));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&sf_b[0])), "=r"(*reinterpret_cast<uint32_t*>(&sf_b[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sf_b[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sf_b[(0) + 3]))
                            : "r"(sf_base + (lane + 32) * 32));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&sf_b[4])), "=r"(*reinterpret_cast<uint32_t*>(&sf_b[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sf_b[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sf_b[(4) + 3]))
                            : "r"(sf_base + (lane + 32) * 32 + 16));
                        int k_empty_stage = pt % 3;
                        int k_empty_phase = pt / 3 & 1 ^ 1;
                        mbarrier_wait(k_empty_addr + (k_empty_stage) * 8, k_empty_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        unsigned int sfb_regs[32];
                        #pragma unroll
                        for (int sf_s = 0; sf_s < 7; sf_s++) {
                            sfb_regs[4 * sf_s] = sf_a[sf_s];
                            sfb_regs[4 * sf_s + 1] = sf_b[sf_s];
                            sfb_regs[4 * sf_s + 2] = 0;
                            sfb_regs[4 * sf_s + 3] = 0;
                        }
                        #pragma unroll
                        for (int sf_pad = 28; sf_pad < 32; sf_pad++) {
                            sfb_regs[sf_pad] = 0;
                        }
                        int sfb_col = 416 + k_stage * 32;
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x16.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(taddr + (unsigned int)sfb_col + (unsigned int)(ld_row_base << 16)), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[0])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[1])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[2])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[3])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[4])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[5])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[6])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[7])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[8])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[9])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[10])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[11])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[12])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[13])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[14])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[15])));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x16.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(taddr + (unsigned int)sfb_col + 16 + (unsigned int)(ld_row_base << 16)), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs + 16)[0])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs + 16)[1])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs + 16)[2])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs + 16)[3])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs + 16)[4])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs + 16)[5])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs + 16)[6])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs + 16)[7])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs + 16)[8])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs + 16)[9])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs + 16)[10])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs + 16)[11])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs + 16)[12])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs + 16)[13])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs + 16)[14])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs + 16)[15])));
                        unsigned int k68 = 0;
                        unsigned int k70 = 0;
                        unsigned int k78 = 0;
                        #pragma unroll
                        for (int sf_s_1 = 0; sf_s_1 < 7; sf_s_1++) {
                            k68 = k68 | sf_a[sf_s_1] + 404232216 | sf_b[sf_s_1] + 404232216;
                            k70 = k70 | sf_a[sf_s_1] + 269488144 | sf_b[sf_s_1] + 269488144;
                            k78 = k78 | sf_a[sf_s_1] + 134744072 | sf_b[sf_s_1] + 134744072;
                        }
                        int _vote_0 = __any_sync(0xFFFFFFFF, (k68 & 2155905152u) != 0);
                        int _vote_1 = __any_sync(0xFFFFFFFF, (k70 & 2155905152u) != 0);
                        int _vote_2 = __any_sync(0xFFFFFFFF, (k78 & 2155905152u) != 0);
                        unsigned int kexp_u = (unsigned int)(_vote_0 != 0) + (unsigned int)(_vote_1 != 0) + (unsigned int)(_vote_2 != 0);
                        unsigned int kexp_f16x2 = (15 - kexp_u << 10) * 65537;
                        float kexp_scale = __uint_as_float(127 - kexp_u << 23);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        if (ld_warp == 0) {
                            if (elect_sync()) {
                                mbarrier_arrive(sf_full_addr + (k_stage) * 8);
                            }
                        }
                        int v_free_stage = pt & 1;
                        int v_free_phase = pt >> 1 & 1 ^ 1;
                        mbarrier_wait(v_empty_addr + (v_free_stage) * 8, v_free_phase);
                        if (ld_warp == 0) {
                            if (elect_sync()) {
                                smem_kexp[p_stage * 4] = kexp_u;
                                mbarrier_arrive(kexp_full_addr + (p_stage) * 8);
                            }
                        }
                        int swz16 = (v_key & 7) << 4;
                        int v_row_off = v_key * 128;
                        int v_sf_addr = sf_base + v_key * 32 + v_half * 16;
                        int v_codes_src = smem_k_a_addr + (unsigned int)(k_stage * 26624) + (unsigned int)(v_half * 8192) + (unsigned int)v_row_off;
                        int v_dst_base0 = p_stage * 32768 + v_key * 128 + 2 * v_half * 8192;
                        int v_dst_base1 = v_dst_base0 + 8192;
                        unsigned int v_sc[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&v_sc[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_sc[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v_sc[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v_sc[(0) + 3]))
                            : "r"(v_sf_addr));
                        if ((int)(kexp_u == 0) != 0) {
                            if (v_half == 0) {
                                unsigned int codes0[4];
                                unsigned int codes1[4];
                                unsigned int codes2[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&codes0[0])), "=r"(*reinterpret_cast<uint32_t*>(&codes0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&codes0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&codes0[(0) + 3]))
                                    : "r"(v_codes_src + (0 ^ swz16)));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&codes1[0])), "=r"(*reinterpret_cast<uint32_t*>(&codes1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&codes1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&codes1[(0) + 3]))
                                    : "r"(v_codes_src + (16 ^ swz16)));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&codes2[0])), "=r"(*reinterpret_cast<uint32_t*>(&codes2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&codes2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&codes2[(0) + 3]))
                                    : "r"(v_codes_src + (32 ^ swz16)));
                                unsigned int v_sc_lo = v_sc[0];
                                unsigned int v_sc_hi = v_sc[0] >> 8;
                                unsigned int v_out[8];
                                #pragma unroll
                                for (int v_w = 0; v_w < 4; v_w++) {
                                    unsigned int v_sc_w = ((v_w < 2) ? v_sc_lo : v_sc_hi);
                                    v_out[2 * v_w] = qmul4_lower4(codes0[v_w], v_sc_w);
                                    v_out[2 * v_w + 1] = qmul4_higher4(codes0[v_w], v_sc_w);
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base0 + (0 ^ swz16))), "r"(v_out[0]), "r"(v_out[1]), "r"(v_out[2]), "r"(v_out[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base0 + (16 ^ swz16))), "r"(v_out[4]), "r"(v_out[5]), "r"(v_out[6]), "r"(v_out[7]) : "memory");
                                unsigned int v_sc_lo_0 = v_sc[0] >> 16;
                                unsigned int v_sc_hi_1 = v_sc[0] >> 24;
                                unsigned int v_out_2[8];
                                #pragma unroll
                                for (int v_w_1 = 0; v_w_1 < 4; v_w_1++) {
                                    unsigned int v_sc_w_1 = ((v_w_1 < 2) ? v_sc_lo_0 : v_sc_hi_1);
                                    v_out_2[2 * v_w_1] = qmul4_lower4(codes1[v_w_1], v_sc_w_1);
                                    v_out_2[2 * v_w_1 + 1] = qmul4_higher4(codes1[v_w_1], v_sc_w_1);
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base0 + (32 ^ swz16))), "r"(v_out_2[0]), "r"(v_out_2[1]), "r"(v_out_2[2]), "r"(v_out_2[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base0 + (48 ^ swz16))), "r"(v_out_2[4]), "r"(v_out_2[5]), "r"(v_out_2[6]), "r"(v_out_2[7]) : "memory");
                                unsigned int v_sc_lo_3 = v_sc[1];
                                unsigned int v_sc_hi_4 = v_sc[1] >> 8;
                                unsigned int v_out_5[8];
                                #pragma unroll
                                for (int v_w_2 = 0; v_w_2 < 4; v_w_2++) {
                                    unsigned int v_sc_w_2 = ((v_w_2 < 2) ? v_sc_lo_3 : v_sc_hi_4);
                                    v_out_5[2 * v_w_2] = qmul4_lower4(codes2[v_w_2], v_sc_w_2);
                                    v_out_5[2 * v_w_2 + 1] = qmul4_higher4(codes2[v_w_2], v_sc_w_2);
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base0 + (64 ^ swz16))), "r"(v_out_5[0]), "r"(v_out_5[1]), "r"(v_out_5[2]), "r"(v_out_5[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base0 + (80 ^ swz16))), "r"(v_out_5[4]), "r"(v_out_5[5]), "r"(v_out_5[6]), "r"(v_out_5[7]) : "memory");
                            }
                            if (v_half == 1) {
                                unsigned int codes0_1[4];
                                unsigned int codes1_1[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&codes0_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&codes0_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&codes0_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&codes0_1[(0) + 3]))
                                    : "r"(v_codes_src + (0 ^ swz16)));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&codes1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&codes1_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&codes1_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&codes1_1[(0) + 3]))
                                    : "r"(v_codes_src + (16 ^ swz16)));
                                unsigned int v_sc_lo_1 = v_sc[0];
                                unsigned int v_sc_hi_2 = v_sc[0] >> 8;
                                unsigned int v_out_1[8];
                                #pragma unroll
                                for (int v_w_3 = 0; v_w_3 < 4; v_w_3++) {
                                    unsigned int v_sc_w_3 = ((v_w_3 < 2) ? v_sc_lo_1 : v_sc_hi_2);
                                    v_out_1[2 * v_w_3] = qmul4_lower4(codes0_1[v_w_3], v_sc_w_3);
                                    v_out_1[2 * v_w_3 + 1] = qmul4_higher4(codes0_1[v_w_3], v_sc_w_3);
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base0 + (0 ^ swz16))), "r"(v_out_1[0]), "r"(v_out_1[1]), "r"(v_out_1[2]), "r"(v_out_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base0 + (16 ^ swz16))), "r"(v_out_1[4]), "r"(v_out_1[5]), "r"(v_out_1[6]), "r"(v_out_1[7]) : "memory");
                                unsigned int v_sc_lo_0_1 = v_sc[0] >> 16;
                                unsigned int v_sc_hi_1_1 = v_sc[0] >> 24;
                                unsigned int v_out_2_1[8];
                                #pragma unroll
                                for (int v_w_4 = 0; v_w_4 < 4; v_w_4++) {
                                    unsigned int v_sc_w_4 = ((v_w_4 < 2) ? v_sc_lo_0_1 : v_sc_hi_1_1);
                                    v_out_2_1[2 * v_w_4] = qmul4_lower4(codes1_1[v_w_4], v_sc_w_4);
                                    v_out_2_1[2 * v_w_4 + 1] = qmul4_higher4(codes1_1[v_w_4], v_sc_w_4);
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base0 + (32 ^ swz16))), "r"(v_out_2_1[0]), "r"(v_out_2_1[1]), "r"(v_out_2_1[2]), "r"(v_out_2_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base0 + (48 ^ swz16))), "r"(v_out_2_1[4]), "r"(v_out_2_1[5]), "r"(v_out_2_1[6]), "r"(v_out_2_1[7]) : "memory");
                            }
                        } else {
                            #pragma unroll 1
                            for (int u = 0; u < 3 - v_half; u++) {
                                unsigned int v_in[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&v_in[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_in[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v_in[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v_in[(0) + 3]))
                                    : "r"(v_codes_src + (u << 4 ^ swz16)));
                                int v_w_idx = u >> 1;
                                unsigned int v_sc_lo_2 = ((v_w_idx == 0) ? v_sc[0] : v_sc[1]);
                                unsigned int v_sc_hi_3 = ((v_w_idx == 2) ? v_sc[2] : v_sc[3]);
                                unsigned int v_sc_word = ((v_w_idx < 2) ? v_sc_lo_2 : v_sc_hi_3);
                                uint32_t _e4m3x2_to_f16x2_0;
                                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_0) : "h"((uint16_t)(v_sc_word >> (unsigned int)((u & 1) * 16))));
                                unsigned int v_sc_pair = _e4m3x2_to_f16x2_0;
                                uint32_t _f16x2_mul_0;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_0) : "r"(v_sc_pair), "r"(kexp_f16x2));
                                v_sc_pair = _f16x2_mul_0;
                                uint32_t _prmt_b32_0;
                                asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_b32_0) : "r"(v_sc_pair), "r"(v_sc_pair));
                                unsigned int v_sc0 = _prmt_b32_0;
                                uint32_t _prmt_b32_1;
                                asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_b32_1) : "r"(v_sc_pair), "r"(v_sc_pair));
                                unsigned int v_sc1 = _prmt_b32_1;
                                unsigned int v_out_3[8];
                                #pragma unroll
                                for (int v_w_5 = 0; v_w_5 < 4; v_w_5++) {
                                    unsigned int v_sc_w_5 = ((v_w_5 < 2) ? v_sc0 : v_sc1);
                                    unsigned int v_word = v_in[v_w_5];
                                    uint32_t _e2m1x2_to_f16x2_0;
                                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_0) : "r"((uint32_t)(v_word)));
                                    uint32_t _f16x2_mul_1;
                                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_1) : "r"(_e2m1x2_to_f16x2_0), "r"(v_sc_w_5));
                                    uint16_t _e4m3x2_0;
                                    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_e4m3x2_0) : "r"(_f16x2_mul_1));
                                    uint32_t _e2m1x2_to_f16x2_1;
                                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_1) : "r"((uint32_t)(v_word >> 8)));
                                    uint32_t _f16x2_mul_2;
                                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_2) : "r"(_e2m1x2_to_f16x2_1), "r"(v_sc_w_5));
                                    uint16_t _e4m3x2_1;
                                    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_e4m3x2_1) : "r"(_f16x2_mul_2));
                                    uint32_t _e2m1x2_to_f16x2_2;
                                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_2) : "r"((uint32_t)(v_word >> 16)));
                                    uint32_t _f16x2_mul_3;
                                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_3) : "r"(_e2m1x2_to_f16x2_2), "r"(v_sc_w_5));
                                    uint16_t _e4m3x2_2;
                                    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_e4m3x2_2) : "r"(_f16x2_mul_3));
                                    uint32_t _e2m1x2_to_f16x2_3;
                                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_3) : "r"((uint32_t)(v_word >> 24)));
                                    uint32_t _f16x2_mul_4;
                                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_4) : "r"(_e2m1x2_to_f16x2_3), "r"(v_sc_w_5));
                                    uint16_t _e4m3x2_3;
                                    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_e4m3x2_3) : "r"(_f16x2_mul_4));
                                    uint32_t _pack_u16x2_0;
                                    asm("mov.b32 %0, {%1, %2};" : "=r"(_pack_u16x2_0) : "h"(_e4m3x2_0), "h"(_e4m3x2_1));
                                    v_out_3[2 * v_w_5] = _pack_u16x2_0;
                                    uint32_t _pack_u16x2_1;
                                    asm("mov.b32 %0, {%1, %2};" : "=r"(_pack_u16x2_1) : "h"(_e4m3x2_2), "h"(_e4m3x2_3));
                                    v_out_3[2 * v_w_5 + 1] = _pack_u16x2_1;
                                }
                                int v_dst = ((u >> 2 == 0) ? v_dst_base0 : v_dst_base1);
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst + ((u & 3) * 2 << 4 ^ swz16))), "r"(v_out_3[0]), "r"(v_out_3[1]), "r"(v_out_3[2]), "r"(v_out_3[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst + ((u & 3) * 2 + 1 << 4 ^ swz16))), "r"(v_out_3[4]), "r"(v_out_3[5]), "r"(v_out_3[6]), "r"(v_out_3[7]) : "memory");
                            }
                        }
                        if (v_half != 0) {
                            int r_src_row = smem_k_a_addr + (unsigned int)(k_stage * 26624) + 16384 + (unsigned int)v_row_off;
                            int r_dst = p_stage * 32768 + v_key * 128 + 24576;
                            #pragma unroll
                            for (int r_u = 0; r_u < 2; r_u++) {
                                unsigned int r_out[4];
                                #pragma unroll
                                for (int r_half = 0; r_half < 2; r_half++) {
                                    unsigned int r_in[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&r_in[0])), "=r"(*reinterpret_cast<uint32_t*>(&r_in[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_in[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_in[(0) + 3]))
                                        : "r"(r_src_row + (2 * r_u + r_half << 4 ^ swz16)));
                                    #pragma unroll
                                    for (int r_w = 0; r_w < 2; r_w++) {
                                        float r_lo0 = __uint_as_float(r_in[2 * r_w] << 16);
                                        float r_hi0 = __uint_as_float(r_in[2 * r_w] & 4294901760u);
                                        float r_lo1 = __uint_as_float(r_in[2 * r_w + 1] << 16);
                                        float r_hi1 = __uint_as_float(r_in[2 * r_w + 1] & 4294901760u);
                                        if ((int)(kexp_u == 0) == 0) {
                                            r_lo0 = r_lo0 * kexp_scale;
                                            r_hi0 = r_hi0 * kexp_scale;
                                            r_lo1 = r_lo1 * kexp_scale;
                                            r_hi1 = r_hi1 * kexp_scale;
                                        }
                                        uint16_t _e4m3x2_f32_0;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(r_hi0), "f"(r_lo0));
                                        uint16_t _e4m3x2_f32_1;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1) : "f"(r_hi1), "f"(r_lo1));
                                        uint32_t _pack_u16x2_2;
                                        asm("mov.b32 %0, {%1, %2};" : "=r"(_pack_u16x2_2) : "h"(_e4m3x2_f32_0), "h"(_e4m3x2_f32_1));
                                        r_out[2 * r_half + r_w] = _pack_u16x2_2;
                                    }
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(r_dst + (4 + r_u << 4 ^ swz16))), "r"(r_out[0]), "r"(r_out[1]), "r"(r_out[2]), "r"(r_out[3]) : "memory");
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        if (ld_warp == 0) {
                            if (elect_sync()) {
                                mbarrier_arrive(kstage_free_addr + (k_stage) * 8);
                                mbarrier_arrive(v_full_addr + (p_stage) * 8);
                            }
                        }
                    }
                    ld_tile_cursor = ld_tile_cursor + ld_tiles;
                }
                mbarrier_wait(work_full_addr + (ld_work_stage) * 8, _phase_work_full_6);
                uint32_t _clc_valid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_2)
                    : "r"(work_response_addr + ld_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_2)
                    : "r"(work_response_addr + ld_work_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (ld_work_stage) * 8);
                ld_work_stage += 1;
                if (ld_work_stage == 2) { ld_work_stage = 0; _phase_work_full_6 ^= 1; }
                if (_clc_valid_2 == 0) {
                    break;
                }
                ld_query = _clc_ctaid_2;
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: gather ----
    if (warp >= 16 && warp <= 19) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
        { // gather_main
            const int g_warp = warp - 16;
            int g_tile_cursor = 0;
            unsigned int g_index_stage = 0;
            unsigned int g_work_stage = 0;
            int g_query = blockIdx.x;
            unsigned int _phase_index_full_1 = 0;
            unsigned int _phase_work_full_7 = 0;
            #pragma unroll 1
            for (unsigned int g_work = 0; g_work < 1048576; g_work++) {
                if (g_query < num_query_tokens) {
                    int _max_3 = ((sparse_topk_lens[g_query] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[g_query] + sparse_topk_lens_offset) : (0));
                    int _min_2 = ((_max_3) < (swa_width) ? (_max_3) : (swa_width));
                    int g_main_active = _min_2;
                    int g_extra_active = 0;
                    if (compressed_width > 0) {
                        int _max_4 = ((extra_topk_lens[g_query]) > (0) ? (extra_topk_lens[g_query]) : (0));
                        int _min_3 = ((_max_4) < (compressed_width) ? (_max_4) : (compressed_width));
                        g_extra_active = _min_3;
                    }
                    int g_n_main = (g_main_active + 64 - 1) / 64;
                    int g_n_extra = (g_extra_active + 64 - 1) / 64;
                    int _max_5 = ((g_n_main + g_n_extra) > (1) ? (g_n_main + g_n_extra) : (1));
                    int g_tiles = _max_5;
                    #pragma unroll 1
                    for (int g_tile = 0; g_tile < g_tiles; g_tile++) {
                        int gt = g_tile_cursor + g_tile;
                        int g_stage = gt % 3;
                        int g_slot_tile = g_tile & 3;
                        if (g_slot_tile == 0) {
                            mbarrier_wait(index_full_addr + (g_index_stage) * 8, _phase_index_full_1);
                        }
                        int g_free_stage = gt % 3;
                        int g_free_phase = gt / 3 & 1 ^ 1;
                        mbarrier_wait(k_empty_addr + (g_free_stage) * 8, g_free_phase);
                        mbarrier_wait(kstage_free_addr + (g_free_stage) * 8, g_free_phase);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        if (g_warp == 0) {
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(k_full_addr + (g_stage) * 8, 26624);
                            }
                        }
                        if (lane < 16) {
                            int g_group = lane;
                            int g_rows[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&g_rows[0])), "=r"(*reinterpret_cast<uint32_t*>(&g_rows[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g_rows[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g_rows[(0) + 3]))
                                : "r"(smem_indices_addr + g_index_stage * 1024 + (unsigned int)((g_slot_tile * 64 + g_group * 4) * 4)));
                            int g_use_main = g_n_main > g_tile || g_n_extra == 0;
                            int g_page_log2 = ((g_use_main != 0) ? swa_page_log2 : compressed_page_log2);
                            int g_pitch = ((g_use_main != 0) ? swa_pitch_units : compressed_pitch_units);
                            int g_footer_units = ((g_use_main != 0) ? swa_footer_units : compressed_footer_units);
                            int g_data_rows[4];
                            int g_foot_rows[4];
                            #pragma unroll
                            for (int g_j = 0; g_j < 4; g_j++) {
                                int g_tok = ((g_rows[g_j] >= 0) ? g_rows[g_j] : 0);
                                int g_page = g_tok >> g_page_log2;
                                int g_pslot = g_tok - (g_page << g_page_log2);
                                g_data_rows[g_j] = g_page * g_pitch + g_pslot * 11;
                                g_foot_rows[g_j] = g_page * g_pitch + g_footer_units + g_pslot;
                            }
                            int g_stage_base = smem_k_a_addr + (unsigned int)(g_stage * 26624);
                            if (g_warp < 3) {
                                int g_col = ((g_warp == 2) ? 224 : g_warp * 128);
                                int g_dst = g_stage_base + g_warp * 8192 + g_group * 512;
                                if (g_use_main != 0) {
                                    tma_gather4_gmem2smem(g_dst, (&tmap_swa_kv), g_col, g_data_rows[0], g_data_rows[1], g_data_rows[2], g_data_rows[3], k_full_addr + (g_stage) * 8);
                                } else {
                                    tma_gather4_gmem2smem(g_dst, (&tmap_compressed_kv), g_col, g_data_rows[0], g_data_rows[1], g_data_rows[2], g_data_rows[3], k_full_addr + (g_stage) * 8);
                                }
                            } else {
                                int g_fdst = g_stage_base + 24576 + g_group * 128;
                                if (g_use_main != 0) {
                                    tma_gather4_gmem2smem(g_fdst, (&tmap_swa_sf), 0, g_foot_rows[0], g_foot_rows[1], g_foot_rows[2], g_foot_rows[3], k_full_addr + (g_stage) * 8);
                                } else {
                                    tma_gather4_gmem2smem(g_fdst, (&tmap_compressed_sf), 0, g_foot_rows[0], g_foot_rows[1], g_foot_rows[2], g_foot_rows[3], k_full_addr + (g_stage) * 8);
                                }
                            }
                        }
                        if (g_slot_tile == 3 || g_tile + 1 == g_tiles) {
                            mbarrier_arrive(index_empty_addr + (g_index_stage) * 8);
                            g_index_stage += 1;
                            if (g_index_stage == 6) { g_index_stage = 0; _phase_index_full_1 ^= 1; }
                        }
                    }
                    g_tile_cursor = g_tile_cursor + g_tiles;
                }
                mbarrier_wait(work_full_addr + (g_work_stage) * 8, _phase_work_full_7);
                uint32_t _clc_valid_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_1)
                    : "r"(work_response_addr + g_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_1)
                    : "r"(work_response_addr + g_work_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (g_work_stage) * 8);
                g_work_stage += 1;
                if (g_work_stage == 2) { g_work_stage = 0; _phase_work_full_7 ^= 1; }
                if (_clc_valid_1 == 0) {
                    break;
                }
                g_query = _clc_ctaid_1;
            }
        }
    }
    // ---- Role: vx ----
    if (warp >= 20 && warp <= 23) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 72;");
        { // vx_main
            const int vx_thread = (warp - 20) * 32 + lane;
            const int vx_key = vx_thread & 63;
            const int vx_half = vx_thread >> 6;
            int vx_tile_cursor = 0;
            unsigned int vx_work_stage = 0;
            int vx_query = blockIdx.x;
            unsigned int _phase_work_full_8 = 0;
            #pragma unroll 1
            for (unsigned int vx_work = 0; vx_work < 1048576; vx_work++) {
                if (vx_query < num_query_tokens) {
                    int _max_9 = ((sparse_topk_lens[vx_query] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[vx_query] + sparse_topk_lens_offset) : (0));
                    int _min_6 = ((_max_9) < (swa_width) ? (_max_9) : (swa_width));
                    int vx_main_active = _min_6;
                    int vx_extra_active = 0;
                    if (compressed_width > 0) {
                        int _max_10 = ((extra_topk_lens[vx_query]) > (0) ? (extra_topk_lens[vx_query]) : (0));
                        int _min_7 = ((_max_10) < (compressed_width) ? (_max_10) : (compressed_width));
                        vx_extra_active = _min_7;
                    }
                    int _max_11 = (((vx_main_active + 64 - 1) / 64 + (vx_extra_active + 64 - 1) / 64) > (1) ? ((vx_main_active + 64 - 1) / 64 + (vx_extra_active + 64 - 1) / 64) : (1));
                    int vx_tiles = _max_11;
                    #pragma unroll 1
                    for (int vx_tile = 0; vx_tile < vx_tiles; vx_tile++) {
                        int xt = vx_tile_cursor + vx_tile;
                        int x_stage = xt & 1;
                        int x_phase = xt >> 1 & 1;
                        int xk_stage = xt % 3;
                        int xk_phase = xt / 3 & 1;
                        mbarrier_wait(k_full_addr + (xk_stage) * 8, xk_phase);
                        mbarrier_wait(kexp_full_addr + (x_stage) * 8, x_phase);
                        unsigned int x_kexp = smem_kexp[x_stage * 4];
                        unsigned int x_kexp_f16x2 = (15 - x_kexp << 10) * 65537;
                        float x_kexp_scale = __uint_as_float(127 - x_kexp << 23);
                        int swz16_1 = (vx_key & 7) << 4;
                        int v_row_off_1 = vx_key * 128;
                        int v_sf_addr_1 = smem_k_sf_addr + (unsigned int)(xk_stage * 26624) + (unsigned int)(vx_key * 32) + (unsigned int)(vx_half * 16);
                        int v_codes_src_1 = smem_k_a_addr + (unsigned int)(xk_stage * 26624) + (unsigned int)(vx_half * 8192) + (unsigned int)v_row_off_1;
                        int v_dst_base0_1 = x_stage * 32768 + vx_key * 128 + 2 * vx_half * 8192;
                        int v_dst_base1_1 = v_dst_base0_1 + 8192;
                        unsigned int v_sc_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&v_sc_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_sc_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v_sc_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v_sc_1[(0) + 3]))
                            : "r"(v_sf_addr_1));
                        if ((int)(x_kexp == 0) != 0) {
                            if (vx_half == 0) {
                                unsigned int codes3[4];
                                unsigned int codes4[4];
                                unsigned int codes5[4];
                                unsigned int codes6[4];
                                unsigned int codes7[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&codes3[0])), "=r"(*reinterpret_cast<uint32_t*>(&codes3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&codes3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&codes3[(0) + 3]))
                                    : "r"(v_codes_src_1 + (48 ^ swz16_1)));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&codes4[0])), "=r"(*reinterpret_cast<uint32_t*>(&codes4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&codes4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&codes4[(0) + 3]))
                                    : "r"(v_codes_src_1 + (64 ^ swz16_1)));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&codes5[0])), "=r"(*reinterpret_cast<uint32_t*>(&codes5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&codes5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&codes5[(0) + 3]))
                                    : "r"(v_codes_src_1 + (80 ^ swz16_1)));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&codes6[0])), "=r"(*reinterpret_cast<uint32_t*>(&codes6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&codes6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&codes6[(0) + 3]))
                                    : "r"(v_codes_src_1 + (96 ^ swz16_1)));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&codes7[0])), "=r"(*reinterpret_cast<uint32_t*>(&codes7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&codes7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&codes7[(0) + 3]))
                                    : "r"(v_codes_src_1 + (112 ^ swz16_1)));
                                unsigned int v_sc_lo_4 = v_sc_1[1] >> 16;
                                unsigned int v_sc_hi_5 = v_sc_1[1] >> 24;
                                unsigned int v_out_4[8];
                                #pragma unroll
                                for (int v_w_6 = 0; v_w_6 < 4; v_w_6++) {
                                    unsigned int v_sc_w_6 = ((v_w_6 < 2) ? v_sc_lo_4 : v_sc_hi_5);
                                    v_out_4[2 * v_w_6] = qmul4_lower4(codes3[v_w_6], v_sc_w_6);
                                    v_out_4[2 * v_w_6 + 1] = qmul4_higher4(codes3[v_w_6], v_sc_w_6);
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base0_1 + (96 ^ swz16_1))), "r"(v_out_4[0]), "r"(v_out_4[1]), "r"(v_out_4[2]), "r"(v_out_4[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base0_1 + (112 ^ swz16_1))), "r"(v_out_4[4]), "r"(v_out_4[5]), "r"(v_out_4[6]), "r"(v_out_4[7]) : "memory");
                                unsigned int v_sc_lo_0_2 = v_sc_1[2];
                                unsigned int v_sc_hi_1_2 = v_sc_1[2] >> 8;
                                unsigned int v_out_2_2[8];
                                #pragma unroll
                                for (int v_w_7 = 0; v_w_7 < 4; v_w_7++) {
                                    unsigned int v_sc_w_7 = ((v_w_7 < 2) ? v_sc_lo_0_2 : v_sc_hi_1_2);
                                    v_out_2_2[2 * v_w_7] = qmul4_lower4(codes4[v_w_7], v_sc_w_7);
                                    v_out_2_2[2 * v_w_7 + 1] = qmul4_higher4(codes4[v_w_7], v_sc_w_7);
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base1_1 + (0 ^ swz16_1))), "r"(v_out_2_2[0]), "r"(v_out_2_2[1]), "r"(v_out_2_2[2]), "r"(v_out_2_2[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base1_1 + (16 ^ swz16_1))), "r"(v_out_2_2[4]), "r"(v_out_2_2[5]), "r"(v_out_2_2[6]), "r"(v_out_2_2[7]) : "memory");
                                unsigned int v_sc_lo_3_1 = v_sc_1[2] >> 16;
                                unsigned int v_sc_hi_4_1 = v_sc_1[2] >> 24;
                                unsigned int v_out_5_1[8];
                                #pragma unroll
                                for (int v_w_8 = 0; v_w_8 < 4; v_w_8++) {
                                    unsigned int v_sc_w_8 = ((v_w_8 < 2) ? v_sc_lo_3_1 : v_sc_hi_4_1);
                                    v_out_5_1[2 * v_w_8] = qmul4_lower4(codes5[v_w_8], v_sc_w_8);
                                    v_out_5_1[2 * v_w_8 + 1] = qmul4_higher4(codes5[v_w_8], v_sc_w_8);
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base1_1 + (32 ^ swz16_1))), "r"(v_out_5_1[0]), "r"(v_out_5_1[1]), "r"(v_out_5_1[2]), "r"(v_out_5_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base1_1 + (48 ^ swz16_1))), "r"(v_out_5_1[4]), "r"(v_out_5_1[5]), "r"(v_out_5_1[6]), "r"(v_out_5_1[7]) : "memory");
                                unsigned int v_sc_lo_6 = v_sc_1[3];
                                unsigned int v_sc_hi_7 = v_sc_1[3] >> 8;
                                unsigned int v_out_8[8];
                                #pragma unroll
                                for (int v_w_9 = 0; v_w_9 < 4; v_w_9++) {
                                    unsigned int v_sc_w_9 = ((v_w_9 < 2) ? v_sc_lo_6 : v_sc_hi_7);
                                    v_out_8[2 * v_w_9] = qmul4_lower4(codes6[v_w_9], v_sc_w_9);
                                    v_out_8[2 * v_w_9 + 1] = qmul4_higher4(codes6[v_w_9], v_sc_w_9);
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base1_1 + (64 ^ swz16_1))), "r"(v_out_8[0]), "r"(v_out_8[1]), "r"(v_out_8[2]), "r"(v_out_8[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base1_1 + (80 ^ swz16_1))), "r"(v_out_8[4]), "r"(v_out_8[5]), "r"(v_out_8[6]), "r"(v_out_8[7]) : "memory");
                                unsigned int v_sc_lo_9 = v_sc_1[3] >> 16;
                                unsigned int v_sc_hi_10 = v_sc_1[3] >> 24;
                                unsigned int v_out_11[8];
                                #pragma unroll
                                for (int v_w_10 = 0; v_w_10 < 4; v_w_10++) {
                                    unsigned int v_sc_w_10 = ((v_w_10 < 2) ? v_sc_lo_9 : v_sc_hi_10);
                                    v_out_11[2 * v_w_10] = qmul4_lower4(codes7[v_w_10], v_sc_w_10);
                                    v_out_11[2 * v_w_10 + 1] = qmul4_higher4(codes7[v_w_10], v_sc_w_10);
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base1_1 + (96 ^ swz16_1))), "r"(v_out_11[0]), "r"(v_out_11[1]), "r"(v_out_11[2]), "r"(v_out_11[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base1_1 + (112 ^ swz16_1))), "r"(v_out_11[4]), "r"(v_out_11[5]), "r"(v_out_11[6]), "r"(v_out_11[7]) : "memory");
                            }
                            if (vx_half == 1) {
                                unsigned int codes2_1[4];
                                unsigned int codes3_1[4];
                                unsigned int codes4_1[4];
                                unsigned int codes5_1[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&codes2_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&codes2_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&codes2_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&codes2_1[(0) + 3]))
                                    : "r"(v_codes_src_1 + (32 ^ swz16_1)));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&codes3_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&codes3_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&codes3_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&codes3_1[(0) + 3]))
                                    : "r"(v_codes_src_1 + (48 ^ swz16_1)));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&codes4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&codes4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&codes4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&codes4_1[(0) + 3]))
                                    : "r"(v_codes_src_1 + (64 ^ swz16_1)));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&codes5_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&codes5_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&codes5_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&codes5_1[(0) + 3]))
                                    : "r"(v_codes_src_1 + (80 ^ swz16_1)));
                                unsigned int v_sc_lo_5 = v_sc_1[1];
                                unsigned int v_sc_hi_6 = v_sc_1[1] >> 8;
                                unsigned int v_out_6[8];
                                #pragma unroll
                                for (int v_w_11 = 0; v_w_11 < 4; v_w_11++) {
                                    unsigned int v_sc_w_11 = ((v_w_11 < 2) ? v_sc_lo_5 : v_sc_hi_6);
                                    v_out_6[2 * v_w_11] = qmul4_lower4(codes2_1[v_w_11], v_sc_w_11);
                                    v_out_6[2 * v_w_11 + 1] = qmul4_higher4(codes2_1[v_w_11], v_sc_w_11);
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base0_1 + (64 ^ swz16_1))), "r"(v_out_6[0]), "r"(v_out_6[1]), "r"(v_out_6[2]), "r"(v_out_6[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base0_1 + (80 ^ swz16_1))), "r"(v_out_6[4]), "r"(v_out_6[5]), "r"(v_out_6[6]), "r"(v_out_6[7]) : "memory");
                                unsigned int v_sc_lo_0_3 = v_sc_1[1] >> 16;
                                unsigned int v_sc_hi_1_3 = v_sc_1[1] >> 24;
                                unsigned int v_out_2_3[8];
                                #pragma unroll
                                for (int v_w_12 = 0; v_w_12 < 4; v_w_12++) {
                                    unsigned int v_sc_w_12 = ((v_w_12 < 2) ? v_sc_lo_0_3 : v_sc_hi_1_3);
                                    v_out_2_3[2 * v_w_12] = qmul4_lower4(codes3_1[v_w_12], v_sc_w_12);
                                    v_out_2_3[2 * v_w_12 + 1] = qmul4_higher4(codes3_1[v_w_12], v_sc_w_12);
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base0_1 + (96 ^ swz16_1))), "r"(v_out_2_3[0]), "r"(v_out_2_3[1]), "r"(v_out_2_3[2]), "r"(v_out_2_3[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base0_1 + (112 ^ swz16_1))), "r"(v_out_2_3[4]), "r"(v_out_2_3[5]), "r"(v_out_2_3[6]), "r"(v_out_2_3[7]) : "memory");
                                unsigned int v_sc_lo_3_2 = v_sc_1[2];
                                unsigned int v_sc_hi_4_2 = v_sc_1[2] >> 8;
                                unsigned int v_out_5_2[8];
                                #pragma unroll
                                for (int v_w_13 = 0; v_w_13 < 4; v_w_13++) {
                                    unsigned int v_sc_w_13 = ((v_w_13 < 2) ? v_sc_lo_3_2 : v_sc_hi_4_2);
                                    v_out_5_2[2 * v_w_13] = qmul4_lower4(codes4_1[v_w_13], v_sc_w_13);
                                    v_out_5_2[2 * v_w_13 + 1] = qmul4_higher4(codes4_1[v_w_13], v_sc_w_13);
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base1_1 + (0 ^ swz16_1))), "r"(v_out_5_2[0]), "r"(v_out_5_2[1]), "r"(v_out_5_2[2]), "r"(v_out_5_2[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base1_1 + (16 ^ swz16_1))), "r"(v_out_5_2[4]), "r"(v_out_5_2[5]), "r"(v_out_5_2[6]), "r"(v_out_5_2[7]) : "memory");
                                unsigned int v_sc_lo_6_1 = v_sc_1[2] >> 16;
                                unsigned int v_sc_hi_7_1 = v_sc_1[2] >> 24;
                                unsigned int v_out_8_1[8];
                                #pragma unroll
                                for (int v_w_14 = 0; v_w_14 < 4; v_w_14++) {
                                    unsigned int v_sc_w_14 = ((v_w_14 < 2) ? v_sc_lo_6_1 : v_sc_hi_7_1);
                                    v_out_8_1[2 * v_w_14] = qmul4_lower4(codes5_1[v_w_14], v_sc_w_14);
                                    v_out_8_1[2 * v_w_14 + 1] = qmul4_higher4(codes5_1[v_w_14], v_sc_w_14);
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base1_1 + (32 ^ swz16_1))), "r"(v_out_8_1[0]), "r"(v_out_8_1[1]), "r"(v_out_8_1[2]), "r"(v_out_8_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base1_1 + (48 ^ swz16_1))), "r"(v_out_8_1[4]), "r"(v_out_8_1[5]), "r"(v_out_8_1[6]), "r"(v_out_8_1[7]) : "memory");
                            }
                        } else {
                            #pragma unroll 1
                            for (int u_1 = 3 - vx_half; u_1 < 8 - 2 * vx_half; u_1++) {
                                unsigned int v_in_1[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&v_in_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_in_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v_in_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v_in_1[(0) + 3]))
                                    : "r"(v_codes_src_1 + (u_1 << 4 ^ swz16_1)));
                                int v_w_idx_1 = u_1 >> 1;
                                unsigned int v_sc_lo_7 = ((v_w_idx_1 == 0) ? v_sc_1[0] : v_sc_1[1]);
                                unsigned int v_sc_hi_8 = ((v_w_idx_1 == 2) ? v_sc_1[2] : v_sc_1[3]);
                                unsigned int v_sc_word_1 = ((v_w_idx_1 < 2) ? v_sc_lo_7 : v_sc_hi_8);
                                uint32_t _e4m3x2_to_f16x2_1;
                                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_1) : "h"((uint16_t)(v_sc_word_1 >> (unsigned int)((u_1 & 1) * 16))));
                                unsigned int v_sc_pair_1 = _e4m3x2_to_f16x2_1;
                                uint32_t _f16x2_mul_5;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_5) : "r"(v_sc_pair_1), "r"(x_kexp_f16x2));
                                v_sc_pair_1 = _f16x2_mul_5;
                                uint32_t _prmt_b32_2;
                                asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_b32_2) : "r"(v_sc_pair_1), "r"(v_sc_pair_1));
                                unsigned int v_sc0_1 = _prmt_b32_2;
                                uint32_t _prmt_b32_3;
                                asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_b32_3) : "r"(v_sc_pair_1), "r"(v_sc_pair_1));
                                unsigned int v_sc1_1 = _prmt_b32_3;
                                unsigned int v_out_7[8];
                                #pragma unroll
                                for (int v_w_15 = 0; v_w_15 < 4; v_w_15++) {
                                    unsigned int v_sc_w_15 = ((v_w_15 < 2) ? v_sc0_1 : v_sc1_1);
                                    unsigned int v_word_1 = v_in_1[v_w_15];
                                    uint32_t _e2m1x2_to_f16x2_4;
                                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_4) : "r"((uint32_t)(v_word_1)));
                                    uint32_t _f16x2_mul_6;
                                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_6) : "r"(_e2m1x2_to_f16x2_4), "r"(v_sc_w_15));
                                    uint16_t _e4m3x2_4;
                                    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_e4m3x2_4) : "r"(_f16x2_mul_6));
                                    uint32_t _e2m1x2_to_f16x2_5;
                                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_5) : "r"((uint32_t)(v_word_1 >> 8)));
                                    uint32_t _f16x2_mul_7;
                                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_7) : "r"(_e2m1x2_to_f16x2_5), "r"(v_sc_w_15));
                                    uint16_t _e4m3x2_5;
                                    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_e4m3x2_5) : "r"(_f16x2_mul_7));
                                    uint32_t _e2m1x2_to_f16x2_6;
                                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_6) : "r"((uint32_t)(v_word_1 >> 16)));
                                    uint32_t _f16x2_mul_8;
                                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_8) : "r"(_e2m1x2_to_f16x2_6), "r"(v_sc_w_15));
                                    uint16_t _e4m3x2_6;
                                    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_e4m3x2_6) : "r"(_f16x2_mul_8));
                                    uint32_t _e2m1x2_to_f16x2_7;
                                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_7) : "r"((uint32_t)(v_word_1 >> 24)));
                                    uint32_t _f16x2_mul_9;
                                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_9) : "r"(_e2m1x2_to_f16x2_7), "r"(v_sc_w_15));
                                    uint16_t _e4m3x2_7;
                                    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_e4m3x2_7) : "r"(_f16x2_mul_9));
                                    uint32_t _pack_u16x2_3;
                                    asm("mov.b32 %0, {%1, %2};" : "=r"(_pack_u16x2_3) : "h"(_e4m3x2_4), "h"(_e4m3x2_5));
                                    v_out_7[2 * v_w_15] = _pack_u16x2_3;
                                    uint32_t _pack_u16x2_4;
                                    asm("mov.b32 %0, {%1, %2};" : "=r"(_pack_u16x2_4) : "h"(_e4m3x2_6), "h"(_e4m3x2_7));
                                    v_out_7[2 * v_w_15 + 1] = _pack_u16x2_4;
                                }
                                int v_dst_1 = ((u_1 >> 2 == 0) ? v_dst_base0_1 : v_dst_base1_1);
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_1 + ((u_1 & 3) * 2 << 4 ^ swz16_1))), "r"(v_out_7[0]), "r"(v_out_7[1]), "r"(v_out_7[2]), "r"(v_out_7[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_1 + ((u_1 & 3) * 2 + 1 << 4 ^ swz16_1))), "r"(v_out_7[4]), "r"(v_out_7[5]), "r"(v_out_7[6]), "r"(v_out_7[7]) : "memory");
                            }
                        }
                        if (vx_half != 0) {
                            int r_src_row_1 = smem_k_a_addr + (unsigned int)(xk_stage * 26624) + 16384 + (unsigned int)v_row_off_1;
                            int r_dst_1 = x_stage * 32768 + vx_key * 128 + 24576;
                            #pragma unroll
                            for (int r_u_1 = 2; r_u_1 < 4; r_u_1++) {
                                unsigned int r_out_1[4];
                                #pragma unroll
                                for (int r_half_1 = 0; r_half_1 < 2; r_half_1++) {
                                    unsigned int r_in_1[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&r_in_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&r_in_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_in_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_in_1[(0) + 3]))
                                        : "r"(r_src_row_1 + (2 * r_u_1 + r_half_1 << 4 ^ swz16_1)));
                                    #pragma unroll
                                    for (int r_w_1 = 0; r_w_1 < 2; r_w_1++) {
                                        float r_lo0_1 = __uint_as_float(r_in_1[2 * r_w_1] << 16);
                                        float r_hi0_1 = __uint_as_float(r_in_1[2 * r_w_1] & 4294901760u);
                                        float r_lo1_1 = __uint_as_float(r_in_1[2 * r_w_1 + 1] << 16);
                                        float r_hi1_1 = __uint_as_float(r_in_1[2 * r_w_1 + 1] & 4294901760u);
                                        if ((int)(x_kexp == 0) == 0) {
                                            r_lo0_1 = r_lo0_1 * x_kexp_scale;
                                            r_hi0_1 = r_hi0_1 * x_kexp_scale;
                                            r_lo1_1 = r_lo1_1 * x_kexp_scale;
                                            r_hi1_1 = r_hi1_1 * x_kexp_scale;
                                        }
                                        uint16_t _e4m3x2_f32_2;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_2) : "f"(r_hi0_1), "f"(r_lo0_1));
                                        uint16_t _e4m3x2_f32_3;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_3) : "f"(r_hi1_1), "f"(r_lo1_1));
                                        uint32_t _pack_u16x2_5;
                                        asm("mov.b32 %0, {%1, %2};" : "=r"(_pack_u16x2_5) : "h"(_e4m3x2_f32_2), "h"(_e4m3x2_f32_3));
                                        r_out_1[2 * r_half_1 + r_w_1] = _pack_u16x2_5;
                                    }
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(r_dst_1 + (4 + r_u_1 << 4 ^ swz16_1))), "r"(r_out_1[0]), "r"(r_out_1[1]), "r"(r_out_1[2]), "r"(r_out_1[3]) : "memory");
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 10, 128;" ::: "memory");
                        if (vx_thread == 0) {
                            mbarrier_arrive(kstage_free_addr + (xk_stage) * 8);
                            mbarrier_arrive(v_full_addr + (x_stage) * 8);
                        }
                    }
                    vx_tile_cursor = vx_tile_cursor + vx_tiles;
                }
                mbarrier_wait(work_full_addr + (vx_work_stage) * 8, _phase_work_full_8);
                uint32_t _clc_valid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_3)
                    : "r"(work_response_addr + vx_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_3)
                    : "r"(work_response_addr + vx_work_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (vx_work_stage) * 8);
                vx_work_stage += 1;
                if (vx_work_stage == 2) { vx_work_stage = 0; _phase_work_full_8 ^= 1; }
                if (_clc_valid_3 == 0) {
                    break;
                }
                vx_query = _clc_ctaid_3;
            }
        }
    }

    // Cleanup
}

} // extern "C"

