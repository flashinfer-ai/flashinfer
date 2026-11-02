/*
 * Copyright (c) 2023 by FlashInfer team.
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
#include "cake_dsv4_nvfp4_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_TMEM_S_OFFSET 0
#define TMEM_TMEM_O0_OFFSET 128
#define TMEM_TMEM_O1_OFFSET 192
#define TMEM_TMEM_O2_OFFSET 256
#define TMEM_TMEM_O3_OFFSET 320
#define TMEM_TMEM_SFQ0_OFFSET 384
#define TMEM_TMEM_SFQ1_OFFSET 400
#define TMEM_TMEM_SFK0_OFFSET 416
#define TMEM_TMEM_SFK1_OFFSET 432
#define TMEM_TMEM_Q0_OFFSET 448
#define TMEM_TMEM_Q1_OFFSET 480
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QSF_OFF 19456
#define SMEM_SMEM_QSF_STAGE_BYTES 2048
#define SMEM_SMEM_QSF_STRIDE 2048
#define SMEM_SMEM_QSF32_OFF 19456
#define SMEM_SMEM_QSF32_STAGE_BYTES 4096
#define SMEM_SMEM_QSF32_STRIDE 4096
#define SMEM_SMEM_QROPE_OFF 23552
#define SMEM_SMEM_QROPE_STAGE_BYTES 16384
#define SMEM_SMEM_QROPE_STRIDE 16384
#define SMEM_SMEM_QSTAGE_OFF 150528
#define SMEM_SMEM_QSTAGE_STAGE_BYTES 8192
#define SMEM_SMEM_QSTAGE_STRIDE 8192
#define SMEM_SMEM_KSF_OFF 39936
#define SMEM_SMEM_KSF_STAGE_BYTES 2048
#define SMEM_SMEM_KSF_STRIDE 2048
#define SMEM_SMEM_KSF32_OFF 39936
#define SMEM_SMEM_KSF32_STAGE_BYTES 4096
#define SMEM_SMEM_KSF32_STRIDE 4096
#define SMEM_SMEM_KF4_0_OFF 44032
#define SMEM_SMEM_KF4_0_STAGE_BYTES 16384
#define SMEM_SMEM_KF4_0_STRIDE 16384
#define SMEM_SMEM_KF4_1_OFF 97280
#define SMEM_SMEM_KF4_1_STAGE_BYTES 16384
#define SMEM_SMEM_KF4_1_STRIDE 16384
#define SMEM_SMEM_KROPE_0_OFF 76800
#define SMEM_SMEM_KROPE_0_STAGE_BYTES 16384
#define SMEM_SMEM_KROPE_0_STRIDE 16384
#define SMEM_SMEM_KROPE_1_OFF 130048
#define SMEM_SMEM_KROPE_1_STAGE_BYTES 16384
#define SMEM_SMEM_KROPE_1_STRIDE 16384
#define SMEM_SMEM_SFS_0_OFF 93184
#define SMEM_SMEM_SFS_0_STAGE_BYTES 4096
#define SMEM_SMEM_SFS_0_STRIDE 4096
#define SMEM_SMEM_SFS_1_OFF 146432
#define SMEM_SMEM_SFS_1_STAGE_BYTES 4096
#define SMEM_SMEM_SFS_1_STRIDE 4096
#define SMEM_SMEM_KZ32V_OFF 44032
#define SMEM_SMEM_KZ32V_STAGE_BYTES 106496
#define SMEM_SMEM_KZ32V_STRIDE 106496
#define SMEM_SMEM_KZ32_OFF 44032
#define SMEM_SMEM_KZ32_STAGE_BYTES 106496
#define SMEM_SMEM_KZ32_STRIDE 106496
#define SMEM_SMEM_KW32_OFF 17408
#define SMEM_SMEM_KW32_STAGE_BYTES 2048
#define SMEM_SMEM_KW32_STRIDE 2048
#define SMEM_SMEM_ROWOFF_0_OFF 217872
#define SMEM_SMEM_ROWOFF_0_STAGE_BYTES 2048
#define SMEM_SMEM_ROWOFF_0_STRIDE 2048
#define SMEM_SMEM_ROWOFF_1_OFF 219920
#define SMEM_SMEM_ROWOFF_1_STAGE_BYTES 2048
#define SMEM_SMEM_ROWOFF_1_STRIDE 2048
#define SMEM_SMEM_TOK32V_OFF 17408
#define SMEM_SMEM_TOK32V_STAGE_BYTES 2048
#define SMEM_SMEM_TOK32V_STRIDE 2048
#define SMEM_SMEM_P_0_OFF 1024
#define SMEM_SMEM_P_0_STAGE_BYTES 8192
#define SMEM_SMEM_P_0_STRIDE 8192
#define SMEM_SMEM_P_1_OFF 9216
#define SMEM_SMEM_P_1_STAGE_BYTES 8192
#define SMEM_SMEM_P_1_STRIDE 8192
#define SMEM_SMEM_V_OFF 150528
#define SMEM_SMEM_V_STAGE_BYTES 16384
#define SMEM_SMEM_V_STRIDE 16384
#define SMEM_SMEM_FLAG_OFF 216064
#define SMEM_SMEM_FLAG_STAGE_BYTES 16
#define SMEM_SMEM_FLAG_STRIDE 16
#define SMEM_SMEM_ALPHA_OFF 216080
#define SMEM_SMEM_ALPHA_STAGE_BYTES 256
#define SMEM_SMEM_ALPHA_STRIDE 256
#define SMEM_SMEM_PMAX_OFF 216336
#define SMEM_SMEM_PMAX_STAGE_BYTES 1536
#define SMEM_SMEM_PMAX_STRIDE 1536
#define SMEM_SMEM_NORM_OFF 216336
#define SMEM_SMEM_NORM_STAGE_BYTES 256
#define SMEM_SMEM_NORM_STRIDE 256
#define SMEM_SMEM_XSUM_OFF 44032
#define SMEM_SMEM_XSUM_STAGE_BYTES 3072
#define SMEM_SMEM_XSUM_STRIDE 3072
#define SMEM_SMEM_O32_OFF 150528
#define SMEM_SMEM_O32_STAGE_BYTES 65536
#define SMEM_SMEM_O32_STRIDE 65536
#define SMEM_SMEM_O16_OFF 150528
#define SMEM_SMEM_O16_STAGE_BYTES 65536
#define SMEM_SMEM_O16_STRIDE 65536
#define SMEM_TOTAL 222080
#define THREADS 512
#define LAUNCH_MIN_BLOCKS 1

extern "C" {

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) void
kernel_cake_dsv4_nvfp4_3c1fd27d4ebde48da03d(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_out, const __grid_constant__ CUtensorMap tmap_g4d, const __grid_constant__ CUtensorMap tmap_g4f, const __grid_constant__ CUtensorMap tmap_g4dx, const __grid_constant__ CUtensorMap tmap_g4fx, __nv_bfloat16* __restrict__ q_rows, uint8_t* __restrict__ main_cache, uint8_t* __restrict__ extra_cache, int* __restrict__ main_indices, int* __restrict__ extra_indices, int* __restrict__ main_lengths, int* __restrict__ extra_lengths, float* __restrict__ sinks, float* __restrict__ bmm1_scale, float* __restrict__ bmm2_scale, __nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_lse, __nv_bfloat16* __restrict__ O, float* __restrict__ lse_out, int num_heads, int num_head_tiles, int num_splits, int num_main_tiles, int tiles_per_split, int main_width, int extra_width, int main_index_stride, int extra_index_stride, int has_main_lengths, int has_extra_lengths, int main_page_shift, int extra_page_shift, long long main_page_stride, long long extra_page_stride, int has_sinks, float lse_partial_scale, float lse_scale)
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
    #define q_rope_full_addr (mbar_base + 0)
    #define q_nope_full0_addr (mbar_base + 8)
    #define q_nope_full1_addr (mbar_base + 16)
    #define q_nope_full2_addr (mbar_base + 24)
    #define q_ready_addr (mbar_base + 32)
    #define kv_full_addr (mbar_base + 40)
    #define v_full_addr (mbar_base + 48)
    #define s_full_addr (mbar_base + 56)
    #define p_full_addr (mbar_base + 64)
    #define o_full_addr (mbar_base + 72)
    #define tmem_dealloc_addr (mbar_base + 104)
    #define tok_full_addr (mbar_base + 112)
    #define tok_free_addr (mbar_base + 128)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* smem_qsf = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_QSF_OFF);
    const int smem_qsf_addr = smem + SMEM_SMEM_QSF_OFF;
    unsigned int* smem_qsf32 = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_QSF32_OFF);
    const int smem_qsf32_addr = smem + SMEM_SMEM_QSF32_OFF;
    __nv_bfloat16* smem_qrope = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_QROPE_OFF);
    const int smem_qrope_addr = smem + SMEM_SMEM_QROPE_OFF;
    __nv_bfloat16* smem_qstage = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_QSTAGE_OFF);
    const int smem_qstage_addr = smem + SMEM_SMEM_QSTAGE_OFF;
    uint8_t* smem_ksf = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_KSF_OFF);
    const int smem_ksf_addr = smem + SMEM_SMEM_KSF_OFF;
    unsigned int* smem_ksf32 = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_KSF32_OFF);
    const int smem_ksf32_addr = smem + SMEM_SMEM_KSF32_OFF;
    uint8_t* smem_kf4_0 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_KF4_0_OFF);
    const int smem_kf4_0_addr = smem + SMEM_SMEM_KF4_0_OFF;
    uint8_t* smem_kf4_1 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_KF4_1_OFF);
    const int smem_kf4_1_addr = smem + SMEM_SMEM_KF4_1_OFF;
    __nv_bfloat16* smem_krope_0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_KROPE_0_OFF);
    const int smem_krope_0_addr = smem + SMEM_SMEM_KROPE_0_OFF;
    __nv_bfloat16* smem_krope_1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_KROPE_1_OFF);
    const int smem_krope_1_addr = smem + SMEM_SMEM_KROPE_1_OFF;
    uint8_t* smem_sfs_0 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_SFS_0_OFF);
    const int smem_sfs_0_addr = smem + SMEM_SMEM_SFS_0_OFF;
    uint8_t* smem_sfs_1 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_SFS_1_OFF);
    const int smem_sfs_1_addr = smem + SMEM_SMEM_SFS_1_OFF;
    int* smem_kz32v = reinterpret_cast<int*>(smem_raw + SMEM_SMEM_KZ32V_OFF);
    const int smem_kz32v_addr = smem + SMEM_SMEM_KZ32V_OFF;
    unsigned int* smem_kz32 = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_KZ32_OFF);
    const int smem_kz32_addr = smem + SMEM_SMEM_KZ32_OFF;
    int* smem_kw32 = reinterpret_cast<int*>(smem_raw + SMEM_SMEM_KW32_OFF);
    const int smem_kw32_addr = smem + SMEM_SMEM_KW32_OFF;
    unsigned int* smem_rowoff_0 = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_ROWOFF_0_OFF);
    const int smem_rowoff_0_addr = smem + SMEM_SMEM_ROWOFF_0_OFF;
    unsigned int* smem_rowoff_1 = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_ROWOFF_1_OFF);
    const int smem_rowoff_1_addr = smem + SMEM_SMEM_ROWOFF_1_OFF;
    int* smem_tok32v = reinterpret_cast<int*>(smem_raw + SMEM_SMEM_TOK32V_OFF);
    const int smem_tok32v_addr = smem + SMEM_SMEM_TOK32V_OFF;
    uint8_t* smem_p_0 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_P_0_OFF);
    const int smem_p_0_addr = smem + SMEM_SMEM_P_0_OFF;
    uint8_t* smem_p_1 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_P_1_OFF);
    const int smem_p_1_addr = smem + SMEM_SMEM_P_1_OFF;
    uint8_t* smem_v = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V_OFF);
    const int smem_v_addr = smem + SMEM_SMEM_V_OFF;
    unsigned int* smem_flag = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_FLAG_OFF);
    const int smem_flag_addr = smem + SMEM_SMEM_FLAG_OFF;
    float* smem_alpha = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_ALPHA_OFF);
    const int smem_alpha_addr = smem + SMEM_SMEM_ALPHA_OFF;
    float* smem_pmax = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_PMAX_OFF);
    const int smem_pmax_addr = smem + SMEM_SMEM_PMAX_OFF;
    float* smem_norm = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_NORM_OFF);
    const int smem_norm_addr = smem + SMEM_SMEM_NORM_OFF;
    float* smem_xsum = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_XSUM_OFF);
    const int smem_xsum_addr = smem + SMEM_SMEM_XSUM_OFF;
    unsigned int* smem_o32 = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_O32_OFF);
    const int smem_o32_addr = smem + SMEM_SMEM_O32_OFF;
    uint16_t* smem_o16 = reinterpret_cast<uint16_t*>(smem_raw + SMEM_SMEM_O16_OFF);
    const int smem_o16_addr = smem + SMEM_SMEM_O16_OFF;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&tmap_q))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&tmap_out))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&tmap_g4d))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&tmap_g4f))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&tmap_g4dx))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&tmap_g4fx))) : "memory");

    // Mbarrier init (13 pipeline groups, 0 ordered-sequence groups, 18 barriers)
    // Mbarriers at smem_raw[0..144)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_rope_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_nope_full0: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // q_nope_full1: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            // q_nope_full2: 1 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            // q_ready: 1 barriers, init_count=384
            mbarrier_init(smem + 32, 384);
            // kv_full: 1 barriers, init_count=384
            mbarrier_init(smem + 40, 384);
            // v_full: 1 barriers, init_count=384
            mbarrier_init(smem + 48, 384);
            // s_full: 1 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            // p_full: 1 barriers, init_count=384
            mbarrier_init(smem + 64, 384);
            // o_full: 4 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            // tmem_dealloc: 1 barriers, init_count=384
            mbarrier_init(smem + 104, 384);
            // tok_full: 2 barriers, init_count=96
            mbarrier_init(smem + 112, 96);
            mbarrier_init(smem + 120, 96);
            // tok_free: 2 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 144);
    if (warp == 0) {
        int _tmem_hold = smem + 144;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_s = taddr;
    const int tmem_tmem_o0 = taddr + 128;
    const int tmem_tmem_o1 = taddr + 192;
    const int tmem_tmem_o2 = taddr + 256;
    const int tmem_tmem_o3 = taddr + 320;
    const int tmem_tmem_sfq0 = taddr + 384;
    const int tmem_tmem_sfq1 = taddr + 400;
    const int tmem_tmem_sfk0 = taddr + 416;
    const int tmem_tmem_sfk1 = taddr + 432;
    const int tmem_tmem_q0 = taddr + 448;
    const int tmem_tmem_q1 = taddr + 480;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
    }

    // ---- Role: compute0 ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 144;");
        { // compute0_main
            const int local_warp = warp;
            const int half = local_warp / 2;
            int o_chunk = 0;
            int work_idx = blockIdx.x;
            int head_tile = work_idx % num_head_tiles;
            int split_work = work_idx / num_head_tiles;
            int split_idx = split_work % num_splits;
            int query_idx = split_work / num_splits;
            int head_base = head_tile * 64;
            const int row = local_warp * 32 + lane;
            const int head = row & 63;
            const int tmem_row_origin = local_warp * 32;
            float inv_six = 0.16666666666666666f;
            unsigned int _phase_q_nope_full0_0 = 0;
            mbarrier_wait_hint(q_nope_full0_addr, _phase_q_nope_full0_0, 10000000);
            _phase_q_nope_full0_0 ^= 1;
            unsigned int _phase_q_nope_full1_0 = 0;
            mbarrier_wait_hint(q_nope_full1_addr, _phase_q_nope_full1_0, 10000000);
            _phase_q_nope_full1_0 ^= 1;
            unsigned int _phase_q_nope_full2_0 = 0;
            mbarrier_wait_hint(q_nope_full2_addr, _phase_q_nope_full2_0, 10000000);
            _phase_q_nope_full2_0 ^= 1;
            int q_taddr = taddr + 448 + (unsigned int)(tmem_row_origin << 16);
            const int q_par = local_warp % 2;
            const int q_slot = local_warp / 2;
            int q_block_live = ((head_base + q_par * 32 < num_heads) ? 1 : 0);
            int q_head = q_par * 32 + lane;
            int exch_lane = smem_p_0_addr + (unsigned int)(lane * 32);
            unsigned int zero8[8];
            zero8[0] = 0;
            zero8[1] = 0;
            zero8[2] = 0;
            zero8[3] = 0;
            zero8[4] = 0;
            zero8[5] = 0;
            zero8[6] = 0;
            zero8[7] = 0;
            int kset_u = ((1) ? q_slot : 6);
            int do_u = ((1) ? 1 : ((q_slot == 0) ? 1 : 0));
            if (do_u != 0) {
                int exch_u = exch_lane + (q_par * 7 + kset_u) * 1024;
                if (q_block_live != 0) {
                    int q_row_addr = smem_qstage_addr + (unsigned int)(kset_u * 8192) + (unsigned int)(q_head * 128);
                    unsigned int words[8];
                    unsigned int sf_word = 0;
                    unsigned int qa[4];
                    unsigned int qb[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qa[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa[(0) + 3]))
                        : "r"(q_row_addr + (0 ^ row % 8) * 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qb[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb[(0) + 3]))
                        : "r"(q_row_addr + (1 ^ row % 8) * 16));
                    float qv[16];
                    qv[0] = __uint_as_float(qa[0] << 16);
                    qv[1] = __uint_as_float(qa[0] & 4294901760u);
                    qv[8] = __uint_as_float(qb[0] << 16);
                    qv[9] = __uint_as_float(qb[0] & 4294901760u);
                    qv[2] = __uint_as_float(qa[1] << 16);
                    qv[3] = __uint_as_float(qa[1] & 4294901760u);
                    qv[10] = __uint_as_float(qb[1] << 16);
                    qv[11] = __uint_as_float(qb[1] & 4294901760u);
                    qv[4] = __uint_as_float(qa[2] << 16);
                    qv[5] = __uint_as_float(qa[2] & 4294901760u);
                    qv[12] = __uint_as_float(qb[2] << 16);
                    qv[13] = __uint_as_float(qb[2] & 4294901760u);
                    qv[6] = __uint_as_float(qa[3] << 16);
                    qv[7] = __uint_as_float(qa[3] & 4294901760u);
                    qv[14] = __uint_as_float(qb[3] << 16);
                    qv[15] = __uint_as_float(qb[3] & 4294901760u);
                    float m8[8];
                    float _fabs_0 = fabsf(qv[0]);
                    float _fabs_1 = fabsf(qv[1]);
                    float _max_0 = max_noftz(_fabs_0, _fabs_1);
                    m8[0] = _max_0;
                    float _fabs_2 = fabsf(qv[2]);
                    float _fabs_3 = fabsf(qv[3]);
                    float _max_1 = max_noftz(_fabs_2, _fabs_3);
                    m8[1] = _max_1;
                    float _fabs_4 = fabsf(qv[4]);
                    float _fabs_5 = fabsf(qv[5]);
                    float _max_2 = max_noftz(_fabs_4, _fabs_5);
                    m8[2] = _max_2;
                    float _fabs_6 = fabsf(qv[6]);
                    float _fabs_7 = fabsf(qv[7]);
                    float _max_3 = max_noftz(_fabs_6, _fabs_7);
                    m8[3] = _max_3;
                    float _fabs_8 = fabsf(qv[8]);
                    float _fabs_9 = fabsf(qv[9]);
                    float _max_4 = max_noftz(_fabs_8, _fabs_9);
                    m8[4] = _max_4;
                    float _fabs_10 = fabsf(qv[10]);
                    float _fabs_11 = fabsf(qv[11]);
                    float _max_5 = max_noftz(_fabs_10, _fabs_11);
                    m8[5] = _max_5;
                    float _fabs_12 = fabsf(qv[12]);
                    float _fabs_13 = fabsf(qv[13]);
                    float _max_6 = max_noftz(_fabs_12, _fabs_13);
                    m8[6] = _max_6;
                    float _fabs_14 = fabsf(qv[14]);
                    float _fabs_15 = fabsf(qv[15]);
                    float _max_7 = max_noftz(_fabs_14, _fabs_15);
                    m8[7] = _max_7;
                    float m4[4];
                    float _max_8 = max_noftz(m8[0], m8[1]);
                    m4[0] = _max_8;
                    float _max_9 = max_noftz(m8[2], m8[3]);
                    m4[1] = _max_9;
                    float _max_10 = max_noftz(m8[4], m8[5]);
                    m4[2] = _max_10;
                    float _max_11 = max_noftz(m8[6], m8[7]);
                    m4[3] = _max_11;
                    float _max_12 = max_noftz(m4[0], m4[1]);
                    float _max_13 = max_noftz(m4[2], m4[3]);
                    float _max_14 = max_noftz(_max_12, _max_13);
                    float amax = _max_14;
                    float sc = amax * inv_six;
                    uint16_t _e4m3x2_f32_0;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(0.0f), "f"(sc));
                    uint16_t sc_pair = _e4m3x2_f32_0;
                    unsigned int sc_byte = (unsigned int)sc_pair & 255;
                    unsigned int sc_exp = sc_byte >> 3 & 15;
                    unsigned int sc_man = sc_byte & 7;
                    float sc_norm = __uint_as_float(sc_exp + 120 << 23 | sc_man << 20);
                    float sc_sub = (float)sc_man * 0.001953125f;
                    float sc_dec = ((sc_exp == 0) ? sc_sub : sc_norm);
                    float _rcp_0 = __frcp_rn(sc_dec);
                    float inv = ((sc_dec > 0.0f) ? _rcp_0 : 0.0f);
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_0 = {inv, inv};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qv)[_ls], _scale2_0);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        qv[_ls] = qv[_ls] * inv;
                    }
                    #endif
                    uint32_t _fp4_pair_0;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_0) : "f"(qv[0]), "f"(qv[1]));
                    uint32_t _fp4_pair_1;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_1) : "f"(qv[2]), "f"(qv[3]));
                    uint32_t _fp4_pair_2;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_2) : "f"(qv[4]), "f"(qv[5]));
                    uint32_t _fp4_pair_3;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_3) : "f"(qv[6]), "f"(qv[7]));
                    uint32_t _fp4_pair_4;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_4) : "f"(qv[8]), "f"(qv[9]));
                    uint32_t _fp4_pair_5;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_5) : "f"(qv[10]), "f"(qv[11]));
                    uint32_t _fp4_pair_6;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_6) : "f"(qv[12]), "f"(qv[13]));
                    uint32_t _fp4_pair_7;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_7) : "f"(qv[14]), "f"(qv[15]));
                    words[0] = _fp4_pair_0 | _fp4_pair_1 << 8 | _fp4_pair_2 << 16 | _fp4_pair_3 << 24;
                    words[1] = _fp4_pair_4 | _fp4_pair_5 << 8 | _fp4_pair_6 << 16 | _fp4_pair_7 << 24;
                    sf_word = sf_word | sc_byte;
                    unsigned int qa_0[4];
                    unsigned int qb_1[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qa_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0[(0) + 3]))
                        : "r"(q_row_addr + (2 ^ row % 8) * 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qb_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1[(0) + 3]))
                        : "r"(q_row_addr + (3 ^ row % 8) * 16));
                    float qv_2[16];
                    qv_2[0] = __uint_as_float(qa_0[0] << 16);
                    qv_2[1] = __uint_as_float(qa_0[0] & 4294901760u);
                    qv_2[8] = __uint_as_float(qb_1[0] << 16);
                    qv_2[9] = __uint_as_float(qb_1[0] & 4294901760u);
                    qv_2[2] = __uint_as_float(qa_0[1] << 16);
                    qv_2[3] = __uint_as_float(qa_0[1] & 4294901760u);
                    qv_2[10] = __uint_as_float(qb_1[1] << 16);
                    qv_2[11] = __uint_as_float(qb_1[1] & 4294901760u);
                    qv_2[4] = __uint_as_float(qa_0[2] << 16);
                    qv_2[5] = __uint_as_float(qa_0[2] & 4294901760u);
                    qv_2[12] = __uint_as_float(qb_1[2] << 16);
                    qv_2[13] = __uint_as_float(qb_1[2] & 4294901760u);
                    qv_2[6] = __uint_as_float(qa_0[3] << 16);
                    qv_2[7] = __uint_as_float(qa_0[3] & 4294901760u);
                    qv_2[14] = __uint_as_float(qb_1[3] << 16);
                    qv_2[15] = __uint_as_float(qb_1[3] & 4294901760u);
                    float m8_3[8];
                    float _fabs_16 = fabsf(qv_2[0]);
                    float _fabs_17 = fabsf(qv_2[1]);
                    float _max_15 = max_noftz(_fabs_16, _fabs_17);
                    m8_3[0] = _max_15;
                    float _fabs_18 = fabsf(qv_2[2]);
                    float _fabs_19 = fabsf(qv_2[3]);
                    float _max_16 = max_noftz(_fabs_18, _fabs_19);
                    m8_3[1] = _max_16;
                    float _fabs_20 = fabsf(qv_2[4]);
                    float _fabs_21 = fabsf(qv_2[5]);
                    float _max_17 = max_noftz(_fabs_20, _fabs_21);
                    m8_3[2] = _max_17;
                    float _fabs_22 = fabsf(qv_2[6]);
                    float _fabs_23 = fabsf(qv_2[7]);
                    float _max_18 = max_noftz(_fabs_22, _fabs_23);
                    m8_3[3] = _max_18;
                    float _fabs_24 = fabsf(qv_2[8]);
                    float _fabs_25 = fabsf(qv_2[9]);
                    float _max_19 = max_noftz(_fabs_24, _fabs_25);
                    m8_3[4] = _max_19;
                    float _fabs_26 = fabsf(qv_2[10]);
                    float _fabs_27 = fabsf(qv_2[11]);
                    float _max_20 = max_noftz(_fabs_26, _fabs_27);
                    m8_3[5] = _max_20;
                    float _fabs_28 = fabsf(qv_2[12]);
                    float _fabs_29 = fabsf(qv_2[13]);
                    float _max_21 = max_noftz(_fabs_28, _fabs_29);
                    m8_3[6] = _max_21;
                    float _fabs_30 = fabsf(qv_2[14]);
                    float _fabs_31 = fabsf(qv_2[15]);
                    float _max_22 = max_noftz(_fabs_30, _fabs_31);
                    m8_3[7] = _max_22;
                    float m4_4[4];
                    float _max_23 = max_noftz(m8_3[0], m8_3[1]);
                    m4_4[0] = _max_23;
                    float _max_24 = max_noftz(m8_3[2], m8_3[3]);
                    m4_4[1] = _max_24;
                    float _max_25 = max_noftz(m8_3[4], m8_3[5]);
                    m4_4[2] = _max_25;
                    float _max_26 = max_noftz(m8_3[6], m8_3[7]);
                    m4_4[3] = _max_26;
                    float _max_27 = max_noftz(m4_4[0], m4_4[1]);
                    float _max_28 = max_noftz(m4_4[2], m4_4[3]);
                    float _max_29 = max_noftz(_max_27, _max_28);
                    float amax_5 = _max_29;
                    float sc_6 = amax_5 * inv_six;
                    uint16_t _e4m3x2_f32_1;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1) : "f"(0.0f), "f"(sc_6));
                    uint16_t sc_pair_7 = _e4m3x2_f32_1;
                    unsigned int sc_byte_8 = (unsigned int)sc_pair_7 & 255;
                    unsigned int sc_exp_9 = sc_byte_8 >> 3 & 15;
                    unsigned int sc_man_10 = sc_byte_8 & 7;
                    float sc_norm_11 = __uint_as_float(sc_exp_9 + 120 << 23 | sc_man_10 << 20);
                    float sc_sub_12 = (float)sc_man_10 * 0.001953125f;
                    float sc_dec_13 = ((sc_exp_9 == 0) ? sc_sub_12 : sc_norm_11);
                    float _rcp_1 = __frcp_rn(sc_dec_13);
                    float inv_14 = ((sc_dec_13 > 0.0f) ? _rcp_1 : 0.0f);
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_1 = {inv_14, inv_14};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_2)[_ls], _scale2_1);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        qv_2[_ls] = qv_2[_ls] * inv_14;
                    }
                    #endif
                    uint32_t _fp4_pair_8;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_8) : "f"(qv_2[0]), "f"(qv_2[1]));
                    uint32_t _fp4_pair_9;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_9) : "f"(qv_2[2]), "f"(qv_2[3]));
                    uint32_t _fp4_pair_10;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_10) : "f"(qv_2[4]), "f"(qv_2[5]));
                    uint32_t _fp4_pair_11;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_11) : "f"(qv_2[6]), "f"(qv_2[7]));
                    uint32_t _fp4_pair_12;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_12) : "f"(qv_2[8]), "f"(qv_2[9]));
                    uint32_t _fp4_pair_13;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_13) : "f"(qv_2[10]), "f"(qv_2[11]));
                    uint32_t _fp4_pair_14;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_14) : "f"(qv_2[12]), "f"(qv_2[13]));
                    uint32_t _fp4_pair_15;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_15) : "f"(qv_2[14]), "f"(qv_2[15]));
                    words[2] = _fp4_pair_8 | _fp4_pair_9 << 8 | _fp4_pair_10 << 16 | _fp4_pair_11 << 24;
                    words[3] = _fp4_pair_12 | _fp4_pair_13 << 8 | _fp4_pair_14 << 16 | _fp4_pair_15 << 24;
                    sf_word = sf_word | sc_byte_8 << 8;
                    unsigned int qa_15[4];
                    unsigned int qb_16[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qa_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_15[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_15[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_15[(0) + 3]))
                        : "r"(q_row_addr + (4 ^ row % 8) * 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qb_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_16[(0) + 3]))
                        : "r"(q_row_addr + (5 ^ row % 8) * 16));
                    float qv_17[16];
                    qv_17[0] = __uint_as_float(qa_15[0] << 16);
                    qv_17[1] = __uint_as_float(qa_15[0] & 4294901760u);
                    qv_17[8] = __uint_as_float(qb_16[0] << 16);
                    qv_17[9] = __uint_as_float(qb_16[0] & 4294901760u);
                    qv_17[2] = __uint_as_float(qa_15[1] << 16);
                    qv_17[3] = __uint_as_float(qa_15[1] & 4294901760u);
                    qv_17[10] = __uint_as_float(qb_16[1] << 16);
                    qv_17[11] = __uint_as_float(qb_16[1] & 4294901760u);
                    qv_17[4] = __uint_as_float(qa_15[2] << 16);
                    qv_17[5] = __uint_as_float(qa_15[2] & 4294901760u);
                    qv_17[12] = __uint_as_float(qb_16[2] << 16);
                    qv_17[13] = __uint_as_float(qb_16[2] & 4294901760u);
                    qv_17[6] = __uint_as_float(qa_15[3] << 16);
                    qv_17[7] = __uint_as_float(qa_15[3] & 4294901760u);
                    qv_17[14] = __uint_as_float(qb_16[3] << 16);
                    qv_17[15] = __uint_as_float(qb_16[3] & 4294901760u);
                    float m8_18[8];
                    float _fabs_32 = fabsf(qv_17[0]);
                    float _fabs_33 = fabsf(qv_17[1]);
                    float _max_30 = max_noftz(_fabs_32, _fabs_33);
                    m8_18[0] = _max_30;
                    float _fabs_34 = fabsf(qv_17[2]);
                    float _fabs_35 = fabsf(qv_17[3]);
                    float _max_31 = max_noftz(_fabs_34, _fabs_35);
                    m8_18[1] = _max_31;
                    float _fabs_36 = fabsf(qv_17[4]);
                    float _fabs_37 = fabsf(qv_17[5]);
                    float _max_32 = max_noftz(_fabs_36, _fabs_37);
                    m8_18[2] = _max_32;
                    float _fabs_38 = fabsf(qv_17[6]);
                    float _fabs_39 = fabsf(qv_17[7]);
                    float _max_33 = max_noftz(_fabs_38, _fabs_39);
                    m8_18[3] = _max_33;
                    float _fabs_40 = fabsf(qv_17[8]);
                    float _fabs_41 = fabsf(qv_17[9]);
                    float _max_34 = max_noftz(_fabs_40, _fabs_41);
                    m8_18[4] = _max_34;
                    float _fabs_42 = fabsf(qv_17[10]);
                    float _fabs_43 = fabsf(qv_17[11]);
                    float _max_35 = max_noftz(_fabs_42, _fabs_43);
                    m8_18[5] = _max_35;
                    float _fabs_44 = fabsf(qv_17[12]);
                    float _fabs_45 = fabsf(qv_17[13]);
                    float _max_36 = max_noftz(_fabs_44, _fabs_45);
                    m8_18[6] = _max_36;
                    float _fabs_46 = fabsf(qv_17[14]);
                    float _fabs_47 = fabsf(qv_17[15]);
                    float _max_37 = max_noftz(_fabs_46, _fabs_47);
                    m8_18[7] = _max_37;
                    float m4_19[4];
                    float _max_38 = max_noftz(m8_18[0], m8_18[1]);
                    m4_19[0] = _max_38;
                    float _max_39 = max_noftz(m8_18[2], m8_18[3]);
                    m4_19[1] = _max_39;
                    float _max_40 = max_noftz(m8_18[4], m8_18[5]);
                    m4_19[2] = _max_40;
                    float _max_41 = max_noftz(m8_18[6], m8_18[7]);
                    m4_19[3] = _max_41;
                    float _max_42 = max_noftz(m4_19[0], m4_19[1]);
                    float _max_43 = max_noftz(m4_19[2], m4_19[3]);
                    float _max_44 = max_noftz(_max_42, _max_43);
                    float amax_20 = _max_44;
                    float sc_21 = amax_20 * inv_six;
                    uint16_t _e4m3x2_f32_2;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_2) : "f"(0.0f), "f"(sc_21));
                    uint16_t sc_pair_22 = _e4m3x2_f32_2;
                    unsigned int sc_byte_23 = (unsigned int)sc_pair_22 & 255;
                    unsigned int sc_exp_24 = sc_byte_23 >> 3 & 15;
                    unsigned int sc_man_25 = sc_byte_23 & 7;
                    float sc_norm_26 = __uint_as_float(sc_exp_24 + 120 << 23 | sc_man_25 << 20);
                    float sc_sub_27 = (float)sc_man_25 * 0.001953125f;
                    float sc_dec_28 = ((sc_exp_24 == 0) ? sc_sub_27 : sc_norm_26);
                    float _rcp_2 = __frcp_rn(sc_dec_28);
                    float inv_29 = ((sc_dec_28 > 0.0f) ? _rcp_2 : 0.0f);
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_2 = {inv_29, inv_29};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_17)[_ls], _scale2_2);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        qv_17[_ls] = qv_17[_ls] * inv_29;
                    }
                    #endif
                    uint32_t _fp4_pair_16;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_16) : "f"(qv_17[0]), "f"(qv_17[1]));
                    uint32_t _fp4_pair_17;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_17) : "f"(qv_17[2]), "f"(qv_17[3]));
                    uint32_t _fp4_pair_18;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_18) : "f"(qv_17[4]), "f"(qv_17[5]));
                    uint32_t _fp4_pair_19;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_19) : "f"(qv_17[6]), "f"(qv_17[7]));
                    uint32_t _fp4_pair_20;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_20) : "f"(qv_17[8]), "f"(qv_17[9]));
                    uint32_t _fp4_pair_21;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_21) : "f"(qv_17[10]), "f"(qv_17[11]));
                    uint32_t _fp4_pair_22;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_22) : "f"(qv_17[12]), "f"(qv_17[13]));
                    uint32_t _fp4_pair_23;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_23) : "f"(qv_17[14]), "f"(qv_17[15]));
                    words[4] = _fp4_pair_16 | _fp4_pair_17 << 8 | _fp4_pair_18 << 16 | _fp4_pair_19 << 24;
                    words[5] = _fp4_pair_20 | _fp4_pair_21 << 8 | _fp4_pair_22 << 16 | _fp4_pair_23 << 24;
                    sf_word = sf_word | sc_byte_23 << 16;
                    unsigned int qa_30[4];
                    unsigned int qb_31[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qa_30[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_30[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_30[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_30[(0) + 3]))
                        : "r"(q_row_addr + (6 ^ row % 8) * 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qb_31[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_31[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_31[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_31[(0) + 3]))
                        : "r"(q_row_addr + (7 ^ row % 8) * 16));
                    float qv_32[16];
                    qv_32[0] = __uint_as_float(qa_30[0] << 16);
                    qv_32[1] = __uint_as_float(qa_30[0] & 4294901760u);
                    qv_32[8] = __uint_as_float(qb_31[0] << 16);
                    qv_32[9] = __uint_as_float(qb_31[0] & 4294901760u);
                    qv_32[2] = __uint_as_float(qa_30[1] << 16);
                    qv_32[3] = __uint_as_float(qa_30[1] & 4294901760u);
                    qv_32[10] = __uint_as_float(qb_31[1] << 16);
                    qv_32[11] = __uint_as_float(qb_31[1] & 4294901760u);
                    qv_32[4] = __uint_as_float(qa_30[2] << 16);
                    qv_32[5] = __uint_as_float(qa_30[2] & 4294901760u);
                    qv_32[12] = __uint_as_float(qb_31[2] << 16);
                    qv_32[13] = __uint_as_float(qb_31[2] & 4294901760u);
                    qv_32[6] = __uint_as_float(qa_30[3] << 16);
                    qv_32[7] = __uint_as_float(qa_30[3] & 4294901760u);
                    qv_32[14] = __uint_as_float(qb_31[3] << 16);
                    qv_32[15] = __uint_as_float(qb_31[3] & 4294901760u);
                    float m8_33[8];
                    float _fabs_48 = fabsf(qv_32[0]);
                    float _fabs_49 = fabsf(qv_32[1]);
                    float _max_45 = max_noftz(_fabs_48, _fabs_49);
                    m8_33[0] = _max_45;
                    float _fabs_50 = fabsf(qv_32[2]);
                    float _fabs_51 = fabsf(qv_32[3]);
                    float _max_46 = max_noftz(_fabs_50, _fabs_51);
                    m8_33[1] = _max_46;
                    float _fabs_52 = fabsf(qv_32[4]);
                    float _fabs_53 = fabsf(qv_32[5]);
                    float _max_47 = max_noftz(_fabs_52, _fabs_53);
                    m8_33[2] = _max_47;
                    float _fabs_54 = fabsf(qv_32[6]);
                    float _fabs_55 = fabsf(qv_32[7]);
                    float _max_48 = max_noftz(_fabs_54, _fabs_55);
                    m8_33[3] = _max_48;
                    float _fabs_56 = fabsf(qv_32[8]);
                    float _fabs_57 = fabsf(qv_32[9]);
                    float _max_49 = max_noftz(_fabs_56, _fabs_57);
                    m8_33[4] = _max_49;
                    float _fabs_58 = fabsf(qv_32[10]);
                    float _fabs_59 = fabsf(qv_32[11]);
                    float _max_50 = max_noftz(_fabs_58, _fabs_59);
                    m8_33[5] = _max_50;
                    float _fabs_60 = fabsf(qv_32[12]);
                    float _fabs_61 = fabsf(qv_32[13]);
                    float _max_51 = max_noftz(_fabs_60, _fabs_61);
                    m8_33[6] = _max_51;
                    float _fabs_62 = fabsf(qv_32[14]);
                    float _fabs_63 = fabsf(qv_32[15]);
                    float _max_52 = max_noftz(_fabs_62, _fabs_63);
                    m8_33[7] = _max_52;
                    float m4_34[4];
                    float _max_53 = max_noftz(m8_33[0], m8_33[1]);
                    m4_34[0] = _max_53;
                    float _max_54 = max_noftz(m8_33[2], m8_33[3]);
                    m4_34[1] = _max_54;
                    float _max_55 = max_noftz(m8_33[4], m8_33[5]);
                    m4_34[2] = _max_55;
                    float _max_56 = max_noftz(m8_33[6], m8_33[7]);
                    m4_34[3] = _max_56;
                    float _max_57 = max_noftz(m4_34[0], m4_34[1]);
                    float _max_58 = max_noftz(m4_34[2], m4_34[3]);
                    float _max_59 = max_noftz(_max_57, _max_58);
                    float amax_35 = _max_59;
                    float sc_36 = amax_35 * inv_six;
                    uint16_t _e4m3x2_f32_3;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_3) : "f"(0.0f), "f"(sc_36));
                    uint16_t sc_pair_37 = _e4m3x2_f32_3;
                    unsigned int sc_byte_38 = (unsigned int)sc_pair_37 & 255;
                    unsigned int sc_exp_39 = sc_byte_38 >> 3 & 15;
                    unsigned int sc_man_40 = sc_byte_38 & 7;
                    float sc_norm_41 = __uint_as_float(sc_exp_39 + 120 << 23 | sc_man_40 << 20);
                    float sc_sub_42 = (float)sc_man_40 * 0.001953125f;
                    float sc_dec_43 = ((sc_exp_39 == 0) ? sc_sub_42 : sc_norm_41);
                    float _rcp_3 = __frcp_rn(sc_dec_43);
                    float inv_44 = ((sc_dec_43 > 0.0f) ? _rcp_3 : 0.0f);
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_3 = {inv_44, inv_44};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_32)[_ls], _scale2_3);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        qv_32[_ls] = qv_32[_ls] * inv_44;
                    }
                    #endif
                    uint32_t _fp4_pair_24;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_24) : "f"(qv_32[0]), "f"(qv_32[1]));
                    uint32_t _fp4_pair_25;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_25) : "f"(qv_32[2]), "f"(qv_32[3]));
                    uint32_t _fp4_pair_26;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_26) : "f"(qv_32[4]), "f"(qv_32[5]));
                    uint32_t _fp4_pair_27;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_27) : "f"(qv_32[6]), "f"(qv_32[7]));
                    uint32_t _fp4_pair_28;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_28) : "f"(qv_32[8]), "f"(qv_32[9]));
                    uint32_t _fp4_pair_29;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_29) : "f"(qv_32[10]), "f"(qv_32[11]));
                    uint32_t _fp4_pair_30;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_30) : "f"(qv_32[12]), "f"(qv_32[13]));
                    uint32_t _fp4_pair_31;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_31) : "f"(qv_32[14]), "f"(qv_32[15]));
                    words[6] = _fp4_pair_24 | _fp4_pair_25 << 8 | _fp4_pair_26 << 16 | _fp4_pair_27 << 24;
                    words[7] = _fp4_pair_28 | _fp4_pair_29 << 8 | _fp4_pair_30 << 16 | _fp4_pair_31 << 24;
                    sf_word = sf_word | sc_byte_38 << 24;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(exch_u), "r"(*reinterpret_cast<uint32_t*>(&words[0])), "r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(exch_u + 16), "r"(*reinterpret_cast<uint32_t*>(&words[4])), "r"(*reinterpret_cast<uint32_t*>(&words[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words[(4) + 3])));
                    smem_qsf32[kset_u / 4 * 2048 + row % 32 / 8 * 512 + kset_u % 4 * 128 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = sf_word;
                    smem_qsf32[kset_u / 4 * 2048 + (row ^ 64) % 32 / 8 * 512 + kset_u % 4 * 128 + (row ^ 64) % 8 * 16 + (row ^ 64) / 32 % 4 * 4 >> 2] = sf_word;
                } else {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(exch_u), "r"(*reinterpret_cast<uint32_t*>(&zero8[0])), "r"(*reinterpret_cast<uint32_t*>(&zero8[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero8[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero8[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(exch_u + 16), "r"(*reinterpret_cast<uint32_t*>(&zero8[4])), "r"(*reinterpret_cast<uint32_t*>(&zero8[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero8[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero8[(4) + 3])));
                    smem_qsf32[kset_u / 4 * 2048 + row % 32 / 8 * 512 + kset_u % 4 * 128 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = 0;
                    smem_qsf32[kset_u / 4 * 2048 + (row ^ 64) % 32 / 8 * 512 + kset_u % 4 * 128 + (row ^ 64) % 8 * 16 + (row ^ 64) / 32 % 4 * 4 >> 2] = 0;
                }
            }
            int kset_u_0 = ((0) ? q_slot : 6);
            int do_u_1 = ((0) ? 1 : ((q_slot == 0) ? 1 : 0));
            if (do_u_1 != 0) {
                int exch_u_1 = exch_lane + (q_par * 7 + kset_u_0) * 1024;
                if (q_block_live != 0) {
                    int q_row_addr_1 = smem_qstage_addr + (unsigned int)(kset_u_0 * 8192) + (unsigned int)(q_head * 128);
                    unsigned int words_1[8];
                    unsigned int sf_word_1 = 0;
                    unsigned int qa_1[4];
                    unsigned int qb_2[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qa_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_1[(0) + 3]))
                        : "r"(q_row_addr_1 + (0 ^ row % 8) * 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qb_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_2[(0) + 3]))
                        : "r"(q_row_addr_1 + (1 ^ row % 8) * 16));
                    float qv_1[16];
                    qv_1[0] = __uint_as_float(qa_1[0] << 16);
                    qv_1[1] = __uint_as_float(qa_1[0] & 4294901760u);
                    qv_1[8] = __uint_as_float(qb_2[0] << 16);
                    qv_1[9] = __uint_as_float(qb_2[0] & 4294901760u);
                    qv_1[2] = __uint_as_float(qa_1[1] << 16);
                    qv_1[3] = __uint_as_float(qa_1[1] & 4294901760u);
                    qv_1[10] = __uint_as_float(qb_2[1] << 16);
                    qv_1[11] = __uint_as_float(qb_2[1] & 4294901760u);
                    qv_1[4] = __uint_as_float(qa_1[2] << 16);
                    qv_1[5] = __uint_as_float(qa_1[2] & 4294901760u);
                    qv_1[12] = __uint_as_float(qb_2[2] << 16);
                    qv_1[13] = __uint_as_float(qb_2[2] & 4294901760u);
                    qv_1[6] = __uint_as_float(qa_1[3] << 16);
                    qv_1[7] = __uint_as_float(qa_1[3] & 4294901760u);
                    qv_1[14] = __uint_as_float(qb_2[3] << 16);
                    qv_1[15] = __uint_as_float(qb_2[3] & 4294901760u);
                    float m8_1[8];
                    float _fabs_64 = fabsf(qv_1[0]);
                    float _fabs_65 = fabsf(qv_1[1]);
                    float _max_60 = max_noftz(_fabs_64, _fabs_65);
                    m8_1[0] = _max_60;
                    float _fabs_66 = fabsf(qv_1[2]);
                    float _fabs_67 = fabsf(qv_1[3]);
                    float _max_61 = max_noftz(_fabs_66, _fabs_67);
                    m8_1[1] = _max_61;
                    float _fabs_68 = fabsf(qv_1[4]);
                    float _fabs_69 = fabsf(qv_1[5]);
                    float _max_62 = max_noftz(_fabs_68, _fabs_69);
                    m8_1[2] = _max_62;
                    float _fabs_70 = fabsf(qv_1[6]);
                    float _fabs_71 = fabsf(qv_1[7]);
                    float _max_63 = max_noftz(_fabs_70, _fabs_71);
                    m8_1[3] = _max_63;
                    float _fabs_72 = fabsf(qv_1[8]);
                    float _fabs_73 = fabsf(qv_1[9]);
                    float _max_64 = max_noftz(_fabs_72, _fabs_73);
                    m8_1[4] = _max_64;
                    float _fabs_74 = fabsf(qv_1[10]);
                    float _fabs_75 = fabsf(qv_1[11]);
                    float _max_65 = max_noftz(_fabs_74, _fabs_75);
                    m8_1[5] = _max_65;
                    float _fabs_76 = fabsf(qv_1[12]);
                    float _fabs_77 = fabsf(qv_1[13]);
                    float _max_66 = max_noftz(_fabs_76, _fabs_77);
                    m8_1[6] = _max_66;
                    float _fabs_78 = fabsf(qv_1[14]);
                    float _fabs_79 = fabsf(qv_1[15]);
                    float _max_67 = max_noftz(_fabs_78, _fabs_79);
                    m8_1[7] = _max_67;
                    float m4_1[4];
                    float _max_68 = max_noftz(m8_1[0], m8_1[1]);
                    m4_1[0] = _max_68;
                    float _max_69 = max_noftz(m8_1[2], m8_1[3]);
                    m4_1[1] = _max_69;
                    float _max_70 = max_noftz(m8_1[4], m8_1[5]);
                    m4_1[2] = _max_70;
                    float _max_71 = max_noftz(m8_1[6], m8_1[7]);
                    m4_1[3] = _max_71;
                    float _max_72 = max_noftz(m4_1[0], m4_1[1]);
                    float _max_73 = max_noftz(m4_1[2], m4_1[3]);
                    float _max_74 = max_noftz(_max_72, _max_73);
                    float amax_1 = _max_74;
                    float sc_1 = amax_1 * inv_six;
                    uint16_t _e4m3x2_f32_4;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_4) : "f"(0.0f), "f"(sc_1));
                    uint16_t sc_pair_1 = _e4m3x2_f32_4;
                    unsigned int sc_byte_1 = (unsigned int)sc_pair_1 & 255;
                    unsigned int sc_exp_1 = sc_byte_1 >> 3 & 15;
                    unsigned int sc_man_1 = sc_byte_1 & 7;
                    float sc_norm_1 = __uint_as_float(sc_exp_1 + 120 << 23 | sc_man_1 << 20);
                    float sc_sub_1 = (float)sc_man_1 * 0.001953125f;
                    float sc_dec_1 = ((sc_exp_1 == 0) ? sc_sub_1 : sc_norm_1);
                    float _rcp_4 = __frcp_rn(sc_dec_1);
                    float inv_1 = ((sc_dec_1 > 0.0f) ? _rcp_4 : 0.0f);
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_4 = {inv_1, inv_1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_1)[_ls], _scale2_4);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        qv_1[_ls] = qv_1[_ls] * inv_1;
                    }
                    #endif
                    uint32_t _fp4_pair_32;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_32) : "f"(qv_1[0]), "f"(qv_1[1]));
                    uint32_t _fp4_pair_33;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_33) : "f"(qv_1[2]), "f"(qv_1[3]));
                    uint32_t _fp4_pair_34;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_34) : "f"(qv_1[4]), "f"(qv_1[5]));
                    uint32_t _fp4_pair_35;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_35) : "f"(qv_1[6]), "f"(qv_1[7]));
                    uint32_t _fp4_pair_36;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_36) : "f"(qv_1[8]), "f"(qv_1[9]));
                    uint32_t _fp4_pair_37;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_37) : "f"(qv_1[10]), "f"(qv_1[11]));
                    uint32_t _fp4_pair_38;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_38) : "f"(qv_1[12]), "f"(qv_1[13]));
                    uint32_t _fp4_pair_39;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_39) : "f"(qv_1[14]), "f"(qv_1[15]));
                    words_1[0] = _fp4_pair_32 | _fp4_pair_33 << 8 | _fp4_pair_34 << 16 | _fp4_pair_35 << 24;
                    words_1[1] = _fp4_pair_36 | _fp4_pair_37 << 8 | _fp4_pair_38 << 16 | _fp4_pair_39 << 24;
                    sf_word_1 = sf_word_1 | sc_byte_1;
                    unsigned int qa_0_1[4];
                    unsigned int qb_1_1[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qa_0_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0_1[(0) + 3]))
                        : "r"(q_row_addr_1 + (2 ^ row % 8) * 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qb_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1_1[(0) + 3]))
                        : "r"(q_row_addr_1 + (3 ^ row % 8) * 16));
                    float qv_2_1[16];
                    qv_2_1[0] = __uint_as_float(qa_0_1[0] << 16);
                    qv_2_1[1] = __uint_as_float(qa_0_1[0] & 4294901760u);
                    qv_2_1[8] = __uint_as_float(qb_1_1[0] << 16);
                    qv_2_1[9] = __uint_as_float(qb_1_1[0] & 4294901760u);
                    qv_2_1[2] = __uint_as_float(qa_0_1[1] << 16);
                    qv_2_1[3] = __uint_as_float(qa_0_1[1] & 4294901760u);
                    qv_2_1[10] = __uint_as_float(qb_1_1[1] << 16);
                    qv_2_1[11] = __uint_as_float(qb_1_1[1] & 4294901760u);
                    qv_2_1[4] = __uint_as_float(qa_0_1[2] << 16);
                    qv_2_1[5] = __uint_as_float(qa_0_1[2] & 4294901760u);
                    qv_2_1[12] = __uint_as_float(qb_1_1[2] << 16);
                    qv_2_1[13] = __uint_as_float(qb_1_1[2] & 4294901760u);
                    qv_2_1[6] = __uint_as_float(qa_0_1[3] << 16);
                    qv_2_1[7] = __uint_as_float(qa_0_1[3] & 4294901760u);
                    qv_2_1[14] = __uint_as_float(qb_1_1[3] << 16);
                    qv_2_1[15] = __uint_as_float(qb_1_1[3] & 4294901760u);
                    float m8_3_1[8];
                    float _fabs_80 = fabsf(qv_2_1[0]);
                    float _fabs_81 = fabsf(qv_2_1[1]);
                    float _max_75 = max_noftz(_fabs_80, _fabs_81);
                    m8_3_1[0] = _max_75;
                    float _fabs_82 = fabsf(qv_2_1[2]);
                    float _fabs_83 = fabsf(qv_2_1[3]);
                    float _max_76 = max_noftz(_fabs_82, _fabs_83);
                    m8_3_1[1] = _max_76;
                    float _fabs_84 = fabsf(qv_2_1[4]);
                    float _fabs_85 = fabsf(qv_2_1[5]);
                    float _max_77 = max_noftz(_fabs_84, _fabs_85);
                    m8_3_1[2] = _max_77;
                    float _fabs_86 = fabsf(qv_2_1[6]);
                    float _fabs_87 = fabsf(qv_2_1[7]);
                    float _max_78 = max_noftz(_fabs_86, _fabs_87);
                    m8_3_1[3] = _max_78;
                    float _fabs_88 = fabsf(qv_2_1[8]);
                    float _fabs_89 = fabsf(qv_2_1[9]);
                    float _max_79 = max_noftz(_fabs_88, _fabs_89);
                    m8_3_1[4] = _max_79;
                    float _fabs_90 = fabsf(qv_2_1[10]);
                    float _fabs_91 = fabsf(qv_2_1[11]);
                    float _max_80 = max_noftz(_fabs_90, _fabs_91);
                    m8_3_1[5] = _max_80;
                    float _fabs_92 = fabsf(qv_2_1[12]);
                    float _fabs_93 = fabsf(qv_2_1[13]);
                    float _max_81 = max_noftz(_fabs_92, _fabs_93);
                    m8_3_1[6] = _max_81;
                    float _fabs_94 = fabsf(qv_2_1[14]);
                    float _fabs_95 = fabsf(qv_2_1[15]);
                    float _max_82 = max_noftz(_fabs_94, _fabs_95);
                    m8_3_1[7] = _max_82;
                    float m4_4_1[4];
                    float _max_83 = max_noftz(m8_3_1[0], m8_3_1[1]);
                    m4_4_1[0] = _max_83;
                    float _max_84 = max_noftz(m8_3_1[2], m8_3_1[3]);
                    m4_4_1[1] = _max_84;
                    float _max_85 = max_noftz(m8_3_1[4], m8_3_1[5]);
                    m4_4_1[2] = _max_85;
                    float _max_86 = max_noftz(m8_3_1[6], m8_3_1[7]);
                    m4_4_1[3] = _max_86;
                    float _max_87 = max_noftz(m4_4_1[0], m4_4_1[1]);
                    float _max_88 = max_noftz(m4_4_1[2], m4_4_1[3]);
                    float _max_89 = max_noftz(_max_87, _max_88);
                    float amax_5_1 = _max_89;
                    float sc_6_1 = amax_5_1 * inv_six;
                    uint16_t _e4m3x2_f32_5;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_5) : "f"(0.0f), "f"(sc_6_1));
                    uint16_t sc_pair_7_1 = _e4m3x2_f32_5;
                    unsigned int sc_byte_8_1 = (unsigned int)sc_pair_7_1 & 255;
                    unsigned int sc_exp_9_1 = sc_byte_8_1 >> 3 & 15;
                    unsigned int sc_man_10_1 = sc_byte_8_1 & 7;
                    float sc_norm_11_1 = __uint_as_float(sc_exp_9_1 + 120 << 23 | sc_man_10_1 << 20);
                    float sc_sub_12_1 = (float)sc_man_10_1 * 0.001953125f;
                    float sc_dec_13_1 = ((sc_exp_9_1 == 0) ? sc_sub_12_1 : sc_norm_11_1);
                    float _rcp_5 = __frcp_rn(sc_dec_13_1);
                    float inv_14_1 = ((sc_dec_13_1 > 0.0f) ? _rcp_5 : 0.0f);
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_5 = {inv_14_1, inv_14_1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_2_1)[_ls], _scale2_5);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        qv_2_1[_ls] = qv_2_1[_ls] * inv_14_1;
                    }
                    #endif
                    uint32_t _fp4_pair_40;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_40) : "f"(qv_2_1[0]), "f"(qv_2_1[1]));
                    uint32_t _fp4_pair_41;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_41) : "f"(qv_2_1[2]), "f"(qv_2_1[3]));
                    uint32_t _fp4_pair_42;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_42) : "f"(qv_2_1[4]), "f"(qv_2_1[5]));
                    uint32_t _fp4_pair_43;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_43) : "f"(qv_2_1[6]), "f"(qv_2_1[7]));
                    uint32_t _fp4_pair_44;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_44) : "f"(qv_2_1[8]), "f"(qv_2_1[9]));
                    uint32_t _fp4_pair_45;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_45) : "f"(qv_2_1[10]), "f"(qv_2_1[11]));
                    uint32_t _fp4_pair_46;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_46) : "f"(qv_2_1[12]), "f"(qv_2_1[13]));
                    uint32_t _fp4_pair_47;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_47) : "f"(qv_2_1[14]), "f"(qv_2_1[15]));
                    words_1[2] = _fp4_pair_40 | _fp4_pair_41 << 8 | _fp4_pair_42 << 16 | _fp4_pair_43 << 24;
                    words_1[3] = _fp4_pair_44 | _fp4_pair_45 << 8 | _fp4_pair_46 << 16 | _fp4_pair_47 << 24;
                    sf_word_1 = sf_word_1 | sc_byte_8_1 << 8;
                    unsigned int qa_15_1[4];
                    unsigned int qb_16_1[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qa_15_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_15_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_15_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_15_1[(0) + 3]))
                        : "r"(q_row_addr_1 + (4 ^ row % 8) * 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qb_16_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_16_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_16_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_16_1[(0) + 3]))
                        : "r"(q_row_addr_1 + (5 ^ row % 8) * 16));
                    float qv_17_1[16];
                    qv_17_1[0] = __uint_as_float(qa_15_1[0] << 16);
                    qv_17_1[1] = __uint_as_float(qa_15_1[0] & 4294901760u);
                    qv_17_1[8] = __uint_as_float(qb_16_1[0] << 16);
                    qv_17_1[9] = __uint_as_float(qb_16_1[0] & 4294901760u);
                    qv_17_1[2] = __uint_as_float(qa_15_1[1] << 16);
                    qv_17_1[3] = __uint_as_float(qa_15_1[1] & 4294901760u);
                    qv_17_1[10] = __uint_as_float(qb_16_1[1] << 16);
                    qv_17_1[11] = __uint_as_float(qb_16_1[1] & 4294901760u);
                    qv_17_1[4] = __uint_as_float(qa_15_1[2] << 16);
                    qv_17_1[5] = __uint_as_float(qa_15_1[2] & 4294901760u);
                    qv_17_1[12] = __uint_as_float(qb_16_1[2] << 16);
                    qv_17_1[13] = __uint_as_float(qb_16_1[2] & 4294901760u);
                    qv_17_1[6] = __uint_as_float(qa_15_1[3] << 16);
                    qv_17_1[7] = __uint_as_float(qa_15_1[3] & 4294901760u);
                    qv_17_1[14] = __uint_as_float(qb_16_1[3] << 16);
                    qv_17_1[15] = __uint_as_float(qb_16_1[3] & 4294901760u);
                    float m8_18_1[8];
                    float _fabs_96 = fabsf(qv_17_1[0]);
                    float _fabs_97 = fabsf(qv_17_1[1]);
                    float _max_90 = max_noftz(_fabs_96, _fabs_97);
                    m8_18_1[0] = _max_90;
                    float _fabs_98 = fabsf(qv_17_1[2]);
                    float _fabs_99 = fabsf(qv_17_1[3]);
                    float _max_91 = max_noftz(_fabs_98, _fabs_99);
                    m8_18_1[1] = _max_91;
                    float _fabs_100 = fabsf(qv_17_1[4]);
                    float _fabs_101 = fabsf(qv_17_1[5]);
                    float _max_92 = max_noftz(_fabs_100, _fabs_101);
                    m8_18_1[2] = _max_92;
                    float _fabs_102 = fabsf(qv_17_1[6]);
                    float _fabs_103 = fabsf(qv_17_1[7]);
                    float _max_93 = max_noftz(_fabs_102, _fabs_103);
                    m8_18_1[3] = _max_93;
                    float _fabs_104 = fabsf(qv_17_1[8]);
                    float _fabs_105 = fabsf(qv_17_1[9]);
                    float _max_94 = max_noftz(_fabs_104, _fabs_105);
                    m8_18_1[4] = _max_94;
                    float _fabs_106 = fabsf(qv_17_1[10]);
                    float _fabs_107 = fabsf(qv_17_1[11]);
                    float _max_95 = max_noftz(_fabs_106, _fabs_107);
                    m8_18_1[5] = _max_95;
                    float _fabs_108 = fabsf(qv_17_1[12]);
                    float _fabs_109 = fabsf(qv_17_1[13]);
                    float _max_96 = max_noftz(_fabs_108, _fabs_109);
                    m8_18_1[6] = _max_96;
                    float _fabs_110 = fabsf(qv_17_1[14]);
                    float _fabs_111 = fabsf(qv_17_1[15]);
                    float _max_97 = max_noftz(_fabs_110, _fabs_111);
                    m8_18_1[7] = _max_97;
                    float m4_19_1[4];
                    float _max_98 = max_noftz(m8_18_1[0], m8_18_1[1]);
                    m4_19_1[0] = _max_98;
                    float _max_99 = max_noftz(m8_18_1[2], m8_18_1[3]);
                    m4_19_1[1] = _max_99;
                    float _max_100 = max_noftz(m8_18_1[4], m8_18_1[5]);
                    m4_19_1[2] = _max_100;
                    float _max_101 = max_noftz(m8_18_1[6], m8_18_1[7]);
                    m4_19_1[3] = _max_101;
                    float _max_102 = max_noftz(m4_19_1[0], m4_19_1[1]);
                    float _max_103 = max_noftz(m4_19_1[2], m4_19_1[3]);
                    float _max_104 = max_noftz(_max_102, _max_103);
                    float amax_20_1 = _max_104;
                    float sc_21_1 = amax_20_1 * inv_six;
                    uint16_t _e4m3x2_f32_6;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_6) : "f"(0.0f), "f"(sc_21_1));
                    uint16_t sc_pair_22_1 = _e4m3x2_f32_6;
                    unsigned int sc_byte_23_1 = (unsigned int)sc_pair_22_1 & 255;
                    unsigned int sc_exp_24_1 = sc_byte_23_1 >> 3 & 15;
                    unsigned int sc_man_25_1 = sc_byte_23_1 & 7;
                    float sc_norm_26_1 = __uint_as_float(sc_exp_24_1 + 120 << 23 | sc_man_25_1 << 20);
                    float sc_sub_27_1 = (float)sc_man_25_1 * 0.001953125f;
                    float sc_dec_28_1 = ((sc_exp_24_1 == 0) ? sc_sub_27_1 : sc_norm_26_1);
                    float _rcp_6 = __frcp_rn(sc_dec_28_1);
                    float inv_29_1 = ((sc_dec_28_1 > 0.0f) ? _rcp_6 : 0.0f);
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_6 = {inv_29_1, inv_29_1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_17_1)[_ls], _scale2_6);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        qv_17_1[_ls] = qv_17_1[_ls] * inv_29_1;
                    }
                    #endif
                    uint32_t _fp4_pair_48;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_48) : "f"(qv_17_1[0]), "f"(qv_17_1[1]));
                    uint32_t _fp4_pair_49;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_49) : "f"(qv_17_1[2]), "f"(qv_17_1[3]));
                    uint32_t _fp4_pair_50;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_50) : "f"(qv_17_1[4]), "f"(qv_17_1[5]));
                    uint32_t _fp4_pair_51;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_51) : "f"(qv_17_1[6]), "f"(qv_17_1[7]));
                    uint32_t _fp4_pair_52;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_52) : "f"(qv_17_1[8]), "f"(qv_17_1[9]));
                    uint32_t _fp4_pair_53;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_53) : "f"(qv_17_1[10]), "f"(qv_17_1[11]));
                    uint32_t _fp4_pair_54;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_54) : "f"(qv_17_1[12]), "f"(qv_17_1[13]));
                    uint32_t _fp4_pair_55;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_55) : "f"(qv_17_1[14]), "f"(qv_17_1[15]));
                    words_1[4] = _fp4_pair_48 | _fp4_pair_49 << 8 | _fp4_pair_50 << 16 | _fp4_pair_51 << 24;
                    words_1[5] = _fp4_pair_52 | _fp4_pair_53 << 8 | _fp4_pair_54 << 16 | _fp4_pair_55 << 24;
                    sf_word_1 = sf_word_1 | sc_byte_23_1 << 16;
                    unsigned int qa_30_1[4];
                    unsigned int qb_31_1[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qa_30_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_30_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_30_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_30_1[(0) + 3]))
                        : "r"(q_row_addr_1 + (6 ^ row % 8) * 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qb_31_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_31_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_31_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_31_1[(0) + 3]))
                        : "r"(q_row_addr_1 + (7 ^ row % 8) * 16));
                    float qv_32_1[16];
                    qv_32_1[0] = __uint_as_float(qa_30_1[0] << 16);
                    qv_32_1[1] = __uint_as_float(qa_30_1[0] & 4294901760u);
                    qv_32_1[8] = __uint_as_float(qb_31_1[0] << 16);
                    qv_32_1[9] = __uint_as_float(qb_31_1[0] & 4294901760u);
                    qv_32_1[2] = __uint_as_float(qa_30_1[1] << 16);
                    qv_32_1[3] = __uint_as_float(qa_30_1[1] & 4294901760u);
                    qv_32_1[10] = __uint_as_float(qb_31_1[1] << 16);
                    qv_32_1[11] = __uint_as_float(qb_31_1[1] & 4294901760u);
                    qv_32_1[4] = __uint_as_float(qa_30_1[2] << 16);
                    qv_32_1[5] = __uint_as_float(qa_30_1[2] & 4294901760u);
                    qv_32_1[12] = __uint_as_float(qb_31_1[2] << 16);
                    qv_32_1[13] = __uint_as_float(qb_31_1[2] & 4294901760u);
                    qv_32_1[6] = __uint_as_float(qa_30_1[3] << 16);
                    qv_32_1[7] = __uint_as_float(qa_30_1[3] & 4294901760u);
                    qv_32_1[14] = __uint_as_float(qb_31_1[3] << 16);
                    qv_32_1[15] = __uint_as_float(qb_31_1[3] & 4294901760u);
                    float m8_33_1[8];
                    float _fabs_112 = fabsf(qv_32_1[0]);
                    float _fabs_113 = fabsf(qv_32_1[1]);
                    float _max_105 = max_noftz(_fabs_112, _fabs_113);
                    m8_33_1[0] = _max_105;
                    float _fabs_114 = fabsf(qv_32_1[2]);
                    float _fabs_115 = fabsf(qv_32_1[3]);
                    float _max_106 = max_noftz(_fabs_114, _fabs_115);
                    m8_33_1[1] = _max_106;
                    float _fabs_116 = fabsf(qv_32_1[4]);
                    float _fabs_117 = fabsf(qv_32_1[5]);
                    float _max_107 = max_noftz(_fabs_116, _fabs_117);
                    m8_33_1[2] = _max_107;
                    float _fabs_118 = fabsf(qv_32_1[6]);
                    float _fabs_119 = fabsf(qv_32_1[7]);
                    float _max_108 = max_noftz(_fabs_118, _fabs_119);
                    m8_33_1[3] = _max_108;
                    float _fabs_120 = fabsf(qv_32_1[8]);
                    float _fabs_121 = fabsf(qv_32_1[9]);
                    float _max_109 = max_noftz(_fabs_120, _fabs_121);
                    m8_33_1[4] = _max_109;
                    float _fabs_122 = fabsf(qv_32_1[10]);
                    float _fabs_123 = fabsf(qv_32_1[11]);
                    float _max_110 = max_noftz(_fabs_122, _fabs_123);
                    m8_33_1[5] = _max_110;
                    float _fabs_124 = fabsf(qv_32_1[12]);
                    float _fabs_125 = fabsf(qv_32_1[13]);
                    float _max_111 = max_noftz(_fabs_124, _fabs_125);
                    m8_33_1[6] = _max_111;
                    float _fabs_126 = fabsf(qv_32_1[14]);
                    float _fabs_127 = fabsf(qv_32_1[15]);
                    float _max_112 = max_noftz(_fabs_126, _fabs_127);
                    m8_33_1[7] = _max_112;
                    float m4_34_1[4];
                    float _max_113 = max_noftz(m8_33_1[0], m8_33_1[1]);
                    m4_34_1[0] = _max_113;
                    float _max_114 = max_noftz(m8_33_1[2], m8_33_1[3]);
                    m4_34_1[1] = _max_114;
                    float _max_115 = max_noftz(m8_33_1[4], m8_33_1[5]);
                    m4_34_1[2] = _max_115;
                    float _max_116 = max_noftz(m8_33_1[6], m8_33_1[7]);
                    m4_34_1[3] = _max_116;
                    float _max_117 = max_noftz(m4_34_1[0], m4_34_1[1]);
                    float _max_118 = max_noftz(m4_34_1[2], m4_34_1[3]);
                    float _max_119 = max_noftz(_max_117, _max_118);
                    float amax_35_1 = _max_119;
                    float sc_36_1 = amax_35_1 * inv_six;
                    uint16_t _e4m3x2_f32_7;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_7) : "f"(0.0f), "f"(sc_36_1));
                    uint16_t sc_pair_37_1 = _e4m3x2_f32_7;
                    unsigned int sc_byte_38_1 = (unsigned int)sc_pair_37_1 & 255;
                    unsigned int sc_exp_39_1 = sc_byte_38_1 >> 3 & 15;
                    unsigned int sc_man_40_1 = sc_byte_38_1 & 7;
                    float sc_norm_41_1 = __uint_as_float(sc_exp_39_1 + 120 << 23 | sc_man_40_1 << 20);
                    float sc_sub_42_1 = (float)sc_man_40_1 * 0.001953125f;
                    float sc_dec_43_1 = ((sc_exp_39_1 == 0) ? sc_sub_42_1 : sc_norm_41_1);
                    float _rcp_7 = __frcp_rn(sc_dec_43_1);
                    float inv_44_1 = ((sc_dec_43_1 > 0.0f) ? _rcp_7 : 0.0f);
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_7 = {inv_44_1, inv_44_1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_32_1)[_ls], _scale2_7);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        qv_32_1[_ls] = qv_32_1[_ls] * inv_44_1;
                    }
                    #endif
                    uint32_t _fp4_pair_56;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_56) : "f"(qv_32_1[0]), "f"(qv_32_1[1]));
                    uint32_t _fp4_pair_57;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_57) : "f"(qv_32_1[2]), "f"(qv_32_1[3]));
                    uint32_t _fp4_pair_58;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_58) : "f"(qv_32_1[4]), "f"(qv_32_1[5]));
                    uint32_t _fp4_pair_59;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_59) : "f"(qv_32_1[6]), "f"(qv_32_1[7]));
                    uint32_t _fp4_pair_60;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_60) : "f"(qv_32_1[8]), "f"(qv_32_1[9]));
                    uint32_t _fp4_pair_61;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_61) : "f"(qv_32_1[10]), "f"(qv_32_1[11]));
                    uint32_t _fp4_pair_62;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_62) : "f"(qv_32_1[12]), "f"(qv_32_1[13]));
                    uint32_t _fp4_pair_63;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_63) : "f"(qv_32_1[14]), "f"(qv_32_1[15]));
                    words_1[6] = _fp4_pair_56 | _fp4_pair_57 << 8 | _fp4_pair_58 << 16 | _fp4_pair_59 << 24;
                    words_1[7] = _fp4_pair_60 | _fp4_pair_61 << 8 | _fp4_pair_62 << 16 | _fp4_pair_63 << 24;
                    sf_word_1 = sf_word_1 | sc_byte_38_1 << 24;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(exch_u_1), "r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(exch_u_1 + 16), "r"(*reinterpret_cast<uint32_t*>(&words_1[4])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(4) + 3])));
                    smem_qsf32[kset_u_0 / 4 * 2048 + row % 32 / 8 * 512 + kset_u_0 % 4 * 128 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = sf_word_1;
                    smem_qsf32[kset_u_0 / 4 * 2048 + (row ^ 64) % 32 / 8 * 512 + kset_u_0 % 4 * 128 + (row ^ 64) % 8 * 16 + (row ^ 64) / 32 % 4 * 4 >> 2] = sf_word_1;
                } else {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(exch_u_1), "r"(*reinterpret_cast<uint32_t*>(&zero8[0])), "r"(*reinterpret_cast<uint32_t*>(&zero8[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero8[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero8[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(exch_u_1 + 16), "r"(*reinterpret_cast<uint32_t*>(&zero8[4])), "r"(*reinterpret_cast<uint32_t*>(&zero8[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero8[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero8[(4) + 3])));
                    smem_qsf32[kset_u_0 / 4 * 2048 + row % 32 / 8 * 512 + kset_u_0 % 4 * 128 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = 0;
                    smem_qsf32[kset_u_0 / 4 * 2048 + (row ^ 64) % 32 / 8 * 512 + kset_u_0 % 4 * 128 + (row ^ 64) % 8 * 16 + (row ^ 64) / 32 % 4 * 4 >> 2] = 0;
                }
            }
            asm volatile("barrier.sync 10, 384;" ::: "memory");
            unsigned int qw[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw[0])), "=r"(*reinterpret_cast<uint32_t*>(&qw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw[(0) + 3]))
                : "r"(exch_lane + q_par * 7 * 1024));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw[4])), "=r"(*reinterpret_cast<uint32_t*>(&qw[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw[(4) + 3]))
                : "r"(exch_lane + q_par * 7 * 1024 + 16));
            tmem_st_x8_u32(q_taddr, (const uint32_t*)qw);
            unsigned int qw_2[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&qw_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_2[(0) + 3]))
                : "r"(exch_lane + (q_par * 7 + 1) * 1024));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&qw_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_2[(4) + 3]))
                : "r"(exch_lane + (q_par * 7 + 1) * 1024 + 16));
            tmem_st_x8_u32(q_taddr + 8, (const uint32_t*)qw_2);
            unsigned int qw_3[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&qw_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_3[(0) + 3]))
                : "r"(exch_lane + (q_par * 7 + 2) * 1024));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&qw_3[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_3[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_3[(4) + 3]))
                : "r"(exch_lane + (q_par * 7 + 2) * 1024 + 16));
            tmem_st_x8_u32(q_taddr + 16, (const uint32_t*)qw_3);
            unsigned int qw_4[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&qw_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_4[(0) + 3]))
                : "r"(exch_lane + (q_par * 7 + 3) * 1024));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&qw_4[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_4[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_4[(4) + 3]))
                : "r"(exch_lane + (q_par * 7 + 3) * 1024 + 16));
            tmem_st_x8_u32(q_taddr + 24, (const uint32_t*)qw_4);
            unsigned int qw_5[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&qw_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_5[(0) + 3]))
                : "r"(exch_lane + (q_par * 7 + 4) * 1024));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&qw_5[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_5[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_5[(4) + 3]))
                : "r"(exch_lane + (q_par * 7 + 4) * 1024 + 16));
            tmem_st_x8_u32(q_taddr + 32, (const uint32_t*)qw_5);
            unsigned int qw_6[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&qw_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_6[(0) + 3]))
                : "r"(exch_lane + (q_par * 7 + 5) * 1024));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&qw_6[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_6[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_6[(4) + 3]))
                : "r"(exch_lane + (q_par * 7 + 5) * 1024 + 16));
            tmem_st_x8_u32(q_taddr + 40, (const uint32_t*)qw_6);
            unsigned int qw_7[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&qw_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_7[(0) + 3]))
                : "r"(exch_lane + (q_par * 7 + 6) * 1024));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&qw_7[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_7[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_7[(4) + 3]))
                : "r"(exch_lane + (q_par * 7 + 6) * 1024 + 16));
            tmem_st_x8_u32(q_taddr + 48, (const uint32_t*)qw_7);
            tmem_st_x8_u32(q_taddr + 56, (const uint32_t*)zero8);
            smem_qsf32[2048 + row % 32 / 8 * 512 + 384 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = 0;
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(q_ready_addr);
            asm volatile("barrier.sync 9, 384;" ::: "memory");
            float softmax_scale_log2 = bmm1_scale[0] * 1.4426950408889634f;
            float output_scale = bmm2_scale[0];
            float sm[3];
            sm[0] = -CAKE_INF;
            sm[1] = 0.0f;
            sm[2] = 0.0f;
            float sink_lane = -CAKE_INF;
            if (has_sinks != 0 && split_idx == 0 && head_base + head < num_heads) {
                sink_lane = sinks[head_base + head] * 1.4426950408889634f;
            }
            for (int it = 0; it < tiles_per_split; it++) {
                int buf = it & 1;
                int par = it >> 1 & 1;
                int kbase = smem_kf4_0_addr + (unsigned int)(buf * 53248);
                int kz_off = buf * 53248;
                mbarrier_wait_hint(tok_full_addr + (buf) * 8, par, 10000000);
                int tok_off = buf * 256;
                int pbase = smem_p_0_addr + (unsigned int)(buf * 8192);
                int raw_index = smem_tok32v[tok_off + row];
                unsigned int mask_word = (unsigned int)smem_tok32v[tok_off + ((half == 0) ? 128 : 130)];
                int valid = 1;
                if (raw_index < 0) {
                    valid = 0;
                }
                if (valid != 0) {
                    {
                        unsigned int sfw[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&sfw[0])), "=r"(*reinterpret_cast<uint32_t*>(&sfw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sfw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sfw[(0) + 3]))
                            : "r"(kbase + 49152 + row * 32));
                        smem_ksf32[row % 32 / 8 * 512 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = sfw[0];
                        smem_ksf32[row % 32 / 8 * 512 + 128 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = sfw[1];
                        smem_ksf32[row % 32 / 8 * 512 + 256 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = sfw[2];
                        smem_ksf32[row % 32 / 8 * 512 + 384 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = sfw[3];
                    }
                } else if (1) {
                    smem_ksf32[row % 32 / 8 * 512 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = 0;
                    smem_ksf32[row % 32 / 8 * 512 + 128 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = 0;
                    smem_ksf32[row % 32 / 8 * 512 + 256 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = 0;
                    smem_ksf32[row % 32 / 8 * 512 + 384 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = 0;
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(kv_full_addr);
                if (it > 0) {
                    mbarrier_wait_hint(o_full_addr, it - 1 & 1, 10000000);
                    mbarrier_wait_hint(o_full_addr + 8, it - 1 & 1, 10000000);
                    mbarrier_wait_hint(o_full_addr + 16, it - 1 & 1, 10000000);
                    mbarrier_wait_hint(o_full_addr + 24, it - 1 & 1, 10000000);
                }
                if (valid != 0) {
                    unsigned int kraw4[4];
                    unsigned int sfw32 = 0;
                    int vblock = 32 * o_chunk;
                    unsigned int v8[4];
                    {
                        {
                            int vchunk = 16 * o_chunk;
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&kraw4[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4[(0) + 3]))
                                : "r"(kbase + vchunk / 8 * 16384 + (row * 128 + (vchunk % 8 * 16 ^ row % 8 * 16))));
                        }
                        {
                            sfw32 = smem_kz32[(kz_off + 49152 + row * 32 >> 2) + 8 * o_chunk];
                        }
                        unsigned int scale = sfw32 & 255;
                        {
                            v8[0] = cake_dsv4_qmul4<5>(kraw4[0], scale);
                        }
                        {
                            v8[1] = cake_dsv4_qmul4<6>(kraw4[0], scale);
                        }
                        {
                            v8[2] = cake_dsv4_qmul4<5>(kraw4[1], scale);
                        }
                        {
                            v8[3] = cake_dsv4_qmul4<6>(kraw4[1], scale);
                        }
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(smem_v_addr + (unsigned int)(row * 128 + (0 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8[0])), "r"(*reinterpret_cast<uint32_t*>(&v8[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8[(0) + 3])));
                    int vblock_0 = 32 * o_chunk + 1;
                    unsigned int v8_1[4];
                    {
                        unsigned int scale_1 = sfw32 >> 8 & 255;
                        {
                            v8_1[0] = cake_dsv4_qmul4<5>(kraw4[2], scale_1);
                        }
                        {
                            v8_1[1] = cake_dsv4_qmul4<6>(kraw4[2], scale_1);
                        }
                        {
                            v8_1[2] = cake_dsv4_qmul4<5>(kraw4[3], scale_1);
                        }
                        {
                            v8_1[3] = cake_dsv4_qmul4<6>(kraw4[3], scale_1);
                        }
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(smem_v_addr + (unsigned int)(row * 128 + (16 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_1[(0) + 3])));
                    int vblock_2 = 32 * o_chunk + 2;
                    unsigned int v8_3[4];
                    {
                        {
                            int vchunk_1 = 16 * o_chunk + 1;
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&kraw4[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4[(0) + 3]))
                                : "r"(kbase + vchunk_1 / 8 * 16384 + (row * 128 + (vchunk_1 % 8 * 16 ^ row % 8 * 16))));
                        }
                        unsigned int scale_2 = sfw32 >> 16 & 255;
                        {
                            v8_3[0] = cake_dsv4_qmul4<5>(kraw4[0], scale_2);
                        }
                        {
                            v8_3[1] = cake_dsv4_qmul4<6>(kraw4[0], scale_2);
                        }
                        {
                            v8_3[2] = cake_dsv4_qmul4<5>(kraw4[1], scale_2);
                        }
                        {
                            v8_3[3] = cake_dsv4_qmul4<6>(kraw4[1], scale_2);
                        }
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(smem_v_addr + (unsigned int)(row * 128 + (32 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_3[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_3[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_3[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_3[(0) + 3])));
                    int vblock_4 = 32 * o_chunk + 3;
                    unsigned int v8_5[4];
                    {
                        unsigned int scale_3 = sfw32 >> 24 & 255;
                        {
                            v8_5[0] = cake_dsv4_qmul4<5>(kraw4[2], scale_3);
                        }
                        {
                            v8_5[1] = cake_dsv4_qmul4<6>(kraw4[2], scale_3);
                        }
                        {
                            v8_5[2] = cake_dsv4_qmul4<5>(kraw4[3], scale_3);
                        }
                        {
                            v8_5[3] = cake_dsv4_qmul4<6>(kraw4[3], scale_3);
                        }
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(smem_v_addr + (unsigned int)(row * 128 + (48 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_5[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_5[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_5[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_5[(0) + 3])));
                    int vblock_6 = 32 * o_chunk + 4;
                    unsigned int v8_7[4];
                    {
                        {
                            int vchunk_2 = 16 * o_chunk + 2;
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&kraw4[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4[(0) + 3]))
                                : "r"(kbase + vchunk_2 / 8 * 16384 + (row * 128 + (vchunk_2 % 8 * 16 ^ row % 8 * 16))));
                        }
                        {
                            sfw32 = smem_kz32[(kz_off + 49152 + row * 32 >> 2) + 8 * o_chunk + 1];
                        }
                        unsigned int scale_4 = sfw32 & 255;
                        {
                            v8_7[0] = cake_dsv4_qmul4<5>(kraw4[0], scale_4);
                        }
                        {
                            v8_7[1] = cake_dsv4_qmul4<6>(kraw4[0], scale_4);
                        }
                        {
                            v8_7[2] = cake_dsv4_qmul4<5>(kraw4[1], scale_4);
                        }
                        {
                            v8_7[3] = cake_dsv4_qmul4<6>(kraw4[1], scale_4);
                        }
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(smem_v_addr + (unsigned int)(row * 128 + (64 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_7[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_7[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_7[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_7[(0) + 3])));
                    int vblock_8 = 32 * o_chunk + 5;
                    unsigned int v8_9[4];
                    {
                        unsigned int scale_5 = sfw32 >> 8 & 255;
                        {
                            v8_9[0] = cake_dsv4_qmul4<5>(kraw4[2], scale_5);
                        }
                        {
                            v8_9[1] = cake_dsv4_qmul4<6>(kraw4[2], scale_5);
                        }
                        {
                            v8_9[2] = cake_dsv4_qmul4<5>(kraw4[3], scale_5);
                        }
                        {
                            v8_9[3] = cake_dsv4_qmul4<6>(kraw4[3], scale_5);
                        }
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(smem_v_addr + (unsigned int)(row * 128 + (80 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_9[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_9[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_9[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_9[(0) + 3])));
                    int vblock_10 = 32 * o_chunk + 6;
                    unsigned int v8_11[4];
                    {
                        {
                            int vchunk_3 = 16 * o_chunk + 3;
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&kraw4[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4[(0) + 3]))
                                : "r"(kbase + vchunk_3 / 8 * 16384 + (row * 128 + (vchunk_3 % 8 * 16 ^ row % 8 * 16))));
                        }
                        unsigned int scale_6 = sfw32 >> 16 & 255;
                        {
                            v8_11[0] = cake_dsv4_qmul4<5>(kraw4[0], scale_6);
                        }
                        {
                            v8_11[1] = cake_dsv4_qmul4<6>(kraw4[0], scale_6);
                        }
                        {
                            v8_11[2] = cake_dsv4_qmul4<5>(kraw4[1], scale_6);
                        }
                        {
                            v8_11[3] = cake_dsv4_qmul4<6>(kraw4[1], scale_6);
                        }
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(smem_v_addr + (unsigned int)(row * 128 + (96 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_11[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_11[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_11[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_11[(0) + 3])));
                    int vblock_12 = 32 * o_chunk + 7;
                    unsigned int v8_13[4];
                    {
                        unsigned int scale_7 = sfw32 >> 24 & 255;
                        {
                            v8_13[0] = cake_dsv4_qmul4<5>(kraw4[2], scale_7);
                        }
                        {
                            v8_13[1] = cake_dsv4_qmul4<6>(kraw4[2], scale_7);
                        }
                        {
                            v8_13[2] = cake_dsv4_qmul4<5>(kraw4[3], scale_7);
                        }
                        {
                            v8_13[3] = cake_dsv4_qmul4<6>(kraw4[3], scale_7);
                        }
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(smem_v_addr + (unsigned int)(row * 128 + (112 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_13[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_13[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_13[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_13[(0) + 3])));
                    int vblock_14 = 32 * o_chunk + 8;
                    unsigned int v8_15[4];
                    {
                        {
                            int vchunk_4 = 16 * o_chunk + 4;
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&kraw4[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4[(0) + 3]))
                                : "r"(kbase + vchunk_4 / 8 * 16384 + (row * 128 + (vchunk_4 % 8 * 16 ^ row % 8 * 16))));
                        }
                        {
                            sfw32 = smem_kz32[(kz_off + 49152 + row * 32 >> 2) + 8 * o_chunk + 2];
                        }
                        unsigned int scale_8 = sfw32 & 255;
                        {
                            v8_15[0] = cake_dsv4_qmul4<5>(kraw4[0], scale_8);
                        }
                        {
                            v8_15[1] = cake_dsv4_qmul4<6>(kraw4[0], scale_8);
                        }
                        {
                            v8_15[2] = cake_dsv4_qmul4<5>(kraw4[1], scale_8);
                        }
                        {
                            v8_15[3] = cake_dsv4_qmul4<6>(kraw4[1], scale_8);
                        }
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(smem_v_addr + (unsigned int)(16384 + (row * 128 + (0 ^ row % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_15[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_15[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_15[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_15[(0) + 3])));
                    int vblock_16 = 32 * o_chunk + 9;
                    unsigned int v8_17[4];
                    {
                        unsigned int scale_9 = sfw32 >> 8 & 255;
                        {
                            v8_17[0] = cake_dsv4_qmul4<5>(kraw4[2], scale_9);
                        }
                        {
                            v8_17[1] = cake_dsv4_qmul4<6>(kraw4[2], scale_9);
                        }
                        {
                            v8_17[2] = cake_dsv4_qmul4<5>(kraw4[3], scale_9);
                        }
                        {
                            v8_17[3] = cake_dsv4_qmul4<6>(kraw4[3], scale_9);
                        }
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(smem_v_addr + (unsigned int)(16384 + (row * 128 + (16 ^ row % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_17[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_17[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_17[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_17[(0) + 3])));
                    int vblock_18 = 32 * o_chunk + 10;
                    unsigned int v8_19[4];
                    {
                        {
                            int vchunk_5 = 16 * o_chunk + 5;
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&kraw4[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4[(0) + 3]))
                                : "r"(kbase + vchunk_5 / 8 * 16384 + (row * 128 + (vchunk_5 % 8 * 16 ^ row % 8 * 16))));
                        }
                        unsigned int scale_10 = sfw32 >> 16 & 255;
                        {
                            v8_19[0] = cake_dsv4_qmul4<5>(kraw4[0], scale_10);
                        }
                        {
                            v8_19[1] = cake_dsv4_qmul4<6>(kraw4[0], scale_10);
                        }
                        {
                            v8_19[2] = cake_dsv4_qmul4<5>(kraw4[1], scale_10);
                        }
                        {
                            v8_19[3] = cake_dsv4_qmul4<6>(kraw4[1], scale_10);
                        }
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(smem_v_addr + (unsigned int)(16384 + (row * 128 + (32 ^ row % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_19[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_19[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_19[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_19[(0) + 3])));
                } else {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(row * 128 + (0 ^ row % 8 * 16))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(row * 128 + (16 ^ row % 8 * 16))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(row * 128 + (32 ^ row % 8 * 16))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(row * 128 + (48 ^ row % 8 * 16))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(row * 128 + (64 ^ row % 8 * 16))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(row * 128 + (80 ^ row % 8 * 16))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(row * 128 + (96 ^ row % 8 * 16))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(row * 128 + (112 ^ row % 8 * 16))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(16384 + (row * 128 + (0 ^ row % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(16384 + (row * 128 + (16 ^ row % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(16384 + (row * 128 + (32 ^ row % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(v_full_addr);
                mbarrier_wait_hint(s_full_addr, it & 1, 10000000);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int col_lo = 64 * half;
                float sv[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(sv[0]), "=f"(sv[1]), "=f"(sv[2]), "=f"(sv[3]), "=f"(sv[4]), "=f"(sv[5]), "=f"(sv[6]), "=f"(sv[7]), "=f"(sv[8]), "=f"(sv[9]), "=f"(sv[10]), "=f"(sv[11]), "=f"(sv[12]), "=f"(sv[13]), "=f"(sv[14]), "=f"(sv[15]), "=f"(sv[16]), "=f"(sv[17]), "=f"(sv[18]), "=f"(sv[19]), "=f"(sv[20]), "=f"(sv[21]), "=f"(sv[22]), "=f"(sv[23]), "=f"(sv[24]), "=f"(sv[25]), "=f"(sv[26]), "=f"(sv[27]), "=f"(sv[28]), "=f"(sv[29]), "=f"(sv[30]), "=f"(sv[31])
                    : "r"(taddr + (unsigned int)col_lo + (unsigned int)(tmem_row_origin << 16)));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int mword = mask_word;
                sv[0] = (((mword & 1) != 0) ? sv[0] : -CAKE_INF);
                sv[1] = (((mword >> 1 & 1) != 0) ? sv[1] : -CAKE_INF);
                sv[2] = (((mword >> 2 & 1) != 0) ? sv[2] : -CAKE_INF);
                sv[3] = (((mword >> 3 & 1) != 0) ? sv[3] : -CAKE_INF);
                sv[4] = (((mword >> 4 & 1) != 0) ? sv[4] : -CAKE_INF);
                sv[5] = (((mword >> 5 & 1) != 0) ? sv[5] : -CAKE_INF);
                sv[6] = (((mword >> 6 & 1) != 0) ? sv[6] : -CAKE_INF);
                sv[7] = (((mword >> 7 & 1) != 0) ? sv[7] : -CAKE_INF);
                sv[8] = (((mword >> 8 & 1) != 0) ? sv[8] : -CAKE_INF);
                sv[9] = (((mword >> 9 & 1) != 0) ? sv[9] : -CAKE_INF);
                sv[10] = (((mword >> 10 & 1) != 0) ? sv[10] : -CAKE_INF);
                sv[11] = (((mword >> 11 & 1) != 0) ? sv[11] : -CAKE_INF);
                sv[12] = (((mword >> 12 & 1) != 0) ? sv[12] : -CAKE_INF);
                sv[13] = (((mword >> 13 & 1) != 0) ? sv[13] : -CAKE_INF);
                sv[14] = (((mword >> 14 & 1) != 0) ? sv[14] : -CAKE_INF);
                sv[15] = (((mword >> 15 & 1) != 0) ? sv[15] : -CAKE_INF);
                sv[16] = (((mword >> 16 & 1) != 0) ? sv[16] : -CAKE_INF);
                sv[17] = (((mword >> 17 & 1) != 0) ? sv[17] : -CAKE_INF);
                sv[18] = (((mword >> 18 & 1) != 0) ? sv[18] : -CAKE_INF);
                sv[19] = (((mword >> 19 & 1) != 0) ? sv[19] : -CAKE_INF);
                sv[20] = (((mword >> 20 & 1) != 0) ? sv[20] : -CAKE_INF);
                sv[21] = (((mword >> 21 & 1) != 0) ? sv[21] : -CAKE_INF);
                sv[22] = (((mword >> 22 & 1) != 0) ? sv[22] : -CAKE_INF);
                sv[23] = (((mword >> 23 & 1) != 0) ? sv[23] : -CAKE_INF);
                sv[24] = (((mword >> 24 & 1) != 0) ? sv[24] : -CAKE_INF);
                sv[25] = (((mword >> 25 & 1) != 0) ? sv[25] : -CAKE_INF);
                sv[26] = (((mword >> 26 & 1) != 0) ? sv[26] : -CAKE_INF);
                sv[27] = (((mword >> 27 & 1) != 0) ? sv[27] : -CAKE_INF);
                sv[28] = (((mword >> 28 & 1) != 0) ? sv[28] : -CAKE_INF);
                sv[29] = (((mword >> 29 & 1) != 0) ? sv[29] : -CAKE_INF);
                sv[30] = (((mword >> 30 & 1) != 0) ? sv[30] : -CAKE_INF);
                sv[31] = (((mword >> 31 & 1) != 0) ? sv[31] : -CAKE_INF);
                float mx[32];
                mx[0] = sv[0];
                mx[1] = sv[1];
                mx[2] = sv[2];
                mx[3] = sv[3];
                mx[4] = sv[4];
                mx[5] = sv[5];
                mx[6] = sv[6];
                mx[7] = sv[7];
                mx[8] = sv[8];
                mx[9] = sv[9];
                mx[10] = sv[10];
                mx[11] = sv[11];
                mx[12] = sv[12];
                mx[13] = sv[13];
                mx[14] = sv[14];
                mx[15] = sv[15];
                mx[16] = sv[16];
                mx[17] = sv[17];
                mx[18] = sv[18];
                mx[19] = sv[19];
                mx[20] = sv[20];
                mx[21] = sv[21];
                mx[22] = sv[22];
                mx[23] = sv[23];
                mx[24] = sv[24];
                mx[25] = sv[25];
                mx[26] = sv[26];
                mx[27] = sv[27];
                mx[28] = sv[28];
                mx[29] = sv[29];
                mx[30] = sv[30];
                mx[31] = sv[31];
                float _max_120 = max_noftz(mx[0], mx[16]);
                mx[0] = _max_120;
                float _max_121 = max_noftz(mx[1], mx[17]);
                mx[1] = _max_121;
                float _max_122 = max_noftz(mx[2], mx[18]);
                mx[2] = _max_122;
                float _max_123 = max_noftz(mx[3], mx[19]);
                mx[3] = _max_123;
                float _max_124 = max_noftz(mx[4], mx[20]);
                mx[4] = _max_124;
                float _max_125 = max_noftz(mx[5], mx[21]);
                mx[5] = _max_125;
                float _max_126 = max_noftz(mx[6], mx[22]);
                mx[6] = _max_126;
                float _max_127 = max_noftz(mx[7], mx[23]);
                mx[7] = _max_127;
                float _max_128 = max_noftz(mx[8], mx[24]);
                mx[8] = _max_128;
                float _max_129 = max_noftz(mx[9], mx[25]);
                mx[9] = _max_129;
                float _max_130 = max_noftz(mx[10], mx[26]);
                mx[10] = _max_130;
                float _max_131 = max_noftz(mx[11], mx[27]);
                mx[11] = _max_131;
                float _max_132 = max_noftz(mx[12], mx[28]);
                mx[12] = _max_132;
                float _max_133 = max_noftz(mx[13], mx[29]);
                mx[13] = _max_133;
                float _max_134 = max_noftz(mx[14], mx[30]);
                mx[14] = _max_134;
                float _max_135 = max_noftz(mx[15], mx[31]);
                mx[15] = _max_135;
                float _max_136 = max_noftz(mx[0], mx[8]);
                mx[0] = _max_136;
                float _max_137 = max_noftz(mx[1], mx[9]);
                mx[1] = _max_137;
                float _max_138 = max_noftz(mx[2], mx[10]);
                mx[2] = _max_138;
                float _max_139 = max_noftz(mx[3], mx[11]);
                mx[3] = _max_139;
                float _max_140 = max_noftz(mx[4], mx[12]);
                mx[4] = _max_140;
                float _max_141 = max_noftz(mx[5], mx[13]);
                mx[5] = _max_141;
                float _max_142 = max_noftz(mx[6], mx[14]);
                mx[6] = _max_142;
                float _max_143 = max_noftz(mx[7], mx[15]);
                mx[7] = _max_143;
                float _max_144 = max_noftz(mx[0], mx[4]);
                mx[0] = _max_144;
                float _max_145 = max_noftz(mx[1], mx[5]);
                mx[1] = _max_145;
                float _max_146 = max_noftz(mx[2], mx[6]);
                mx[2] = _max_146;
                float _max_147 = max_noftz(mx[3], mx[7]);
                mx[3] = _max_147;
                float _max_148 = max_noftz(mx[0], mx[2]);
                mx[0] = _max_148;
                float _max_149 = max_noftz(mx[1], mx[3]);
                mx[1] = _max_149;
                float _max_150 = max_noftz(mx[0], mx[1]);
                mx[0] = _max_150;
                int slice_id = 3 * half;
                smem_pmax[slice_id * 64 + head] = mx[0];
                asm volatile("barrier.sync 10, 384;" ::: "memory");
                float m_tile = smem_pmax[head];
                float _max_151 = max_noftz(m_tile, smem_pmax[64 + head]);
                m_tile = _max_151;
                float _max_152 = max_noftz(m_tile, smem_pmax[128 + head]);
                m_tile = _max_152;
                float _max_153 = max_noftz(m_tile, smem_pmax[192 + head]);
                m_tile = _max_153;
                float _max_154 = max_noftz(m_tile, smem_pmax[256 + head]);
                m_tile = _max_154;
                float _max_155 = max_noftz(m_tile, smem_pmax[320 + head]);
                m_tile = _max_155;
                float cand = m_tile * softmax_scale_log2;
                if (it == 0) {
                    float _max_156 = max_noftz(cand, sink_lane);
                    cand = _max_156;
                }
                float _max_157 = max_noftz(cand, sm[0]);
                cand = _max_157;
                int grow = 0;
                float alpha = 1.0f;
                if (it == 0) {
                    grow = 1;
                }
                if (cand - sm[0] > 8.0f) {
                    grow = 1;
                }
                if (grow != 0) {
                    float _exp2_0 = approx_exp2(sm[0] - cand);
                    alpha = ((sm[0] > -CAKE_INF) ? _exp2_0 : 0.0f);
                    sm[1] = sm[1] * alpha;
                    sm[2] = sm[2] * alpha;
                    sm[0] = cand;
                }
                float m_scaled = ((sm[0] > -CAKE_INF) ? sm[0] : 0.0f);
                unsigned int _vote_4 = __ballot_sync(0xFFFFFFFF, grow != 0);
                unsigned int grow_bits = _vote_4;
                if (local_warp < 2) {
                    smem_alpha[head] = alpha;
                    if (lane == 0) {
                        smem_flag[local_warp] = grow_bits;
                    }
                }
                asm volatile("barrier.sync 10, 384;" ::: "memory");
                if (it > 0) {
                    unsigned int any_grow = smem_flag[0] | smem_flag[1];
                    if (any_grow != 0) {
                        float alpha_c[32];
                        alpha_c[0] = smem_alpha[0];
                        alpha_c[1] = smem_alpha[1];
                        alpha_c[2] = smem_alpha[2];
                        alpha_c[3] = smem_alpha[3];
                        alpha_c[4] = smem_alpha[4];
                        alpha_c[5] = smem_alpha[5];
                        alpha_c[6] = smem_alpha[6];
                        alpha_c[7] = smem_alpha[7];
                        alpha_c[8] = smem_alpha[8];
                        alpha_c[9] = smem_alpha[9];
                        alpha_c[10] = smem_alpha[10];
                        alpha_c[11] = smem_alpha[11];
                        alpha_c[12] = smem_alpha[12];
                        alpha_c[13] = smem_alpha[13];
                        alpha_c[14] = smem_alpha[14];
                        alpha_c[15] = smem_alpha[15];
                        alpha_c[16] = smem_alpha[16];
                        alpha_c[17] = smem_alpha[17];
                        alpha_c[18] = smem_alpha[18];
                        alpha_c[19] = smem_alpha[19];
                        alpha_c[20] = smem_alpha[20];
                        alpha_c[21] = smem_alpha[21];
                        alpha_c[22] = smem_alpha[22];
                        alpha_c[23] = smem_alpha[23];
                        alpha_c[24] = smem_alpha[24];
                        alpha_c[25] = smem_alpha[25];
                        alpha_c[26] = smem_alpha[26];
                        alpha_c[27] = smem_alpha[27];
                        alpha_c[28] = smem_alpha[28];
                        alpha_c[29] = smem_alpha[29];
                        alpha_c[30] = smem_alpha[30];
                        alpha_c[31] = smem_alpha[31];
                        float ov[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(ov[0]), "=f"(ov[1]), "=f"(ov[2]), "=f"(ov[3]), "=f"(ov[4]), "=f"(ov[5]), "=f"(ov[6]), "=f"(ov[7]), "=f"(ov[8]), "=f"(ov[9]), "=f"(ov[10]), "=f"(ov[11]), "=f"(ov[12]), "=f"(ov[13]), "=f"(ov[14]), "=f"(ov[15]), "=f"(ov[16]), "=f"(ov[17]), "=f"(ov[18]), "=f"(ov[19]), "=f"(ov[20]), "=f"(ov[21]), "=f"(ov[22]), "=f"(ov[23]), "=f"(ov[24]), "=f"(ov[25]), "=f"(ov[26]), "=f"(ov[27]), "=f"(ov[28]), "=f"(ov[29]), "=f"(ov[30]), "=f"(ov[31])
                            : "r"(taddr + 128 + (unsigned int)(tmem_row_origin << 16)));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        ov[0] = ov[0] * alpha_c[0];
                        ov[1] = ov[1] * alpha_c[1];
                        ov[2] = ov[2] * alpha_c[2];
                        ov[3] = ov[3] * alpha_c[3];
                        ov[4] = ov[4] * alpha_c[4];
                        ov[5] = ov[5] * alpha_c[5];
                        ov[6] = ov[6] * alpha_c[6];
                        ov[7] = ov[7] * alpha_c[7];
                        ov[8] = ov[8] * alpha_c[8];
                        ov[9] = ov[9] * alpha_c[9];
                        ov[10] = ov[10] * alpha_c[10];
                        ov[11] = ov[11] * alpha_c[11];
                        ov[12] = ov[12] * alpha_c[12];
                        ov[13] = ov[13] * alpha_c[13];
                        ov[14] = ov[14] * alpha_c[14];
                        ov[15] = ov[15] * alpha_c[15];
                        ov[16] = ov[16] * alpha_c[16];
                        ov[17] = ov[17] * alpha_c[17];
                        ov[18] = ov[18] * alpha_c[18];
                        ov[19] = ov[19] * alpha_c[19];
                        ov[20] = ov[20] * alpha_c[20];
                        ov[21] = ov[21] * alpha_c[21];
                        ov[22] = ov[22] * alpha_c[22];
                        ov[23] = ov[23] * alpha_c[23];
                        ov[24] = ov[24] * alpha_c[24];
                        ov[25] = ov[25] * alpha_c[25];
                        ov[26] = ov[26] * alpha_c[26];
                        ov[27] = ov[27] * alpha_c[27];
                        ov[28] = ov[28] * alpha_c[28];
                        ov[29] = ov[29] * alpha_c[29];
                        ov[30] = ov[30] * alpha_c[30];
                        ov[31] = ov[31] * alpha_c[31];
                        tmem_st_x32_f32(taddr + 128 + (unsigned int)(tmem_row_origin << 16), ov);
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(ov[0]), "=f"(ov[1]), "=f"(ov[2]), "=f"(ov[3]), "=f"(ov[4]), "=f"(ov[5]), "=f"(ov[6]), "=f"(ov[7]), "=f"(ov[8]), "=f"(ov[9]), "=f"(ov[10]), "=f"(ov[11]), "=f"(ov[12]), "=f"(ov[13]), "=f"(ov[14]), "=f"(ov[15]), "=f"(ov[16]), "=f"(ov[17]), "=f"(ov[18]), "=f"(ov[19]), "=f"(ov[20]), "=f"(ov[21]), "=f"(ov[22]), "=f"(ov[23]), "=f"(ov[24]), "=f"(ov[25]), "=f"(ov[26]), "=f"(ov[27]), "=f"(ov[28]), "=f"(ov[29]), "=f"(ov[30]), "=f"(ov[31])
                            : "r"(taddr + 192 + (unsigned int)(tmem_row_origin << 16)));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        ov[0] = ov[0] * alpha_c[0];
                        ov[1] = ov[1] * alpha_c[1];
                        ov[2] = ov[2] * alpha_c[2];
                        ov[3] = ov[3] * alpha_c[3];
                        ov[4] = ov[4] * alpha_c[4];
                        ov[5] = ov[5] * alpha_c[5];
                        ov[6] = ov[6] * alpha_c[6];
                        ov[7] = ov[7] * alpha_c[7];
                        ov[8] = ov[8] * alpha_c[8];
                        ov[9] = ov[9] * alpha_c[9];
                        ov[10] = ov[10] * alpha_c[10];
                        ov[11] = ov[11] * alpha_c[11];
                        ov[12] = ov[12] * alpha_c[12];
                        ov[13] = ov[13] * alpha_c[13];
                        ov[14] = ov[14] * alpha_c[14];
                        ov[15] = ov[15] * alpha_c[15];
                        ov[16] = ov[16] * alpha_c[16];
                        ov[17] = ov[17] * alpha_c[17];
                        ov[18] = ov[18] * alpha_c[18];
                        ov[19] = ov[19] * alpha_c[19];
                        ov[20] = ov[20] * alpha_c[20];
                        ov[21] = ov[21] * alpha_c[21];
                        ov[22] = ov[22] * alpha_c[22];
                        ov[23] = ov[23] * alpha_c[23];
                        ov[24] = ov[24] * alpha_c[24];
                        ov[25] = ov[25] * alpha_c[25];
                        ov[26] = ov[26] * alpha_c[26];
                        ov[27] = ov[27] * alpha_c[27];
                        ov[28] = ov[28] * alpha_c[28];
                        ov[29] = ov[29] * alpha_c[29];
                        ov[30] = ov[30] * alpha_c[30];
                        ov[31] = ov[31] * alpha_c[31];
                        tmem_st_x32_f32(taddr + 192 + (unsigned int)(tmem_row_origin << 16), ov);
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(ov[0]), "=f"(ov[1]), "=f"(ov[2]), "=f"(ov[3]), "=f"(ov[4]), "=f"(ov[5]), "=f"(ov[6]), "=f"(ov[7]), "=f"(ov[8]), "=f"(ov[9]), "=f"(ov[10]), "=f"(ov[11]), "=f"(ov[12]), "=f"(ov[13]), "=f"(ov[14]), "=f"(ov[15]), "=f"(ov[16]), "=f"(ov[17]), "=f"(ov[18]), "=f"(ov[19]), "=f"(ov[20]), "=f"(ov[21]), "=f"(ov[22]), "=f"(ov[23]), "=f"(ov[24]), "=f"(ov[25]), "=f"(ov[26]), "=f"(ov[27]), "=f"(ov[28]), "=f"(ov[29]), "=f"(ov[30]), "=f"(ov[31])
                            : "r"(taddr + 256 + (unsigned int)(tmem_row_origin << 16)));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        ov[0] = ov[0] * alpha_c[0];
                        ov[1] = ov[1] * alpha_c[1];
                        ov[2] = ov[2] * alpha_c[2];
                        ov[3] = ov[3] * alpha_c[3];
                        ov[4] = ov[4] * alpha_c[4];
                        ov[5] = ov[5] * alpha_c[5];
                        ov[6] = ov[6] * alpha_c[6];
                        ov[7] = ov[7] * alpha_c[7];
                        ov[8] = ov[8] * alpha_c[8];
                        ov[9] = ov[9] * alpha_c[9];
                        ov[10] = ov[10] * alpha_c[10];
                        ov[11] = ov[11] * alpha_c[11];
                        ov[12] = ov[12] * alpha_c[12];
                        ov[13] = ov[13] * alpha_c[13];
                        ov[14] = ov[14] * alpha_c[14];
                        ov[15] = ov[15] * alpha_c[15];
                        ov[16] = ov[16] * alpha_c[16];
                        ov[17] = ov[17] * alpha_c[17];
                        ov[18] = ov[18] * alpha_c[18];
                        ov[19] = ov[19] * alpha_c[19];
                        ov[20] = ov[20] * alpha_c[20];
                        ov[21] = ov[21] * alpha_c[21];
                        ov[22] = ov[22] * alpha_c[22];
                        ov[23] = ov[23] * alpha_c[23];
                        ov[24] = ov[24] * alpha_c[24];
                        ov[25] = ov[25] * alpha_c[25];
                        ov[26] = ov[26] * alpha_c[26];
                        ov[27] = ov[27] * alpha_c[27];
                        ov[28] = ov[28] * alpha_c[28];
                        ov[29] = ov[29] * alpha_c[29];
                        ov[30] = ov[30] * alpha_c[30];
                        ov[31] = ov[31] * alpha_c[31];
                        tmem_st_x32_f32(taddr + 256 + (unsigned int)(tmem_row_origin << 16), ov);
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(ov[0]), "=f"(ov[1]), "=f"(ov[2]), "=f"(ov[3]), "=f"(ov[4]), "=f"(ov[5]), "=f"(ov[6]), "=f"(ov[7]), "=f"(ov[8]), "=f"(ov[9]), "=f"(ov[10]), "=f"(ov[11]), "=f"(ov[12]), "=f"(ov[13]), "=f"(ov[14]), "=f"(ov[15]), "=f"(ov[16]), "=f"(ov[17]), "=f"(ov[18]), "=f"(ov[19]), "=f"(ov[20]), "=f"(ov[21]), "=f"(ov[22]), "=f"(ov[23]), "=f"(ov[24]), "=f"(ov[25]), "=f"(ov[26]), "=f"(ov[27]), "=f"(ov[28]), "=f"(ov[29]), "=f"(ov[30]), "=f"(ov[31])
                            : "r"(taddr + 320 + (unsigned int)(tmem_row_origin << 16)));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        ov[0] = ov[0] * alpha_c[0];
                        ov[1] = ov[1] * alpha_c[1];
                        ov[2] = ov[2] * alpha_c[2];
                        ov[3] = ov[3] * alpha_c[3];
                        ov[4] = ov[4] * alpha_c[4];
                        ov[5] = ov[5] * alpha_c[5];
                        ov[6] = ov[6] * alpha_c[6];
                        ov[7] = ov[7] * alpha_c[7];
                        ov[8] = ov[8] * alpha_c[8];
                        ov[9] = ov[9] * alpha_c[9];
                        ov[10] = ov[10] * alpha_c[10];
                        ov[11] = ov[11] * alpha_c[11];
                        ov[12] = ov[12] * alpha_c[12];
                        ov[13] = ov[13] * alpha_c[13];
                        ov[14] = ov[14] * alpha_c[14];
                        ov[15] = ov[15] * alpha_c[15];
                        ov[16] = ov[16] * alpha_c[16];
                        ov[17] = ov[17] * alpha_c[17];
                        ov[18] = ov[18] * alpha_c[18];
                        ov[19] = ov[19] * alpha_c[19];
                        ov[20] = ov[20] * alpha_c[20];
                        ov[21] = ov[21] * alpha_c[21];
                        ov[22] = ov[22] * alpha_c[22];
                        ov[23] = ov[23] * alpha_c[23];
                        ov[24] = ov[24] * alpha_c[24];
                        ov[25] = ov[25] * alpha_c[25];
                        ov[26] = ov[26] * alpha_c[26];
                        ov[27] = ov[27] * alpha_c[27];
                        ov[28] = ov[28] * alpha_c[28];
                        ov[29] = ov[29] * alpha_c[29];
                        ov[30] = ov[30] * alpha_c[30];
                        ov[31] = ov[31] * alpha_c[31];
                        tmem_st_x32_f32(taddr + 320 + (unsigned int)(tmem_row_origin << 16), ov);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                }
                float psum = 0.0f;
                float rsum = 0.0f;
                float _exp2_1 = approx_exp2(sv[0] * softmax_scale_log2 - m_scaled);
                sv[0] = _exp2_1;
                float _exp2_2 = approx_exp2(sv[1] * softmax_scale_log2 - m_scaled);
                sv[1] = _exp2_2;
                float _exp2_3 = approx_exp2(sv[2] * softmax_scale_log2 - m_scaled);
                sv[2] = _exp2_3;
                float _exp2_4 = approx_exp2(sv[3] * softmax_scale_log2 - m_scaled);
                sv[3] = _exp2_4;
                float _exp2_5 = approx_exp2(sv[4] * softmax_scale_log2 - m_scaled);
                sv[4] = _exp2_5;
                float _exp2_6 = approx_exp2(sv[5] * softmax_scale_log2 - m_scaled);
                sv[5] = _exp2_6;
                float _exp2_7 = approx_exp2(sv[6] * softmax_scale_log2 - m_scaled);
                sv[6] = _exp2_7;
                float _exp2_8 = approx_exp2(sv[7] * softmax_scale_log2 - m_scaled);
                sv[7] = _exp2_8;
                float _exp2_9 = approx_exp2(sv[8] * softmax_scale_log2 - m_scaled);
                sv[8] = _exp2_9;
                float _exp2_10 = approx_exp2(sv[9] * softmax_scale_log2 - m_scaled);
                sv[9] = _exp2_10;
                float _exp2_11 = approx_exp2(sv[10] * softmax_scale_log2 - m_scaled);
                sv[10] = _exp2_11;
                float _exp2_12 = approx_exp2(sv[11] * softmax_scale_log2 - m_scaled);
                sv[11] = _exp2_12;
                float _exp2_13 = approx_exp2(sv[12] * softmax_scale_log2 - m_scaled);
                sv[12] = _exp2_13;
                float _exp2_14 = approx_exp2(sv[13] * softmax_scale_log2 - m_scaled);
                sv[13] = _exp2_14;
                float _exp2_15 = approx_exp2(sv[14] * softmax_scale_log2 - m_scaled);
                sv[14] = _exp2_15;
                float _exp2_16 = approx_exp2(sv[15] * softmax_scale_log2 - m_scaled);
                sv[15] = _exp2_16;
                float _exp2_17 = approx_exp2(sv[16] * softmax_scale_log2 - m_scaled);
                sv[16] = _exp2_17;
                float _exp2_18 = approx_exp2(sv[17] * softmax_scale_log2 - m_scaled);
                sv[17] = _exp2_18;
                float _exp2_19 = approx_exp2(sv[18] * softmax_scale_log2 - m_scaled);
                sv[18] = _exp2_19;
                float _exp2_20 = approx_exp2(sv[19] * softmax_scale_log2 - m_scaled);
                sv[19] = _exp2_20;
                float _exp2_21 = approx_exp2(sv[20] * softmax_scale_log2 - m_scaled);
                sv[20] = _exp2_21;
                float _exp2_22 = approx_exp2(sv[21] * softmax_scale_log2 - m_scaled);
                sv[21] = _exp2_22;
                float _exp2_23 = approx_exp2(sv[22] * softmax_scale_log2 - m_scaled);
                sv[22] = _exp2_23;
                float _exp2_24 = approx_exp2(sv[23] * softmax_scale_log2 - m_scaled);
                sv[23] = _exp2_24;
                float _exp2_25 = approx_exp2(sv[24] * softmax_scale_log2 - m_scaled);
                sv[24] = _exp2_25;
                float _exp2_26 = approx_exp2(sv[25] * softmax_scale_log2 - m_scaled);
                sv[25] = _exp2_26;
                float _exp2_27 = approx_exp2(sv[26] * softmax_scale_log2 - m_scaled);
                sv[26] = _exp2_27;
                float _exp2_28 = approx_exp2(sv[27] * softmax_scale_log2 - m_scaled);
                sv[27] = _exp2_28;
                float _exp2_29 = approx_exp2(sv[28] * softmax_scale_log2 - m_scaled);
                sv[28] = _exp2_29;
                float _exp2_30 = approx_exp2(sv[29] * softmax_scale_log2 - m_scaled);
                sv[29] = _exp2_30;
                float _exp2_31 = approx_exp2(sv[30] * softmax_scale_log2 - m_scaled);
                sv[30] = _exp2_31;
                float _exp2_32 = approx_exp2(sv[31] * softmax_scale_log2 - m_scaled);
                sv[31] = _exp2_32;
                unsigned int pw[4];
                uint16_t _e4m3x2_f32_184;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_184) : "f"(sv[1]), "f"(sv[0]));
                uint16_t pair0 = _e4m3x2_f32_184;
                uint16_t _e4m3x2_f32_185;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_185) : "f"(sv[3]), "f"(sv[2]));
                uint16_t pair1 = _e4m3x2_f32_185;
                pw[0] = (unsigned int)pair0 | (unsigned int)pair1 << 16;
                uint16_t _e4m3x2_f32_186;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_186) : "f"(sv[5]), "f"(sv[4]));
                uint16_t pair0_0 = _e4m3x2_f32_186;
                uint16_t _e4m3x2_f32_187;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_187) : "f"(sv[7]), "f"(sv[6]));
                uint16_t pair1_1 = _e4m3x2_f32_187;
                pw[1] = (unsigned int)pair0_0 | (unsigned int)pair1_1 << 16;
                uint16_t _e4m3x2_f32_188;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_188) : "f"(sv[9]), "f"(sv[8]));
                uint16_t pair0_2 = _e4m3x2_f32_188;
                uint16_t _e4m3x2_f32_189;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_189) : "f"(sv[11]), "f"(sv[10]));
                uint16_t pair1_3 = _e4m3x2_f32_189;
                pw[2] = (unsigned int)pair0_2 | (unsigned int)pair1_3 << 16;
                uint16_t _e4m3x2_f32_190;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_190) : "f"(sv[13]), "f"(sv[12]));
                uint16_t pair0_4 = _e4m3x2_f32_190;
                uint16_t _e4m3x2_f32_191;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_191) : "f"(sv[15]), "f"(sv[14]));
                uint16_t pair1_5 = _e4m3x2_f32_191;
                pw[3] = (unsigned int)pair0_4 | (unsigned int)pair1_5 << 16;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(pbase + (head * 128 + ((col_lo >> 4) * 16 ^ head % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&pw[0])), "r"(*reinterpret_cast<uint32_t*>(&pw[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&pw[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&pw[(0) + 3])));
                unsigned int pw_6[4];
                uint16_t _e4m3x2_f32_192;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_192) : "f"(sv[17]), "f"(sv[16]));
                uint16_t pair0_7 = _e4m3x2_f32_192;
                uint16_t _e4m3x2_f32_193;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_193) : "f"(sv[19]), "f"(sv[18]));
                uint16_t pair1_8 = _e4m3x2_f32_193;
                pw_6[0] = (unsigned int)pair0_7 | (unsigned int)pair1_8 << 16;
                uint16_t _e4m3x2_f32_194;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_194) : "f"(sv[21]), "f"(sv[20]));
                uint16_t pair0_9 = _e4m3x2_f32_194;
                uint16_t _e4m3x2_f32_195;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_195) : "f"(sv[23]), "f"(sv[22]));
                uint16_t pair1_10 = _e4m3x2_f32_195;
                pw_6[1] = (unsigned int)pair0_9 | (unsigned int)pair1_10 << 16;
                uint16_t _e4m3x2_f32_196;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_196) : "f"(sv[25]), "f"(sv[24]));
                uint16_t pair0_11 = _e4m3x2_f32_196;
                uint16_t _e4m3x2_f32_197;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_197) : "f"(sv[27]), "f"(sv[26]));
                uint16_t pair1_12 = _e4m3x2_f32_197;
                pw_6[2] = (unsigned int)pair0_11 | (unsigned int)pair1_12 << 16;
                uint16_t _e4m3x2_f32_198;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_198) : "f"(sv[29]), "f"(sv[28]));
                uint16_t pair0_13 = _e4m3x2_f32_198;
                uint16_t _e4m3x2_f32_199;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_199) : "f"(sv[31]), "f"(sv[30]));
                uint16_t pair1_14 = _e4m3x2_f32_199;
                pw_6[3] = (unsigned int)pair0_13 | (unsigned int)pair1_14 << 16;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(pbase + (head * 128 + (((col_lo >> 4) + 1) * 16 ^ head % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&pw_6[0])), "r"(*reinterpret_cast<uint32_t*>(&pw_6[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&pw_6[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&pw_6[(0) + 3])));
                psum = psum + sv[0];
                float _fp8_rt_0;
                uint16_t _e4m3x2_52;
                uint32_t _f16x2_52;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_52) : "f"(0.0f), "f"(sv[0]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_52) : "h"(_e4m3x2_52));
                uint16_t _fp8_h0_52 = (uint16_t)(_f16x2_52 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_52));
                rsum = rsum + _fp8_rt_0;
                psum = psum + sv[1];
                float _fp8_rt_1;
                uint16_t _e4m3x2_53;
                uint32_t _f16x2_53;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_53) : "f"(0.0f), "f"(sv[1]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_53) : "h"(_e4m3x2_53));
                uint16_t _fp8_h0_53 = (uint16_t)(_f16x2_53 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_1) : "h"(_fp8_h0_53));
                rsum = rsum + _fp8_rt_1;
                psum = psum + sv[2];
                float _fp8_rt_2;
                uint16_t _e4m3x2_54;
                uint32_t _f16x2_54;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_54) : "f"(0.0f), "f"(sv[2]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_54) : "h"(_e4m3x2_54));
                uint16_t _fp8_h0_54 = (uint16_t)(_f16x2_54 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_2) : "h"(_fp8_h0_54));
                rsum = rsum + _fp8_rt_2;
                psum = psum + sv[3];
                float _fp8_rt_3;
                uint16_t _e4m3x2_55;
                uint32_t _f16x2_55;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_55) : "f"(0.0f), "f"(sv[3]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_55) : "h"(_e4m3x2_55));
                uint16_t _fp8_h0_55 = (uint16_t)(_f16x2_55 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_3) : "h"(_fp8_h0_55));
                rsum = rsum + _fp8_rt_3;
                psum = psum + sv[4];
                float _fp8_rt_4;
                uint16_t _e4m3x2_56;
                uint32_t _f16x2_56;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_56) : "f"(0.0f), "f"(sv[4]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_56) : "h"(_e4m3x2_56));
                uint16_t _fp8_h0_56 = (uint16_t)(_f16x2_56 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_4) : "h"(_fp8_h0_56));
                rsum = rsum + _fp8_rt_4;
                psum = psum + sv[5];
                float _fp8_rt_5;
                uint16_t _e4m3x2_57;
                uint32_t _f16x2_57;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_57) : "f"(0.0f), "f"(sv[5]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_57) : "h"(_e4m3x2_57));
                uint16_t _fp8_h0_57 = (uint16_t)(_f16x2_57 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_5) : "h"(_fp8_h0_57));
                rsum = rsum + _fp8_rt_5;
                psum = psum + sv[6];
                float _fp8_rt_6;
                uint16_t _e4m3x2_58;
                uint32_t _f16x2_58;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_58) : "f"(0.0f), "f"(sv[6]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_58) : "h"(_e4m3x2_58));
                uint16_t _fp8_h0_58 = (uint16_t)(_f16x2_58 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_6) : "h"(_fp8_h0_58));
                rsum = rsum + _fp8_rt_6;
                psum = psum + sv[7];
                float _fp8_rt_7;
                uint16_t _e4m3x2_59;
                uint32_t _f16x2_59;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_59) : "f"(0.0f), "f"(sv[7]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_59) : "h"(_e4m3x2_59));
                uint16_t _fp8_h0_59 = (uint16_t)(_f16x2_59 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_7) : "h"(_fp8_h0_59));
                rsum = rsum + _fp8_rt_7;
                psum = psum + sv[8];
                float _fp8_rt_8;
                uint16_t _e4m3x2_60;
                uint32_t _f16x2_60;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_60) : "f"(0.0f), "f"(sv[8]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_60) : "h"(_e4m3x2_60));
                uint16_t _fp8_h0_60 = (uint16_t)(_f16x2_60 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_8) : "h"(_fp8_h0_60));
                rsum = rsum + _fp8_rt_8;
                psum = psum + sv[9];
                float _fp8_rt_9;
                uint16_t _e4m3x2_61;
                uint32_t _f16x2_61;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_61) : "f"(0.0f), "f"(sv[9]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_61) : "h"(_e4m3x2_61));
                uint16_t _fp8_h0_61 = (uint16_t)(_f16x2_61 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_9) : "h"(_fp8_h0_61));
                rsum = rsum + _fp8_rt_9;
                psum = psum + sv[10];
                float _fp8_rt_10;
                uint16_t _e4m3x2_62;
                uint32_t _f16x2_62;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_62) : "f"(0.0f), "f"(sv[10]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_62) : "h"(_e4m3x2_62));
                uint16_t _fp8_h0_62 = (uint16_t)(_f16x2_62 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_10) : "h"(_fp8_h0_62));
                rsum = rsum + _fp8_rt_10;
                psum = psum + sv[11];
                float _fp8_rt_11;
                uint16_t _e4m3x2_63;
                uint32_t _f16x2_63;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_63) : "f"(0.0f), "f"(sv[11]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_63) : "h"(_e4m3x2_63));
                uint16_t _fp8_h0_63 = (uint16_t)(_f16x2_63 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_11) : "h"(_fp8_h0_63));
                rsum = rsum + _fp8_rt_11;
                psum = psum + sv[12];
                float _fp8_rt_12;
                uint16_t _e4m3x2_64;
                uint32_t _f16x2_64;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_64) : "f"(0.0f), "f"(sv[12]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_64) : "h"(_e4m3x2_64));
                uint16_t _fp8_h0_64 = (uint16_t)(_f16x2_64 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_12) : "h"(_fp8_h0_64));
                rsum = rsum + _fp8_rt_12;
                psum = psum + sv[13];
                float _fp8_rt_13;
                uint16_t _e4m3x2_65;
                uint32_t _f16x2_65;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_65) : "f"(0.0f), "f"(sv[13]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_65) : "h"(_e4m3x2_65));
                uint16_t _fp8_h0_65 = (uint16_t)(_f16x2_65 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_13) : "h"(_fp8_h0_65));
                rsum = rsum + _fp8_rt_13;
                psum = psum + sv[14];
                float _fp8_rt_14;
                uint16_t _e4m3x2_66;
                uint32_t _f16x2_66;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_66) : "f"(0.0f), "f"(sv[14]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_66) : "h"(_e4m3x2_66));
                uint16_t _fp8_h0_66 = (uint16_t)(_f16x2_66 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_14) : "h"(_fp8_h0_66));
                rsum = rsum + _fp8_rt_14;
                psum = psum + sv[15];
                float _fp8_rt_15;
                uint16_t _e4m3x2_67;
                uint32_t _f16x2_67;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_67) : "f"(0.0f), "f"(sv[15]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_67) : "h"(_e4m3x2_67));
                uint16_t _fp8_h0_67 = (uint16_t)(_f16x2_67 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_15) : "h"(_fp8_h0_67));
                rsum = rsum + _fp8_rt_15;
                psum = psum + sv[16];
                float _fp8_rt_16;
                uint16_t _e4m3x2_68;
                uint32_t _f16x2_68;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_68) : "f"(0.0f), "f"(sv[16]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_68) : "h"(_e4m3x2_68));
                uint16_t _fp8_h0_68 = (uint16_t)(_f16x2_68 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_16) : "h"(_fp8_h0_68));
                rsum = rsum + _fp8_rt_16;
                psum = psum + sv[17];
                float _fp8_rt_17;
                uint16_t _e4m3x2_69;
                uint32_t _f16x2_69;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_69) : "f"(0.0f), "f"(sv[17]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_69) : "h"(_e4m3x2_69));
                uint16_t _fp8_h0_69 = (uint16_t)(_f16x2_69 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_17) : "h"(_fp8_h0_69));
                rsum = rsum + _fp8_rt_17;
                psum = psum + sv[18];
                float _fp8_rt_18;
                uint16_t _e4m3x2_70;
                uint32_t _f16x2_70;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_70) : "f"(0.0f), "f"(sv[18]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_70) : "h"(_e4m3x2_70));
                uint16_t _fp8_h0_70 = (uint16_t)(_f16x2_70 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_18) : "h"(_fp8_h0_70));
                rsum = rsum + _fp8_rt_18;
                psum = psum + sv[19];
                float _fp8_rt_19;
                uint16_t _e4m3x2_71;
                uint32_t _f16x2_71;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_71) : "f"(0.0f), "f"(sv[19]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_71) : "h"(_e4m3x2_71));
                uint16_t _fp8_h0_71 = (uint16_t)(_f16x2_71 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_19) : "h"(_fp8_h0_71));
                rsum = rsum + _fp8_rt_19;
                psum = psum + sv[20];
                float _fp8_rt_20;
                uint16_t _e4m3x2_72;
                uint32_t _f16x2_72;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_72) : "f"(0.0f), "f"(sv[20]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_72) : "h"(_e4m3x2_72));
                uint16_t _fp8_h0_72 = (uint16_t)(_f16x2_72 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_20) : "h"(_fp8_h0_72));
                rsum = rsum + _fp8_rt_20;
                psum = psum + sv[21];
                float _fp8_rt_21;
                uint16_t _e4m3x2_73;
                uint32_t _f16x2_73;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_73) : "f"(0.0f), "f"(sv[21]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_73) : "h"(_e4m3x2_73));
                uint16_t _fp8_h0_73 = (uint16_t)(_f16x2_73 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_21) : "h"(_fp8_h0_73));
                rsum = rsum + _fp8_rt_21;
                psum = psum + sv[22];
                float _fp8_rt_22;
                uint16_t _e4m3x2_74;
                uint32_t _f16x2_74;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_74) : "f"(0.0f), "f"(sv[22]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_74) : "h"(_e4m3x2_74));
                uint16_t _fp8_h0_74 = (uint16_t)(_f16x2_74 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_22) : "h"(_fp8_h0_74));
                rsum = rsum + _fp8_rt_22;
                psum = psum + sv[23];
                float _fp8_rt_23;
                uint16_t _e4m3x2_75;
                uint32_t _f16x2_75;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_75) : "f"(0.0f), "f"(sv[23]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_75) : "h"(_e4m3x2_75));
                uint16_t _fp8_h0_75 = (uint16_t)(_f16x2_75 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_23) : "h"(_fp8_h0_75));
                rsum = rsum + _fp8_rt_23;
                psum = psum + sv[24];
                float _fp8_rt_24;
                uint16_t _e4m3x2_76;
                uint32_t _f16x2_76;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_76) : "f"(0.0f), "f"(sv[24]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_76) : "h"(_e4m3x2_76));
                uint16_t _fp8_h0_76 = (uint16_t)(_f16x2_76 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_24) : "h"(_fp8_h0_76));
                rsum = rsum + _fp8_rt_24;
                psum = psum + sv[25];
                float _fp8_rt_25;
                uint16_t _e4m3x2_77;
                uint32_t _f16x2_77;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_77) : "f"(0.0f), "f"(sv[25]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_77) : "h"(_e4m3x2_77));
                uint16_t _fp8_h0_77 = (uint16_t)(_f16x2_77 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_25) : "h"(_fp8_h0_77));
                rsum = rsum + _fp8_rt_25;
                psum = psum + sv[26];
                float _fp8_rt_26;
                uint16_t _e4m3x2_78;
                uint32_t _f16x2_78;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_78) : "f"(0.0f), "f"(sv[26]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_78) : "h"(_e4m3x2_78));
                uint16_t _fp8_h0_78 = (uint16_t)(_f16x2_78 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_26) : "h"(_fp8_h0_78));
                rsum = rsum + _fp8_rt_26;
                psum = psum + sv[27];
                float _fp8_rt_27;
                uint16_t _e4m3x2_79;
                uint32_t _f16x2_79;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_79) : "f"(0.0f), "f"(sv[27]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_79) : "h"(_e4m3x2_79));
                uint16_t _fp8_h0_79 = (uint16_t)(_f16x2_79 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_27) : "h"(_fp8_h0_79));
                rsum = rsum + _fp8_rt_27;
                psum = psum + sv[28];
                float _fp8_rt_28;
                uint16_t _e4m3x2_80;
                uint32_t _f16x2_80;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_80) : "f"(0.0f), "f"(sv[28]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_80) : "h"(_e4m3x2_80));
                uint16_t _fp8_h0_80 = (uint16_t)(_f16x2_80 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_28) : "h"(_fp8_h0_80));
                rsum = rsum + _fp8_rt_28;
                psum = psum + sv[29];
                float _fp8_rt_29;
                uint16_t _e4m3x2_81;
                uint32_t _f16x2_81;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_81) : "f"(0.0f), "f"(sv[29]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_81) : "h"(_e4m3x2_81));
                uint16_t _fp8_h0_81 = (uint16_t)(_f16x2_81 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_29) : "h"(_fp8_h0_81));
                rsum = rsum + _fp8_rt_29;
                psum = psum + sv[30];
                float _fp8_rt_30;
                uint16_t _e4m3x2_82;
                uint32_t _f16x2_82;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_82) : "f"(0.0f), "f"(sv[30]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_82) : "h"(_e4m3x2_82));
                uint16_t _fp8_h0_82 = (uint16_t)(_f16x2_82 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_30) : "h"(_fp8_h0_82));
                rsum = rsum + _fp8_rt_30;
                psum = psum + sv[31];
                float _fp8_rt_31;
                uint16_t _e4m3x2_83;
                uint32_t _f16x2_83;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_83) : "f"(0.0f), "f"(sv[31]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_83) : "h"(_e4m3x2_83));
                uint16_t _fp8_h0_83 = (uint16_t)(_f16x2_83 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_31) : "h"(_fp8_h0_83));
                rsum = rsum + _fp8_rt_31;
                if (it == 0) {
                    float sink_term = 0.0f;
                    if (half == 0) {
                        float _exp2_33 = approx_exp2(sink_lane - m_scaled);
                        sink_term = _exp2_33;
                    }
                    sm[1] = sink_term;
                    sm[2] = sink_term;
                }
                sm[1] = sm[1] + psum;
                sm[2] = sm[2] + rsum;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(p_full_addr);
            }
            int last_par = tiles_per_split - 1 & 1;
            mbarrier_wait_hint(o_full_addr, last_par, 10000000);
            mbarrier_wait_hint(o_full_addr + 8, last_par, 10000000);
            mbarrier_wait_hint(o_full_addr + 16, last_par, 10000000);
            mbarrier_wait_hint(o_full_addr + 24, last_par, 10000000);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int slice_id_1 = 3 * half;
            smem_xsum[slice_id_1 * 64 + head] = sm[1];
            smem_xsum[384 + slice_id_1 * 64 + head] = sm[2];
            asm volatile("barrier.sync 10, 384;" ::: "memory");
            if (local_warp < 2) {
                float l_tot = smem_xsum[head];
                float r_tot = smem_xsum[384 + head];
                l_tot = l_tot + smem_xsum[64 + head];
                r_tot = r_tot + smem_xsum[448 + head];
                l_tot = l_tot + smem_xsum[128 + head];
                r_tot = r_tot + smem_xsum[512 + head];
                l_tot = l_tot + smem_xsum[192 + head];
                r_tot = r_tot + smem_xsum[576 + head];
                l_tot = l_tot + smem_xsum[256 + head];
                r_tot = r_tot + smem_xsum[640 + head];
                l_tot = l_tot + smem_xsum[320 + head];
                r_tot = r_tot + smem_xsum[704 + head];
                float norm_h = 0.0f;
                if (r_tot > 0.0f) {
                    float _rcp_8 = approx_rcp(r_tot);
                    norm_h = _rcp_8 * output_scale;
                }
                smem_norm[head] = norm_h;
                if (o_chunk == 0 && head_base + head < num_heads) {
                    int lse_offset = (query_idx * num_heads + head_base + head) * num_splits + split_idx;
                    float _log2_0;
                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(l_tot));
                    partial_lse[lse_offset] = ((l_tot > 0.0f) ? (sm[0] + _log2_0) * lse_partial_scale : -CAKE_INF);
                }
            }
            asm volatile("barrier.sync 10, 384;" ::: "memory");
            float norm_c[32];
            norm_c[0] = smem_norm[0];
            norm_c[1] = smem_norm[1];
            norm_c[2] = smem_norm[2];
            norm_c[3] = smem_norm[3];
            norm_c[4] = smem_norm[4];
            norm_c[5] = smem_norm[5];
            norm_c[6] = smem_norm[6];
            norm_c[7] = smem_norm[7];
            norm_c[8] = smem_norm[8];
            norm_c[9] = smem_norm[9];
            norm_c[10] = smem_norm[10];
            norm_c[11] = smem_norm[11];
            norm_c[12] = smem_norm[12];
            norm_c[13] = smem_norm[13];
            norm_c[14] = smem_norm[14];
            norm_c[15] = smem_norm[15];
            norm_c[16] = smem_norm[16];
            norm_c[17] = smem_norm[17];
            norm_c[18] = smem_norm[18];
            norm_c[19] = smem_norm[19];
            norm_c[20] = smem_norm[20];
            norm_c[21] = smem_norm[21];
            norm_c[22] = smem_norm[22];
            norm_c[23] = smem_norm[23];
            norm_c[24] = smem_norm[24];
            norm_c[25] = smem_norm[25];
            norm_c[26] = smem_norm[26];
            norm_c[27] = smem_norm[27];
            norm_c[28] = smem_norm[28];
            norm_c[29] = smem_norm[29];
            norm_c[30] = smem_norm[30];
            norm_c[31] = smem_norm[31];
            float o_values[32];
            float o_scaled[32];
            int is_odd = lane & 1;
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(o_values[0]), "=f"(o_values[1]), "=f"(o_values[2]), "=f"(o_values[3]), "=f"(o_values[4]), "=f"(o_values[5]), "=f"(o_values[6]), "=f"(o_values[7]), "=f"(o_values[8]), "=f"(o_values[9]), "=f"(o_values[10]), "=f"(o_values[11]), "=f"(o_values[12]), "=f"(o_values[13]), "=f"(o_values[14]), "=f"(o_values[15]), "=f"(o_values[16]), "=f"(o_values[17]), "=f"(o_values[18]), "=f"(o_values[19]), "=f"(o_values[20]), "=f"(o_values[21]), "=f"(o_values[22]), "=f"(o_values[23]), "=f"(o_values[24]), "=f"(o_values[25]), "=f"(o_values[26]), "=f"(o_values[27]), "=f"(o_values[28]), "=f"(o_values[29]), "=f"(o_values[30]), "=f"(o_values[31])
                : "r"(taddr + 128 + (unsigned int)(tmem_row_origin << 16)));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            o_scaled[0] = o_values[0] * norm_c[0];
            o_scaled[1] = o_values[1] * norm_c[1];
            o_scaled[2] = o_values[2] * norm_c[2];
            o_scaled[3] = o_values[3] * norm_c[3];
            o_scaled[4] = o_values[4] * norm_c[4];
            o_scaled[5] = o_values[5] * norm_c[5];
            o_scaled[6] = o_values[6] * norm_c[6];
            o_scaled[7] = o_values[7] * norm_c[7];
            o_scaled[8] = o_values[8] * norm_c[8];
            o_scaled[9] = o_values[9] * norm_c[9];
            o_scaled[10] = o_values[10] * norm_c[10];
            o_scaled[11] = o_values[11] * norm_c[11];
            o_scaled[12] = o_values[12] * norm_c[12];
            o_scaled[13] = o_values[13] * norm_c[13];
            o_scaled[14] = o_values[14] * norm_c[14];
            o_scaled[15] = o_values[15] * norm_c[15];
            o_scaled[16] = o_values[16] * norm_c[16];
            o_scaled[17] = o_values[17] * norm_c[17];
            o_scaled[18] = o_values[18] * norm_c[18];
            o_scaled[19] = o_values[19] * norm_c[19];
            o_scaled[20] = o_values[20] * norm_c[20];
            o_scaled[21] = o_values[21] * norm_c[21];
            o_scaled[22] = o_values[22] * norm_c[22];
            o_scaled[23] = o_values[23] * norm_c[23];
            o_scaled[24] = o_values[24] * norm_c[24];
            o_scaled[25] = o_values[25] * norm_c[25];
            o_scaled[26] = o_values[26] * norm_c[26];
            o_scaled[27] = o_values[27] * norm_c[27];
            o_scaled[28] = o_values[28] * norm_c[28];
            o_scaled[29] = o_values[29] * norm_c[29];
            o_scaled[30] = o_values[30] * norm_c[30];
            o_scaled[31] = o_values[31] * norm_c[31];
            uint32_t o_scaled_bf16[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_scaled[_lp*2 + 0], o_scaled[_lp*2+1 + 0]));
                o_scaled_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            int dim_own = row;
            smem_o16[dim_own] = (uint16_t)(o_scaled_bf16[0] & 65535);
            smem_o16[512 + dim_own] = (uint16_t)(o_scaled_bf16[0] >> 16);
            smem_o16[1024 + dim_own] = (uint16_t)(o_scaled_bf16[1] & 65535);
            smem_o16[1536 + dim_own] = (uint16_t)(o_scaled_bf16[1] >> 16);
            smem_o16[2048 + dim_own] = (uint16_t)(o_scaled_bf16[2] & 65535);
            smem_o16[2560 + dim_own] = (uint16_t)(o_scaled_bf16[2] >> 16);
            smem_o16[3072 + dim_own] = (uint16_t)(o_scaled_bf16[3] & 65535);
            smem_o16[3584 + dim_own] = (uint16_t)(o_scaled_bf16[3] >> 16);
            smem_o16[4096 + dim_own] = (uint16_t)(o_scaled_bf16[4] & 65535);
            smem_o16[4608 + dim_own] = (uint16_t)(o_scaled_bf16[4] >> 16);
            smem_o16[5120 + dim_own] = (uint16_t)(o_scaled_bf16[5] & 65535);
            smem_o16[5632 + dim_own] = (uint16_t)(o_scaled_bf16[5] >> 16);
            smem_o16[6144 + dim_own] = (uint16_t)(o_scaled_bf16[6] & 65535);
            smem_o16[6656 + dim_own] = (uint16_t)(o_scaled_bf16[6] >> 16);
            smem_o16[7168 + dim_own] = (uint16_t)(o_scaled_bf16[7] & 65535);
            smem_o16[7680 + dim_own] = (uint16_t)(o_scaled_bf16[7] >> 16);
            smem_o16[8192 + dim_own] = (uint16_t)(o_scaled_bf16[8] & 65535);
            smem_o16[8704 + dim_own] = (uint16_t)(o_scaled_bf16[8] >> 16);
            smem_o16[9216 + dim_own] = (uint16_t)(o_scaled_bf16[9] & 65535);
            smem_o16[9728 + dim_own] = (uint16_t)(o_scaled_bf16[9] >> 16);
            smem_o16[10240 + dim_own] = (uint16_t)(o_scaled_bf16[10] & 65535);
            smem_o16[10752 + dim_own] = (uint16_t)(o_scaled_bf16[10] >> 16);
            smem_o16[11264 + dim_own] = (uint16_t)(o_scaled_bf16[11] & 65535);
            smem_o16[11776 + dim_own] = (uint16_t)(o_scaled_bf16[11] >> 16);
            smem_o16[12288 + dim_own] = (uint16_t)(o_scaled_bf16[12] & 65535);
            smem_o16[12800 + dim_own] = (uint16_t)(o_scaled_bf16[12] >> 16);
            smem_o16[13312 + dim_own] = (uint16_t)(o_scaled_bf16[13] & 65535);
            smem_o16[13824 + dim_own] = (uint16_t)(o_scaled_bf16[13] >> 16);
            smem_o16[14336 + dim_own] = (uint16_t)(o_scaled_bf16[14] & 65535);
            smem_o16[14848 + dim_own] = (uint16_t)(o_scaled_bf16[14] >> 16);
            smem_o16[15360 + dim_own] = (uint16_t)(o_scaled_bf16[15] & 65535);
            smem_o16[15872 + dim_own] = (uint16_t)(o_scaled_bf16[15] >> 16);
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(o_values[0]), "=f"(o_values[1]), "=f"(o_values[2]), "=f"(o_values[3]), "=f"(o_values[4]), "=f"(o_values[5]), "=f"(o_values[6]), "=f"(o_values[7]), "=f"(o_values[8]), "=f"(o_values[9]), "=f"(o_values[10]), "=f"(o_values[11]), "=f"(o_values[12]), "=f"(o_values[13]), "=f"(o_values[14]), "=f"(o_values[15]), "=f"(o_values[16]), "=f"(o_values[17]), "=f"(o_values[18]), "=f"(o_values[19]), "=f"(o_values[20]), "=f"(o_values[21]), "=f"(o_values[22]), "=f"(o_values[23]), "=f"(o_values[24]), "=f"(o_values[25]), "=f"(o_values[26]), "=f"(o_values[27]), "=f"(o_values[28]), "=f"(o_values[29]), "=f"(o_values[30]), "=f"(o_values[31])
                : "r"(taddr + 192 + (unsigned int)(tmem_row_origin << 16)));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            o_scaled[0] = o_values[0] * norm_c[0];
            o_scaled[1] = o_values[1] * norm_c[1];
            o_scaled[2] = o_values[2] * norm_c[2];
            o_scaled[3] = o_values[3] * norm_c[3];
            o_scaled[4] = o_values[4] * norm_c[4];
            o_scaled[5] = o_values[5] * norm_c[5];
            o_scaled[6] = o_values[6] * norm_c[6];
            o_scaled[7] = o_values[7] * norm_c[7];
            o_scaled[8] = o_values[8] * norm_c[8];
            o_scaled[9] = o_values[9] * norm_c[9];
            o_scaled[10] = o_values[10] * norm_c[10];
            o_scaled[11] = o_values[11] * norm_c[11];
            o_scaled[12] = o_values[12] * norm_c[12];
            o_scaled[13] = o_values[13] * norm_c[13];
            o_scaled[14] = o_values[14] * norm_c[14];
            o_scaled[15] = o_values[15] * norm_c[15];
            o_scaled[16] = o_values[16] * norm_c[16];
            o_scaled[17] = o_values[17] * norm_c[17];
            o_scaled[18] = o_values[18] * norm_c[18];
            o_scaled[19] = o_values[19] * norm_c[19];
            o_scaled[20] = o_values[20] * norm_c[20];
            o_scaled[21] = o_values[21] * norm_c[21];
            o_scaled[22] = o_values[22] * norm_c[22];
            o_scaled[23] = o_values[23] * norm_c[23];
            o_scaled[24] = o_values[24] * norm_c[24];
            o_scaled[25] = o_values[25] * norm_c[25];
            o_scaled[26] = o_values[26] * norm_c[26];
            o_scaled[27] = o_values[27] * norm_c[27];
            o_scaled[28] = o_values[28] * norm_c[28];
            o_scaled[29] = o_values[29] * norm_c[29];
            o_scaled[30] = o_values[30] * norm_c[30];
            o_scaled[31] = o_values[31] * norm_c[31];
            uint32_t o_scaled_bf16_8[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_scaled[_lp*2 + 0], o_scaled[_lp*2+1 + 0]));
                o_scaled_bf16_8[_lp] = *(uint32_t*)&_bf2;
            }
            int dim_own_9 = 128 + row;
            smem_o16[dim_own_9] = (uint16_t)(o_scaled_bf16_8[0] & 65535);
            smem_o16[512 + dim_own_9] = (uint16_t)(o_scaled_bf16_8[0] >> 16);
            smem_o16[1024 + dim_own_9] = (uint16_t)(o_scaled_bf16_8[1] & 65535);
            smem_o16[1536 + dim_own_9] = (uint16_t)(o_scaled_bf16_8[1] >> 16);
            smem_o16[2048 + dim_own_9] = (uint16_t)(o_scaled_bf16_8[2] & 65535);
            smem_o16[2560 + dim_own_9] = (uint16_t)(o_scaled_bf16_8[2] >> 16);
            smem_o16[3072 + dim_own_9] = (uint16_t)(o_scaled_bf16_8[3] & 65535);
            smem_o16[3584 + dim_own_9] = (uint16_t)(o_scaled_bf16_8[3] >> 16);
            smem_o16[4096 + dim_own_9] = (uint16_t)(o_scaled_bf16_8[4] & 65535);
            smem_o16[4608 + dim_own_9] = (uint16_t)(o_scaled_bf16_8[4] >> 16);
            smem_o16[5120 + dim_own_9] = (uint16_t)(o_scaled_bf16_8[5] & 65535);
            smem_o16[5632 + dim_own_9] = (uint16_t)(o_scaled_bf16_8[5] >> 16);
            smem_o16[6144 + dim_own_9] = (uint16_t)(o_scaled_bf16_8[6] & 65535);
            smem_o16[6656 + dim_own_9] = (uint16_t)(o_scaled_bf16_8[6] >> 16);
            smem_o16[7168 + dim_own_9] = (uint16_t)(o_scaled_bf16_8[7] & 65535);
            smem_o16[7680 + dim_own_9] = (uint16_t)(o_scaled_bf16_8[7] >> 16);
            smem_o16[8192 + dim_own_9] = (uint16_t)(o_scaled_bf16_8[8] & 65535);
            smem_o16[8704 + dim_own_9] = (uint16_t)(o_scaled_bf16_8[8] >> 16);
            smem_o16[9216 + dim_own_9] = (uint16_t)(o_scaled_bf16_8[9] & 65535);
            smem_o16[9728 + dim_own_9] = (uint16_t)(o_scaled_bf16_8[9] >> 16);
            smem_o16[10240 + dim_own_9] = (uint16_t)(o_scaled_bf16_8[10] & 65535);
            smem_o16[10752 + dim_own_9] = (uint16_t)(o_scaled_bf16_8[10] >> 16);
            smem_o16[11264 + dim_own_9] = (uint16_t)(o_scaled_bf16_8[11] & 65535);
            smem_o16[11776 + dim_own_9] = (uint16_t)(o_scaled_bf16_8[11] >> 16);
            smem_o16[12288 + dim_own_9] = (uint16_t)(o_scaled_bf16_8[12] & 65535);
            smem_o16[12800 + dim_own_9] = (uint16_t)(o_scaled_bf16_8[12] >> 16);
            smem_o16[13312 + dim_own_9] = (uint16_t)(o_scaled_bf16_8[13] & 65535);
            smem_o16[13824 + dim_own_9] = (uint16_t)(o_scaled_bf16_8[13] >> 16);
            smem_o16[14336 + dim_own_9] = (uint16_t)(o_scaled_bf16_8[14] & 65535);
            smem_o16[14848 + dim_own_9] = (uint16_t)(o_scaled_bf16_8[14] >> 16);
            smem_o16[15360 + dim_own_9] = (uint16_t)(o_scaled_bf16_8[15] & 65535);
            smem_o16[15872 + dim_own_9] = (uint16_t)(o_scaled_bf16_8[15] >> 16);
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(o_values[0]), "=f"(o_values[1]), "=f"(o_values[2]), "=f"(o_values[3]), "=f"(o_values[4]), "=f"(o_values[5]), "=f"(o_values[6]), "=f"(o_values[7]), "=f"(o_values[8]), "=f"(o_values[9]), "=f"(o_values[10]), "=f"(o_values[11]), "=f"(o_values[12]), "=f"(o_values[13]), "=f"(o_values[14]), "=f"(o_values[15]), "=f"(o_values[16]), "=f"(o_values[17]), "=f"(o_values[18]), "=f"(o_values[19]), "=f"(o_values[20]), "=f"(o_values[21]), "=f"(o_values[22]), "=f"(o_values[23]), "=f"(o_values[24]), "=f"(o_values[25]), "=f"(o_values[26]), "=f"(o_values[27]), "=f"(o_values[28]), "=f"(o_values[29]), "=f"(o_values[30]), "=f"(o_values[31])
                : "r"(taddr + 256 + (unsigned int)(tmem_row_origin << 16)));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            o_scaled[0] = o_values[0] * norm_c[0];
            o_scaled[1] = o_values[1] * norm_c[1];
            o_scaled[2] = o_values[2] * norm_c[2];
            o_scaled[3] = o_values[3] * norm_c[3];
            o_scaled[4] = o_values[4] * norm_c[4];
            o_scaled[5] = o_values[5] * norm_c[5];
            o_scaled[6] = o_values[6] * norm_c[6];
            o_scaled[7] = o_values[7] * norm_c[7];
            o_scaled[8] = o_values[8] * norm_c[8];
            o_scaled[9] = o_values[9] * norm_c[9];
            o_scaled[10] = o_values[10] * norm_c[10];
            o_scaled[11] = o_values[11] * norm_c[11];
            o_scaled[12] = o_values[12] * norm_c[12];
            o_scaled[13] = o_values[13] * norm_c[13];
            o_scaled[14] = o_values[14] * norm_c[14];
            o_scaled[15] = o_values[15] * norm_c[15];
            o_scaled[16] = o_values[16] * norm_c[16];
            o_scaled[17] = o_values[17] * norm_c[17];
            o_scaled[18] = o_values[18] * norm_c[18];
            o_scaled[19] = o_values[19] * norm_c[19];
            o_scaled[20] = o_values[20] * norm_c[20];
            o_scaled[21] = o_values[21] * norm_c[21];
            o_scaled[22] = o_values[22] * norm_c[22];
            o_scaled[23] = o_values[23] * norm_c[23];
            o_scaled[24] = o_values[24] * norm_c[24];
            o_scaled[25] = o_values[25] * norm_c[25];
            o_scaled[26] = o_values[26] * norm_c[26];
            o_scaled[27] = o_values[27] * norm_c[27];
            o_scaled[28] = o_values[28] * norm_c[28];
            o_scaled[29] = o_values[29] * norm_c[29];
            o_scaled[30] = o_values[30] * norm_c[30];
            o_scaled[31] = o_values[31] * norm_c[31];
            uint32_t o_scaled_bf16_10[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_scaled[_lp*2 + 0], o_scaled[_lp*2+1 + 0]));
                o_scaled_bf16_10[_lp] = *(uint32_t*)&_bf2;
            }
            int dim_own_11 = 256 + row;
            smem_o16[dim_own_11] = (uint16_t)(o_scaled_bf16_10[0] & 65535);
            smem_o16[512 + dim_own_11] = (uint16_t)(o_scaled_bf16_10[0] >> 16);
            smem_o16[1024 + dim_own_11] = (uint16_t)(o_scaled_bf16_10[1] & 65535);
            smem_o16[1536 + dim_own_11] = (uint16_t)(o_scaled_bf16_10[1] >> 16);
            smem_o16[2048 + dim_own_11] = (uint16_t)(o_scaled_bf16_10[2] & 65535);
            smem_o16[2560 + dim_own_11] = (uint16_t)(o_scaled_bf16_10[2] >> 16);
            smem_o16[3072 + dim_own_11] = (uint16_t)(o_scaled_bf16_10[3] & 65535);
            smem_o16[3584 + dim_own_11] = (uint16_t)(o_scaled_bf16_10[3] >> 16);
            smem_o16[4096 + dim_own_11] = (uint16_t)(o_scaled_bf16_10[4] & 65535);
            smem_o16[4608 + dim_own_11] = (uint16_t)(o_scaled_bf16_10[4] >> 16);
            smem_o16[5120 + dim_own_11] = (uint16_t)(o_scaled_bf16_10[5] & 65535);
            smem_o16[5632 + dim_own_11] = (uint16_t)(o_scaled_bf16_10[5] >> 16);
            smem_o16[6144 + dim_own_11] = (uint16_t)(o_scaled_bf16_10[6] & 65535);
            smem_o16[6656 + dim_own_11] = (uint16_t)(o_scaled_bf16_10[6] >> 16);
            smem_o16[7168 + dim_own_11] = (uint16_t)(o_scaled_bf16_10[7] & 65535);
            smem_o16[7680 + dim_own_11] = (uint16_t)(o_scaled_bf16_10[7] >> 16);
            smem_o16[8192 + dim_own_11] = (uint16_t)(o_scaled_bf16_10[8] & 65535);
            smem_o16[8704 + dim_own_11] = (uint16_t)(o_scaled_bf16_10[8] >> 16);
            smem_o16[9216 + dim_own_11] = (uint16_t)(o_scaled_bf16_10[9] & 65535);
            smem_o16[9728 + dim_own_11] = (uint16_t)(o_scaled_bf16_10[9] >> 16);
            smem_o16[10240 + dim_own_11] = (uint16_t)(o_scaled_bf16_10[10] & 65535);
            smem_o16[10752 + dim_own_11] = (uint16_t)(o_scaled_bf16_10[10] >> 16);
            smem_o16[11264 + dim_own_11] = (uint16_t)(o_scaled_bf16_10[11] & 65535);
            smem_o16[11776 + dim_own_11] = (uint16_t)(o_scaled_bf16_10[11] >> 16);
            smem_o16[12288 + dim_own_11] = (uint16_t)(o_scaled_bf16_10[12] & 65535);
            smem_o16[12800 + dim_own_11] = (uint16_t)(o_scaled_bf16_10[12] >> 16);
            smem_o16[13312 + dim_own_11] = (uint16_t)(o_scaled_bf16_10[13] & 65535);
            smem_o16[13824 + dim_own_11] = (uint16_t)(o_scaled_bf16_10[13] >> 16);
            smem_o16[14336 + dim_own_11] = (uint16_t)(o_scaled_bf16_10[14] & 65535);
            smem_o16[14848 + dim_own_11] = (uint16_t)(o_scaled_bf16_10[14] >> 16);
            smem_o16[15360 + dim_own_11] = (uint16_t)(o_scaled_bf16_10[15] & 65535);
            smem_o16[15872 + dim_own_11] = (uint16_t)(o_scaled_bf16_10[15] >> 16);
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(o_values[0]), "=f"(o_values[1]), "=f"(o_values[2]), "=f"(o_values[3]), "=f"(o_values[4]), "=f"(o_values[5]), "=f"(o_values[6]), "=f"(o_values[7]), "=f"(o_values[8]), "=f"(o_values[9]), "=f"(o_values[10]), "=f"(o_values[11]), "=f"(o_values[12]), "=f"(o_values[13]), "=f"(o_values[14]), "=f"(o_values[15]), "=f"(o_values[16]), "=f"(o_values[17]), "=f"(o_values[18]), "=f"(o_values[19]), "=f"(o_values[20]), "=f"(o_values[21]), "=f"(o_values[22]), "=f"(o_values[23]), "=f"(o_values[24]), "=f"(o_values[25]), "=f"(o_values[26]), "=f"(o_values[27]), "=f"(o_values[28]), "=f"(o_values[29]), "=f"(o_values[30]), "=f"(o_values[31])
                : "r"(taddr + 320 + (unsigned int)(tmem_row_origin << 16)));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            o_scaled[0] = o_values[0] * norm_c[0];
            o_scaled[1] = o_values[1] * norm_c[1];
            o_scaled[2] = o_values[2] * norm_c[2];
            o_scaled[3] = o_values[3] * norm_c[3];
            o_scaled[4] = o_values[4] * norm_c[4];
            o_scaled[5] = o_values[5] * norm_c[5];
            o_scaled[6] = o_values[6] * norm_c[6];
            o_scaled[7] = o_values[7] * norm_c[7];
            o_scaled[8] = o_values[8] * norm_c[8];
            o_scaled[9] = o_values[9] * norm_c[9];
            o_scaled[10] = o_values[10] * norm_c[10];
            o_scaled[11] = o_values[11] * norm_c[11];
            o_scaled[12] = o_values[12] * norm_c[12];
            o_scaled[13] = o_values[13] * norm_c[13];
            o_scaled[14] = o_values[14] * norm_c[14];
            o_scaled[15] = o_values[15] * norm_c[15];
            o_scaled[16] = o_values[16] * norm_c[16];
            o_scaled[17] = o_values[17] * norm_c[17];
            o_scaled[18] = o_values[18] * norm_c[18];
            o_scaled[19] = o_values[19] * norm_c[19];
            o_scaled[20] = o_values[20] * norm_c[20];
            o_scaled[21] = o_values[21] * norm_c[21];
            o_scaled[22] = o_values[22] * norm_c[22];
            o_scaled[23] = o_values[23] * norm_c[23];
            o_scaled[24] = o_values[24] * norm_c[24];
            o_scaled[25] = o_values[25] * norm_c[25];
            o_scaled[26] = o_values[26] * norm_c[26];
            o_scaled[27] = o_values[27] * norm_c[27];
            o_scaled[28] = o_values[28] * norm_c[28];
            o_scaled[29] = o_values[29] * norm_c[29];
            o_scaled[30] = o_values[30] * norm_c[30];
            o_scaled[31] = o_values[31] * norm_c[31];
            uint32_t o_scaled_bf16_12[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_scaled[_lp*2 + 0], o_scaled[_lp*2+1 + 0]));
                o_scaled_bf16_12[_lp] = *(uint32_t*)&_bf2;
            }
            int dim_own_13 = 384 + row;
            smem_o16[dim_own_13] = (uint16_t)(o_scaled_bf16_12[0] & 65535);
            smem_o16[512 + dim_own_13] = (uint16_t)(o_scaled_bf16_12[0] >> 16);
            smem_o16[1024 + dim_own_13] = (uint16_t)(o_scaled_bf16_12[1] & 65535);
            smem_o16[1536 + dim_own_13] = (uint16_t)(o_scaled_bf16_12[1] >> 16);
            smem_o16[2048 + dim_own_13] = (uint16_t)(o_scaled_bf16_12[2] & 65535);
            smem_o16[2560 + dim_own_13] = (uint16_t)(o_scaled_bf16_12[2] >> 16);
            smem_o16[3072 + dim_own_13] = (uint16_t)(o_scaled_bf16_12[3] & 65535);
            smem_o16[3584 + dim_own_13] = (uint16_t)(o_scaled_bf16_12[3] >> 16);
            smem_o16[4096 + dim_own_13] = (uint16_t)(o_scaled_bf16_12[4] & 65535);
            smem_o16[4608 + dim_own_13] = (uint16_t)(o_scaled_bf16_12[4] >> 16);
            smem_o16[5120 + dim_own_13] = (uint16_t)(o_scaled_bf16_12[5] & 65535);
            smem_o16[5632 + dim_own_13] = (uint16_t)(o_scaled_bf16_12[5] >> 16);
            smem_o16[6144 + dim_own_13] = (uint16_t)(o_scaled_bf16_12[6] & 65535);
            smem_o16[6656 + dim_own_13] = (uint16_t)(o_scaled_bf16_12[6] >> 16);
            smem_o16[7168 + dim_own_13] = (uint16_t)(o_scaled_bf16_12[7] & 65535);
            smem_o16[7680 + dim_own_13] = (uint16_t)(o_scaled_bf16_12[7] >> 16);
            smem_o16[8192 + dim_own_13] = (uint16_t)(o_scaled_bf16_12[8] & 65535);
            smem_o16[8704 + dim_own_13] = (uint16_t)(o_scaled_bf16_12[8] >> 16);
            smem_o16[9216 + dim_own_13] = (uint16_t)(o_scaled_bf16_12[9] & 65535);
            smem_o16[9728 + dim_own_13] = (uint16_t)(o_scaled_bf16_12[9] >> 16);
            smem_o16[10240 + dim_own_13] = (uint16_t)(o_scaled_bf16_12[10] & 65535);
            smem_o16[10752 + dim_own_13] = (uint16_t)(o_scaled_bf16_12[10] >> 16);
            smem_o16[11264 + dim_own_13] = (uint16_t)(o_scaled_bf16_12[11] & 65535);
            smem_o16[11776 + dim_own_13] = (uint16_t)(o_scaled_bf16_12[11] >> 16);
            smem_o16[12288 + dim_own_13] = (uint16_t)(o_scaled_bf16_12[12] & 65535);
            smem_o16[12800 + dim_own_13] = (uint16_t)(o_scaled_bf16_12[12] >> 16);
            smem_o16[13312 + dim_own_13] = (uint16_t)(o_scaled_bf16_12[13] & 65535);
            smem_o16[13824 + dim_own_13] = (uint16_t)(o_scaled_bf16_12[13] >> 16);
            smem_o16[14336 + dim_own_13] = (uint16_t)(o_scaled_bf16_12[14] & 65535);
            smem_o16[14848 + dim_own_13] = (uint16_t)(o_scaled_bf16_12[14] >> 16);
            smem_o16[15360 + dim_own_13] = (uint16_t)(o_scaled_bf16_12[15] & 65535);
            smem_o16[15872 + dim_own_13] = (uint16_t)(o_scaled_bf16_12[15] >> 16);
            asm volatile("barrier.sync 10, 384;" ::: "memory");
            int tid384 = row;
            int ci = tid384;
            if (ci < 4096) {
                int h = ci / 64;
                int ch = ci - h * 64;
                if (head_base + h < num_heads) {
                    unsigned int w4[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 3]))
                        : "r"(smem_o32_addr + (unsigned int)(h * 1024) + (unsigned int)(ch * 16)));
                    long long out_off = ((long long)(query_idx * num_heads + head_base + h) * (long long)num_splits + (long long)split_idx) * 512 + (long long)(o_chunk * 512) + (long long)(ch * 8);
                    reinterpret_cast<int4*>(partial_O + out_off)[0] = reinterpret_cast<int4*>(w4)[0];
                }
            }
            int ci_14 = tid384 + 384;
            if (ci_14 < 4096) {
                int h_1 = ci_14 / 64;
                int ch_1 = ci_14 - h_1 * 64;
                if (head_base + h_1 < num_heads) {
                    unsigned int w4_1[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 3]))
                        : "r"(smem_o32_addr + (unsigned int)(h_1 * 1024) + (unsigned int)(ch_1 * 16)));
                    long long out_off_1 = ((long long)(query_idx * num_heads + head_base + h_1) * (long long)num_splits + (long long)split_idx) * 512 + (long long)(o_chunk * 512) + (long long)(ch_1 * 8);
                    reinterpret_cast<int4*>(partial_O + out_off_1)[0] = reinterpret_cast<int4*>(w4_1)[0];
                }
            }
            int ci_15 = tid384 + 768;
            if (ci_15 < 4096) {
                int h_2 = ci_15 / 64;
                int ch_2 = ci_15 - h_2 * 64;
                if (head_base + h_2 < num_heads) {
                    unsigned int w4_2[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_2[(0) + 3]))
                        : "r"(smem_o32_addr + (unsigned int)(h_2 * 1024) + (unsigned int)(ch_2 * 16)));
                    long long out_off_2 = ((long long)(query_idx * num_heads + head_base + h_2) * (long long)num_splits + (long long)split_idx) * 512 + (long long)(o_chunk * 512) + (long long)(ch_2 * 8);
                    reinterpret_cast<int4*>(partial_O + out_off_2)[0] = reinterpret_cast<int4*>(w4_2)[0];
                }
            }
            int ci_16 = tid384 + 1152;
            if (ci_16 < 4096) {
                int h_3 = ci_16 / 64;
                int ch_3 = ci_16 - h_3 * 64;
                if (head_base + h_3 < num_heads) {
                    unsigned int w4_3[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_3[(0) + 3]))
                        : "r"(smem_o32_addr + (unsigned int)(h_3 * 1024) + (unsigned int)(ch_3 * 16)));
                    long long out_off_3 = ((long long)(query_idx * num_heads + head_base + h_3) * (long long)num_splits + (long long)split_idx) * 512 + (long long)(o_chunk * 512) + (long long)(ch_3 * 8);
                    reinterpret_cast<int4*>(partial_O + out_off_3)[0] = reinterpret_cast<int4*>(w4_3)[0];
                }
            }
            int ci_17 = tid384 + 1536;
            if (ci_17 < 4096) {
                int h_4 = ci_17 / 64;
                int ch_4 = ci_17 - h_4 * 64;
                if (head_base + h_4 < num_heads) {
                    unsigned int w4_4[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4[(0) + 3]))
                        : "r"(smem_o32_addr + (unsigned int)(h_4 * 1024) + (unsigned int)(ch_4 * 16)));
                    long long out_off_4 = ((long long)(query_idx * num_heads + head_base + h_4) * (long long)num_splits + (long long)split_idx) * 512 + (long long)(o_chunk * 512) + (long long)(ch_4 * 8);
                    reinterpret_cast<int4*>(partial_O + out_off_4)[0] = reinterpret_cast<int4*>(w4_4)[0];
                }
            }
            int ci_18 = tid384 + 1920;
            if (ci_18 < 4096) {
                int h_5 = ci_18 / 64;
                int ch_5 = ci_18 - h_5 * 64;
                if (head_base + h_5 < num_heads) {
                    unsigned int w4_5[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_5[(0) + 3]))
                        : "r"(smem_o32_addr + (unsigned int)(h_5 * 1024) + (unsigned int)(ch_5 * 16)));
                    long long out_off_5 = ((long long)(query_idx * num_heads + head_base + h_5) * (long long)num_splits + (long long)split_idx) * 512 + (long long)(o_chunk * 512) + (long long)(ch_5 * 8);
                    reinterpret_cast<int4*>(partial_O + out_off_5)[0] = reinterpret_cast<int4*>(w4_5)[0];
                }
            }
            int ci_19 = tid384 + 2304;
            if (ci_19 < 4096) {
                int h_6 = ci_19 / 64;
                int ch_6 = ci_19 - h_6 * 64;
                if (head_base + h_6 < num_heads) {
                    unsigned int w4_6[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_6[(0) + 3]))
                        : "r"(smem_o32_addr + (unsigned int)(h_6 * 1024) + (unsigned int)(ch_6 * 16)));
                    long long out_off_6 = ((long long)(query_idx * num_heads + head_base + h_6) * (long long)num_splits + (long long)split_idx) * 512 + (long long)(o_chunk * 512) + (long long)(ch_6 * 8);
                    reinterpret_cast<int4*>(partial_O + out_off_6)[0] = reinterpret_cast<int4*>(w4_6)[0];
                }
            }
            int ci_20 = tid384 + 2688;
            if (ci_20 < 4096) {
                int h_7 = ci_20 / 64;
                int ch_7 = ci_20 - h_7 * 64;
                if (head_base + h_7 < num_heads) {
                    unsigned int w4_7[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_7[(0) + 3]))
                        : "r"(smem_o32_addr + (unsigned int)(h_7 * 1024) + (unsigned int)(ch_7 * 16)));
                    long long out_off_7 = ((long long)(query_idx * num_heads + head_base + h_7) * (long long)num_splits + (long long)split_idx) * 512 + (long long)(o_chunk * 512) + (long long)(ch_7 * 8);
                    reinterpret_cast<int4*>(partial_O + out_off_7)[0] = reinterpret_cast<int4*>(w4_7)[0];
                }
            }
            int ci_21 = tid384 + 3072;
            if (ci_21 < 4096) {
                int h_8 = ci_21 / 64;
                int ch_8 = ci_21 - h_8 * 64;
                if (head_base + h_8 < num_heads) {
                    unsigned int w4_8[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_8[(0) + 3]))
                        : "r"(smem_o32_addr + (unsigned int)(h_8 * 1024) + (unsigned int)(ch_8 * 16)));
                    long long out_off_8 = ((long long)(query_idx * num_heads + head_base + h_8) * (long long)num_splits + (long long)split_idx) * 512 + (long long)(o_chunk * 512) + (long long)(ch_8 * 8);
                    reinterpret_cast<int4*>(partial_O + out_off_8)[0] = reinterpret_cast<int4*>(w4_8)[0];
                }
            }
            int ci_22 = tid384 + 3456;
            if (ci_22 < 4096) {
                int h_9 = ci_22 / 64;
                int ch_9 = ci_22 - h_9 * 64;
                if (head_base + h_9 < num_heads) {
                    unsigned int w4_9[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_9[(0) + 3]))
                        : "r"(smem_o32_addr + (unsigned int)(h_9 * 1024) + (unsigned int)(ch_9 * 16)));
                    long long out_off_9 = ((long long)(query_idx * num_heads + head_base + h_9) * (long long)num_splits + (long long)split_idx) * 512 + (long long)(o_chunk * 512) + (long long)(ch_9 * 8);
                    reinterpret_cast<int4*>(partial_O + out_off_9)[0] = reinterpret_cast<int4*>(w4_9)[0];
                }
            }
            int ci_23 = tid384 + 3840;
            if (ci_23 < 4096) {
                int h_10 = ci_23 / 64;
                int ch_10 = ci_23 - h_10 * 64;
                if (head_base + h_10 < num_heads) {
                    unsigned int w4_10[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_10[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_10[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_10[(0) + 3]))
                        : "r"(smem_o32_addr + (unsigned int)(h_10 * 1024) + (unsigned int)(ch_10 * 16)));
                    long long out_off_10 = ((long long)(query_idx * num_heads + head_base + h_10) * (long long)num_splits + (long long)split_idx) * 512 + (long long)(o_chunk * 512) + (long long)(ch_10 * 8);
                    reinterpret_cast<int4*>(partial_O + out_off_10)[0] = reinterpret_cast<int4*>(w4_10)[0];
                }
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: compute1 ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 144;");
        { // compute1_main
            const int local_warp_1 = warp - 4;
            const int half_1 = local_warp_1 / 2;
            int o_chunk_1 = 0;
            int work_idx_1 = blockIdx.x;
            int head_tile_1 = work_idx_1 % num_head_tiles;
            int split_work_1 = work_idx_1 / num_head_tiles;
            int split_idx_1 = split_work_1 % num_splits;
            int query_idx_1 = split_work_1 / num_splits;
            int head_base_1 = head_tile_1 * 64;
            const int row_1 = local_warp_1 * 32 + lane;
            const int head_1 = row_1 & 63;
            const int tmem_row_origin_1 = local_warp_1 * 32;
            float inv_six_1 = 0.16666666666666666f;
            unsigned int _phase_q_nope_full0_0_1 = 0;
            mbarrier_wait_hint(q_nope_full0_addr, _phase_q_nope_full0_0_1, 10000000);
            _phase_q_nope_full0_0_1 ^= 1;
            unsigned int _phase_q_nope_full1_0_1 = 0;
            mbarrier_wait_hint(q_nope_full1_addr, _phase_q_nope_full1_0_1, 10000000);
            _phase_q_nope_full1_0_1 ^= 1;
            unsigned int _phase_q_nope_full2_0_1 = 0;
            mbarrier_wait_hint(q_nope_full2_addr, _phase_q_nope_full2_0_1, 10000000);
            _phase_q_nope_full2_0_1 ^= 1;
            int q_taddr_1 = taddr + 448 + (unsigned int)(tmem_row_origin_1 << 16);
            const int q_par_1 = local_warp_1 % 2;
            const int q_slot_1 = 2 + local_warp_1 / 2;
            int q_block_live_1 = ((head_base_1 + q_par_1 * 32 < num_heads) ? 1 : 0);
            int q_head_1 = q_par_1 * 32 + lane;
            int exch_lane_1 = smem_p_0_addr + (unsigned int)(lane * 32);
            unsigned int zero8_1[8];
            zero8_1[0] = 0;
            zero8_1[1] = 0;
            zero8_1[2] = 0;
            zero8_1[3] = 0;
            zero8_1[4] = 0;
            zero8_1[5] = 0;
            zero8_1[6] = 0;
            zero8_1[7] = 0;
            int kset_u_1 = ((1) ? q_slot_1 : 6);
            int do_u_2 = ((1) ? 1 : ((q_slot_1 == 0) ? 1 : 0));
            if (do_u_2 != 0) {
                int exch_u_2 = exch_lane_1 + (q_par_1 * 7 + kset_u_1) * 1024;
                if (q_block_live_1 != 0) {
                    int q_row_addr_2 = smem_qstage_addr + (unsigned int)(kset_u_1 * 8192) + (unsigned int)(q_head_1 * 128);
                    unsigned int words_2[8];
                    unsigned int sf_word_2 = 0;
                    unsigned int qa_2[4];
                    unsigned int qb_3[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qa_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_2[(0) + 3]))
                        : "r"(q_row_addr_2 + (0 ^ row_1 % 8) * 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qb_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_3[(0) + 3]))
                        : "r"(q_row_addr_2 + (1 ^ row_1 % 8) * 16));
                    float qv_3[16];
                    qv_3[0] = __uint_as_float(qa_2[0] << 16);
                    qv_3[1] = __uint_as_float(qa_2[0] & 4294901760u);
                    qv_3[8] = __uint_as_float(qb_3[0] << 16);
                    qv_3[9] = __uint_as_float(qb_3[0] & 4294901760u);
                    qv_3[2] = __uint_as_float(qa_2[1] << 16);
                    qv_3[3] = __uint_as_float(qa_2[1] & 4294901760u);
                    qv_3[10] = __uint_as_float(qb_3[1] << 16);
                    qv_3[11] = __uint_as_float(qb_3[1] & 4294901760u);
                    qv_3[4] = __uint_as_float(qa_2[2] << 16);
                    qv_3[5] = __uint_as_float(qa_2[2] & 4294901760u);
                    qv_3[12] = __uint_as_float(qb_3[2] << 16);
                    qv_3[13] = __uint_as_float(qb_3[2] & 4294901760u);
                    qv_3[6] = __uint_as_float(qa_2[3] << 16);
                    qv_3[7] = __uint_as_float(qa_2[3] & 4294901760u);
                    qv_3[14] = __uint_as_float(qb_3[3] << 16);
                    qv_3[15] = __uint_as_float(qb_3[3] & 4294901760u);
                    float m8_2[8];
                    float _fabs_128 = fabsf(qv_3[0]);
                    float _fabs_129 = fabsf(qv_3[1]);
                    float _max_158 = max_noftz(_fabs_128, _fabs_129);
                    m8_2[0] = _max_158;
                    float _fabs_130 = fabsf(qv_3[2]);
                    float _fabs_131 = fabsf(qv_3[3]);
                    float _max_159 = max_noftz(_fabs_130, _fabs_131);
                    m8_2[1] = _max_159;
                    float _fabs_132 = fabsf(qv_3[4]);
                    float _fabs_133 = fabsf(qv_3[5]);
                    float _max_160 = max_noftz(_fabs_132, _fabs_133);
                    m8_2[2] = _max_160;
                    float _fabs_134 = fabsf(qv_3[6]);
                    float _fabs_135 = fabsf(qv_3[7]);
                    float _max_161 = max_noftz(_fabs_134, _fabs_135);
                    m8_2[3] = _max_161;
                    float _fabs_136 = fabsf(qv_3[8]);
                    float _fabs_137 = fabsf(qv_3[9]);
                    float _max_162 = max_noftz(_fabs_136, _fabs_137);
                    m8_2[4] = _max_162;
                    float _fabs_138 = fabsf(qv_3[10]);
                    float _fabs_139 = fabsf(qv_3[11]);
                    float _max_163 = max_noftz(_fabs_138, _fabs_139);
                    m8_2[5] = _max_163;
                    float _fabs_140 = fabsf(qv_3[12]);
                    float _fabs_141 = fabsf(qv_3[13]);
                    float _max_164 = max_noftz(_fabs_140, _fabs_141);
                    m8_2[6] = _max_164;
                    float _fabs_142 = fabsf(qv_3[14]);
                    float _fabs_143 = fabsf(qv_3[15]);
                    float _max_165 = max_noftz(_fabs_142, _fabs_143);
                    m8_2[7] = _max_165;
                    float m4_2[4];
                    float _max_166 = max_noftz(m8_2[0], m8_2[1]);
                    m4_2[0] = _max_166;
                    float _max_167 = max_noftz(m8_2[2], m8_2[3]);
                    m4_2[1] = _max_167;
                    float _max_168 = max_noftz(m8_2[4], m8_2[5]);
                    m4_2[2] = _max_168;
                    float _max_169 = max_noftz(m8_2[6], m8_2[7]);
                    m4_2[3] = _max_169;
                    float _max_170 = max_noftz(m4_2[0], m4_2[1]);
                    float _max_171 = max_noftz(m4_2[2], m4_2[3]);
                    float _max_172 = max_noftz(_max_170, _max_171);
                    float amax_2 = _max_172;
                    float sc_2 = amax_2 * inv_six_1;
                    uint16_t _e4m3x2_f32_200;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_200) : "f"(0.0f), "f"(sc_2));
                    uint16_t sc_pair_2 = _e4m3x2_f32_200;
                    unsigned int sc_byte_2 = (unsigned int)sc_pair_2 & 255;
                    unsigned int sc_exp_2 = sc_byte_2 >> 3 & 15;
                    unsigned int sc_man_2 = sc_byte_2 & 7;
                    float sc_norm_2 = __uint_as_float(sc_exp_2 + 120 << 23 | sc_man_2 << 20);
                    float sc_sub_2 = (float)sc_man_2 * 0.001953125f;
                    float sc_dec_2 = ((sc_exp_2 == 0) ? sc_sub_2 : sc_norm_2);
                    float _rcp_9 = __frcp_rn(sc_dec_2);
                    float inv_2 = ((sc_dec_2 > 0.0f) ? _rcp_9 : 0.0f);
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_0 = {inv_2, inv_2};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_3)[_ls], _scale2_0);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        qv_3[_ls] = qv_3[_ls] * inv_2;
                    }
                    #endif
                    uint32_t _fp4_pair_64;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_64) : "f"(qv_3[0]), "f"(qv_3[1]));
                    uint32_t _fp4_pair_65;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_65) : "f"(qv_3[2]), "f"(qv_3[3]));
                    uint32_t _fp4_pair_66;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_66) : "f"(qv_3[4]), "f"(qv_3[5]));
                    uint32_t _fp4_pair_67;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_67) : "f"(qv_3[6]), "f"(qv_3[7]));
                    uint32_t _fp4_pair_68;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_68) : "f"(qv_3[8]), "f"(qv_3[9]));
                    uint32_t _fp4_pair_69;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_69) : "f"(qv_3[10]), "f"(qv_3[11]));
                    uint32_t _fp4_pair_70;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_70) : "f"(qv_3[12]), "f"(qv_3[13]));
                    uint32_t _fp4_pair_71;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_71) : "f"(qv_3[14]), "f"(qv_3[15]));
                    words_2[0] = _fp4_pair_64 | _fp4_pair_65 << 8 | _fp4_pair_66 << 16 | _fp4_pair_67 << 24;
                    words_2[1] = _fp4_pair_68 | _fp4_pair_69 << 8 | _fp4_pair_70 << 16 | _fp4_pair_71 << 24;
                    sf_word_2 = sf_word_2 | sc_byte_2;
                    unsigned int qa_0_2[4];
                    unsigned int qb_1_2[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qa_0_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0_2[(0) + 3]))
                        : "r"(q_row_addr_2 + (2 ^ row_1 % 8) * 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qb_1_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1_2[(0) + 3]))
                        : "r"(q_row_addr_2 + (3 ^ row_1 % 8) * 16));
                    float qv_2_2[16];
                    qv_2_2[0] = __uint_as_float(qa_0_2[0] << 16);
                    qv_2_2[1] = __uint_as_float(qa_0_2[0] & 4294901760u);
                    qv_2_2[8] = __uint_as_float(qb_1_2[0] << 16);
                    qv_2_2[9] = __uint_as_float(qb_1_2[0] & 4294901760u);
                    qv_2_2[2] = __uint_as_float(qa_0_2[1] << 16);
                    qv_2_2[3] = __uint_as_float(qa_0_2[1] & 4294901760u);
                    qv_2_2[10] = __uint_as_float(qb_1_2[1] << 16);
                    qv_2_2[11] = __uint_as_float(qb_1_2[1] & 4294901760u);
                    qv_2_2[4] = __uint_as_float(qa_0_2[2] << 16);
                    qv_2_2[5] = __uint_as_float(qa_0_2[2] & 4294901760u);
                    qv_2_2[12] = __uint_as_float(qb_1_2[2] << 16);
                    qv_2_2[13] = __uint_as_float(qb_1_2[2] & 4294901760u);
                    qv_2_2[6] = __uint_as_float(qa_0_2[3] << 16);
                    qv_2_2[7] = __uint_as_float(qa_0_2[3] & 4294901760u);
                    qv_2_2[14] = __uint_as_float(qb_1_2[3] << 16);
                    qv_2_2[15] = __uint_as_float(qb_1_2[3] & 4294901760u);
                    float m8_3_2[8];
                    float _fabs_144 = fabsf(qv_2_2[0]);
                    float _fabs_145 = fabsf(qv_2_2[1]);
                    float _max_173 = max_noftz(_fabs_144, _fabs_145);
                    m8_3_2[0] = _max_173;
                    float _fabs_146 = fabsf(qv_2_2[2]);
                    float _fabs_147 = fabsf(qv_2_2[3]);
                    float _max_174 = max_noftz(_fabs_146, _fabs_147);
                    m8_3_2[1] = _max_174;
                    float _fabs_148 = fabsf(qv_2_2[4]);
                    float _fabs_149 = fabsf(qv_2_2[5]);
                    float _max_175 = max_noftz(_fabs_148, _fabs_149);
                    m8_3_2[2] = _max_175;
                    float _fabs_150 = fabsf(qv_2_2[6]);
                    float _fabs_151 = fabsf(qv_2_2[7]);
                    float _max_176 = max_noftz(_fabs_150, _fabs_151);
                    m8_3_2[3] = _max_176;
                    float _fabs_152 = fabsf(qv_2_2[8]);
                    float _fabs_153 = fabsf(qv_2_2[9]);
                    float _max_177 = max_noftz(_fabs_152, _fabs_153);
                    m8_3_2[4] = _max_177;
                    float _fabs_154 = fabsf(qv_2_2[10]);
                    float _fabs_155 = fabsf(qv_2_2[11]);
                    float _max_178 = max_noftz(_fabs_154, _fabs_155);
                    m8_3_2[5] = _max_178;
                    float _fabs_156 = fabsf(qv_2_2[12]);
                    float _fabs_157 = fabsf(qv_2_2[13]);
                    float _max_179 = max_noftz(_fabs_156, _fabs_157);
                    m8_3_2[6] = _max_179;
                    float _fabs_158 = fabsf(qv_2_2[14]);
                    float _fabs_159 = fabsf(qv_2_2[15]);
                    float _max_180 = max_noftz(_fabs_158, _fabs_159);
                    m8_3_2[7] = _max_180;
                    float m4_4_2[4];
                    float _max_181 = max_noftz(m8_3_2[0], m8_3_2[1]);
                    m4_4_2[0] = _max_181;
                    float _max_182 = max_noftz(m8_3_2[2], m8_3_2[3]);
                    m4_4_2[1] = _max_182;
                    float _max_183 = max_noftz(m8_3_2[4], m8_3_2[5]);
                    m4_4_2[2] = _max_183;
                    float _max_184 = max_noftz(m8_3_2[6], m8_3_2[7]);
                    m4_4_2[3] = _max_184;
                    float _max_185 = max_noftz(m4_4_2[0], m4_4_2[1]);
                    float _max_186 = max_noftz(m4_4_2[2], m4_4_2[3]);
                    float _max_187 = max_noftz(_max_185, _max_186);
                    float amax_5_2 = _max_187;
                    float sc_6_2 = amax_5_2 * inv_six_1;
                    uint16_t _e4m3x2_f32_201;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_201) : "f"(0.0f), "f"(sc_6_2));
                    uint16_t sc_pair_7_2 = _e4m3x2_f32_201;
                    unsigned int sc_byte_8_2 = (unsigned int)sc_pair_7_2 & 255;
                    unsigned int sc_exp_9_2 = sc_byte_8_2 >> 3 & 15;
                    unsigned int sc_man_10_2 = sc_byte_8_2 & 7;
                    float sc_norm_11_2 = __uint_as_float(sc_exp_9_2 + 120 << 23 | sc_man_10_2 << 20);
                    float sc_sub_12_2 = (float)sc_man_10_2 * 0.001953125f;
                    float sc_dec_13_2 = ((sc_exp_9_2 == 0) ? sc_sub_12_2 : sc_norm_11_2);
                    float _rcp_10 = __frcp_rn(sc_dec_13_2);
                    float inv_14_2 = ((sc_dec_13_2 > 0.0f) ? _rcp_10 : 0.0f);
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_1 = {inv_14_2, inv_14_2};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_2_2)[_ls], _scale2_1);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        qv_2_2[_ls] = qv_2_2[_ls] * inv_14_2;
                    }
                    #endif
                    uint32_t _fp4_pair_72;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_72) : "f"(qv_2_2[0]), "f"(qv_2_2[1]));
                    uint32_t _fp4_pair_73;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_73) : "f"(qv_2_2[2]), "f"(qv_2_2[3]));
                    uint32_t _fp4_pair_74;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_74) : "f"(qv_2_2[4]), "f"(qv_2_2[5]));
                    uint32_t _fp4_pair_75;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_75) : "f"(qv_2_2[6]), "f"(qv_2_2[7]));
                    uint32_t _fp4_pair_76;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_76) : "f"(qv_2_2[8]), "f"(qv_2_2[9]));
                    uint32_t _fp4_pair_77;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_77) : "f"(qv_2_2[10]), "f"(qv_2_2[11]));
                    uint32_t _fp4_pair_78;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_78) : "f"(qv_2_2[12]), "f"(qv_2_2[13]));
                    uint32_t _fp4_pair_79;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_79) : "f"(qv_2_2[14]), "f"(qv_2_2[15]));
                    words_2[2] = _fp4_pair_72 | _fp4_pair_73 << 8 | _fp4_pair_74 << 16 | _fp4_pair_75 << 24;
                    words_2[3] = _fp4_pair_76 | _fp4_pair_77 << 8 | _fp4_pair_78 << 16 | _fp4_pair_79 << 24;
                    sf_word_2 = sf_word_2 | sc_byte_8_2 << 8;
                    unsigned int qa_15_2[4];
                    unsigned int qb_16_2[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qa_15_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_15_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_15_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_15_2[(0) + 3]))
                        : "r"(q_row_addr_2 + (4 ^ row_1 % 8) * 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qb_16_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_16_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_16_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_16_2[(0) + 3]))
                        : "r"(q_row_addr_2 + (5 ^ row_1 % 8) * 16));
                    float qv_17_2[16];
                    qv_17_2[0] = __uint_as_float(qa_15_2[0] << 16);
                    qv_17_2[1] = __uint_as_float(qa_15_2[0] & 4294901760u);
                    qv_17_2[8] = __uint_as_float(qb_16_2[0] << 16);
                    qv_17_2[9] = __uint_as_float(qb_16_2[0] & 4294901760u);
                    qv_17_2[2] = __uint_as_float(qa_15_2[1] << 16);
                    qv_17_2[3] = __uint_as_float(qa_15_2[1] & 4294901760u);
                    qv_17_2[10] = __uint_as_float(qb_16_2[1] << 16);
                    qv_17_2[11] = __uint_as_float(qb_16_2[1] & 4294901760u);
                    qv_17_2[4] = __uint_as_float(qa_15_2[2] << 16);
                    qv_17_2[5] = __uint_as_float(qa_15_2[2] & 4294901760u);
                    qv_17_2[12] = __uint_as_float(qb_16_2[2] << 16);
                    qv_17_2[13] = __uint_as_float(qb_16_2[2] & 4294901760u);
                    qv_17_2[6] = __uint_as_float(qa_15_2[3] << 16);
                    qv_17_2[7] = __uint_as_float(qa_15_2[3] & 4294901760u);
                    qv_17_2[14] = __uint_as_float(qb_16_2[3] << 16);
                    qv_17_2[15] = __uint_as_float(qb_16_2[3] & 4294901760u);
                    float m8_18_2[8];
                    float _fabs_160 = fabsf(qv_17_2[0]);
                    float _fabs_161 = fabsf(qv_17_2[1]);
                    float _max_188 = max_noftz(_fabs_160, _fabs_161);
                    m8_18_2[0] = _max_188;
                    float _fabs_162 = fabsf(qv_17_2[2]);
                    float _fabs_163 = fabsf(qv_17_2[3]);
                    float _max_189 = max_noftz(_fabs_162, _fabs_163);
                    m8_18_2[1] = _max_189;
                    float _fabs_164 = fabsf(qv_17_2[4]);
                    float _fabs_165 = fabsf(qv_17_2[5]);
                    float _max_190 = max_noftz(_fabs_164, _fabs_165);
                    m8_18_2[2] = _max_190;
                    float _fabs_166 = fabsf(qv_17_2[6]);
                    float _fabs_167 = fabsf(qv_17_2[7]);
                    float _max_191 = max_noftz(_fabs_166, _fabs_167);
                    m8_18_2[3] = _max_191;
                    float _fabs_168 = fabsf(qv_17_2[8]);
                    float _fabs_169 = fabsf(qv_17_2[9]);
                    float _max_192 = max_noftz(_fabs_168, _fabs_169);
                    m8_18_2[4] = _max_192;
                    float _fabs_170 = fabsf(qv_17_2[10]);
                    float _fabs_171 = fabsf(qv_17_2[11]);
                    float _max_193 = max_noftz(_fabs_170, _fabs_171);
                    m8_18_2[5] = _max_193;
                    float _fabs_172 = fabsf(qv_17_2[12]);
                    float _fabs_173 = fabsf(qv_17_2[13]);
                    float _max_194 = max_noftz(_fabs_172, _fabs_173);
                    m8_18_2[6] = _max_194;
                    float _fabs_174 = fabsf(qv_17_2[14]);
                    float _fabs_175 = fabsf(qv_17_2[15]);
                    float _max_195 = max_noftz(_fabs_174, _fabs_175);
                    m8_18_2[7] = _max_195;
                    float m4_19_2[4];
                    float _max_196 = max_noftz(m8_18_2[0], m8_18_2[1]);
                    m4_19_2[0] = _max_196;
                    float _max_197 = max_noftz(m8_18_2[2], m8_18_2[3]);
                    m4_19_2[1] = _max_197;
                    float _max_198 = max_noftz(m8_18_2[4], m8_18_2[5]);
                    m4_19_2[2] = _max_198;
                    float _max_199 = max_noftz(m8_18_2[6], m8_18_2[7]);
                    m4_19_2[3] = _max_199;
                    float _max_200 = max_noftz(m4_19_2[0], m4_19_2[1]);
                    float _max_201 = max_noftz(m4_19_2[2], m4_19_2[3]);
                    float _max_202 = max_noftz(_max_200, _max_201);
                    float amax_20_2 = _max_202;
                    float sc_21_2 = amax_20_2 * inv_six_1;
                    uint16_t _e4m3x2_f32_202;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_202) : "f"(0.0f), "f"(sc_21_2));
                    uint16_t sc_pair_22_2 = _e4m3x2_f32_202;
                    unsigned int sc_byte_23_2 = (unsigned int)sc_pair_22_2 & 255;
                    unsigned int sc_exp_24_2 = sc_byte_23_2 >> 3 & 15;
                    unsigned int sc_man_25_2 = sc_byte_23_2 & 7;
                    float sc_norm_26_2 = __uint_as_float(sc_exp_24_2 + 120 << 23 | sc_man_25_2 << 20);
                    float sc_sub_27_2 = (float)sc_man_25_2 * 0.001953125f;
                    float sc_dec_28_2 = ((sc_exp_24_2 == 0) ? sc_sub_27_2 : sc_norm_26_2);
                    float _rcp_11 = __frcp_rn(sc_dec_28_2);
                    float inv_29_2 = ((sc_dec_28_2 > 0.0f) ? _rcp_11 : 0.0f);
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_2 = {inv_29_2, inv_29_2};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_17_2)[_ls], _scale2_2);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        qv_17_2[_ls] = qv_17_2[_ls] * inv_29_2;
                    }
                    #endif
                    uint32_t _fp4_pair_80;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_80) : "f"(qv_17_2[0]), "f"(qv_17_2[1]));
                    uint32_t _fp4_pair_81;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_81) : "f"(qv_17_2[2]), "f"(qv_17_2[3]));
                    uint32_t _fp4_pair_82;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_82) : "f"(qv_17_2[4]), "f"(qv_17_2[5]));
                    uint32_t _fp4_pair_83;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_83) : "f"(qv_17_2[6]), "f"(qv_17_2[7]));
                    uint32_t _fp4_pair_84;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_84) : "f"(qv_17_2[8]), "f"(qv_17_2[9]));
                    uint32_t _fp4_pair_85;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_85) : "f"(qv_17_2[10]), "f"(qv_17_2[11]));
                    uint32_t _fp4_pair_86;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_86) : "f"(qv_17_2[12]), "f"(qv_17_2[13]));
                    uint32_t _fp4_pair_87;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_87) : "f"(qv_17_2[14]), "f"(qv_17_2[15]));
                    words_2[4] = _fp4_pair_80 | _fp4_pair_81 << 8 | _fp4_pair_82 << 16 | _fp4_pair_83 << 24;
                    words_2[5] = _fp4_pair_84 | _fp4_pair_85 << 8 | _fp4_pair_86 << 16 | _fp4_pair_87 << 24;
                    sf_word_2 = sf_word_2 | sc_byte_23_2 << 16;
                    unsigned int qa_30_2[4];
                    unsigned int qb_31_2[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qa_30_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_30_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_30_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_30_2[(0) + 3]))
                        : "r"(q_row_addr_2 + (6 ^ row_1 % 8) * 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qb_31_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_31_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_31_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_31_2[(0) + 3]))
                        : "r"(q_row_addr_2 + (7 ^ row_1 % 8) * 16));
                    float qv_32_2[16];
                    qv_32_2[0] = __uint_as_float(qa_30_2[0] << 16);
                    qv_32_2[1] = __uint_as_float(qa_30_2[0] & 4294901760u);
                    qv_32_2[8] = __uint_as_float(qb_31_2[0] << 16);
                    qv_32_2[9] = __uint_as_float(qb_31_2[0] & 4294901760u);
                    qv_32_2[2] = __uint_as_float(qa_30_2[1] << 16);
                    qv_32_2[3] = __uint_as_float(qa_30_2[1] & 4294901760u);
                    qv_32_2[10] = __uint_as_float(qb_31_2[1] << 16);
                    qv_32_2[11] = __uint_as_float(qb_31_2[1] & 4294901760u);
                    qv_32_2[4] = __uint_as_float(qa_30_2[2] << 16);
                    qv_32_2[5] = __uint_as_float(qa_30_2[2] & 4294901760u);
                    qv_32_2[12] = __uint_as_float(qb_31_2[2] << 16);
                    qv_32_2[13] = __uint_as_float(qb_31_2[2] & 4294901760u);
                    qv_32_2[6] = __uint_as_float(qa_30_2[3] << 16);
                    qv_32_2[7] = __uint_as_float(qa_30_2[3] & 4294901760u);
                    qv_32_2[14] = __uint_as_float(qb_31_2[3] << 16);
                    qv_32_2[15] = __uint_as_float(qb_31_2[3] & 4294901760u);
                    float m8_33_2[8];
                    float _fabs_176 = fabsf(qv_32_2[0]);
                    float _fabs_177 = fabsf(qv_32_2[1]);
                    float _max_203 = max_noftz(_fabs_176, _fabs_177);
                    m8_33_2[0] = _max_203;
                    float _fabs_178 = fabsf(qv_32_2[2]);
                    float _fabs_179 = fabsf(qv_32_2[3]);
                    float _max_204 = max_noftz(_fabs_178, _fabs_179);
                    m8_33_2[1] = _max_204;
                    float _fabs_180 = fabsf(qv_32_2[4]);
                    float _fabs_181 = fabsf(qv_32_2[5]);
                    float _max_205 = max_noftz(_fabs_180, _fabs_181);
                    m8_33_2[2] = _max_205;
                    float _fabs_182 = fabsf(qv_32_2[6]);
                    float _fabs_183 = fabsf(qv_32_2[7]);
                    float _max_206 = max_noftz(_fabs_182, _fabs_183);
                    m8_33_2[3] = _max_206;
                    float _fabs_184 = fabsf(qv_32_2[8]);
                    float _fabs_185 = fabsf(qv_32_2[9]);
                    float _max_207 = max_noftz(_fabs_184, _fabs_185);
                    m8_33_2[4] = _max_207;
                    float _fabs_186 = fabsf(qv_32_2[10]);
                    float _fabs_187 = fabsf(qv_32_2[11]);
                    float _max_208 = max_noftz(_fabs_186, _fabs_187);
                    m8_33_2[5] = _max_208;
                    float _fabs_188 = fabsf(qv_32_2[12]);
                    float _fabs_189 = fabsf(qv_32_2[13]);
                    float _max_209 = max_noftz(_fabs_188, _fabs_189);
                    m8_33_2[6] = _max_209;
                    float _fabs_190 = fabsf(qv_32_2[14]);
                    float _fabs_191 = fabsf(qv_32_2[15]);
                    float _max_210 = max_noftz(_fabs_190, _fabs_191);
                    m8_33_2[7] = _max_210;
                    float m4_34_2[4];
                    float _max_211 = max_noftz(m8_33_2[0], m8_33_2[1]);
                    m4_34_2[0] = _max_211;
                    float _max_212 = max_noftz(m8_33_2[2], m8_33_2[3]);
                    m4_34_2[1] = _max_212;
                    float _max_213 = max_noftz(m8_33_2[4], m8_33_2[5]);
                    m4_34_2[2] = _max_213;
                    float _max_214 = max_noftz(m8_33_2[6], m8_33_2[7]);
                    m4_34_2[3] = _max_214;
                    float _max_215 = max_noftz(m4_34_2[0], m4_34_2[1]);
                    float _max_216 = max_noftz(m4_34_2[2], m4_34_2[3]);
                    float _max_217 = max_noftz(_max_215, _max_216);
                    float amax_35_2 = _max_217;
                    float sc_36_2 = amax_35_2 * inv_six_1;
                    uint16_t _e4m3x2_f32_203;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_203) : "f"(0.0f), "f"(sc_36_2));
                    uint16_t sc_pair_37_2 = _e4m3x2_f32_203;
                    unsigned int sc_byte_38_2 = (unsigned int)sc_pair_37_2 & 255;
                    unsigned int sc_exp_39_2 = sc_byte_38_2 >> 3 & 15;
                    unsigned int sc_man_40_2 = sc_byte_38_2 & 7;
                    float sc_norm_41_2 = __uint_as_float(sc_exp_39_2 + 120 << 23 | sc_man_40_2 << 20);
                    float sc_sub_42_2 = (float)sc_man_40_2 * 0.001953125f;
                    float sc_dec_43_2 = ((sc_exp_39_2 == 0) ? sc_sub_42_2 : sc_norm_41_2);
                    float _rcp_12 = __frcp_rn(sc_dec_43_2);
                    float inv_44_2 = ((sc_dec_43_2 > 0.0f) ? _rcp_12 : 0.0f);
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_3 = {inv_44_2, inv_44_2};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_32_2)[_ls], _scale2_3);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        qv_32_2[_ls] = qv_32_2[_ls] * inv_44_2;
                    }
                    #endif
                    uint32_t _fp4_pair_88;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_88) : "f"(qv_32_2[0]), "f"(qv_32_2[1]));
                    uint32_t _fp4_pair_89;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_89) : "f"(qv_32_2[2]), "f"(qv_32_2[3]));
                    uint32_t _fp4_pair_90;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_90) : "f"(qv_32_2[4]), "f"(qv_32_2[5]));
                    uint32_t _fp4_pair_91;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_91) : "f"(qv_32_2[6]), "f"(qv_32_2[7]));
                    uint32_t _fp4_pair_92;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_92) : "f"(qv_32_2[8]), "f"(qv_32_2[9]));
                    uint32_t _fp4_pair_93;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_93) : "f"(qv_32_2[10]), "f"(qv_32_2[11]));
                    uint32_t _fp4_pair_94;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_94) : "f"(qv_32_2[12]), "f"(qv_32_2[13]));
                    uint32_t _fp4_pair_95;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_95) : "f"(qv_32_2[14]), "f"(qv_32_2[15]));
                    words_2[6] = _fp4_pair_88 | _fp4_pair_89 << 8 | _fp4_pair_90 << 16 | _fp4_pair_91 << 24;
                    words_2[7] = _fp4_pair_92 | _fp4_pair_93 << 8 | _fp4_pair_94 << 16 | _fp4_pair_95 << 24;
                    sf_word_2 = sf_word_2 | sc_byte_38_2 << 24;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(exch_u_2), "r"(*reinterpret_cast<uint32_t*>(&words_2[0])), "r"(*reinterpret_cast<uint32_t*>(&words_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words_2[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(exch_u_2 + 16), "r"(*reinterpret_cast<uint32_t*>(&words_2[4])), "r"(*reinterpret_cast<uint32_t*>(&words_2[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words_2[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words_2[(4) + 3])));
                    smem_qsf32[kset_u_1 / 4 * 2048 + row_1 % 32 / 8 * 512 + kset_u_1 % 4 * 128 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = sf_word_2;
                    smem_qsf32[kset_u_1 / 4 * 2048 + (row_1 ^ 64) % 32 / 8 * 512 + kset_u_1 % 4 * 128 + (row_1 ^ 64) % 8 * 16 + (row_1 ^ 64) / 32 % 4 * 4 >> 2] = sf_word_2;
                } else {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(exch_u_2), "r"(*reinterpret_cast<uint32_t*>(&zero8_1[0])), "r"(*reinterpret_cast<uint32_t*>(&zero8_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero8_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero8_1[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(exch_u_2 + 16), "r"(*reinterpret_cast<uint32_t*>(&zero8_1[4])), "r"(*reinterpret_cast<uint32_t*>(&zero8_1[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero8_1[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero8_1[(4) + 3])));
                    smem_qsf32[kset_u_1 / 4 * 2048 + row_1 % 32 / 8 * 512 + kset_u_1 % 4 * 128 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = 0;
                    smem_qsf32[kset_u_1 / 4 * 2048 + (row_1 ^ 64) % 32 / 8 * 512 + kset_u_1 % 4 * 128 + (row_1 ^ 64) % 8 * 16 + (row_1 ^ 64) / 32 % 4 * 4 >> 2] = 0;
                }
            }
            int kset_u_0_1 = ((0) ? q_slot_1 : 6);
            int do_u_1_1 = ((0) ? 1 : ((q_slot_1 == 0) ? 1 : 0));
            if (do_u_1_1 != 0) {
                int exch_u_3 = exch_lane_1 + (q_par_1 * 7 + kset_u_0_1) * 1024;
                if (q_block_live_1 != 0) {
                    int q_row_addr_3 = smem_qstage_addr + (unsigned int)(kset_u_0_1 * 8192) + (unsigned int)(q_head_1 * 128);
                    unsigned int words_3[8];
                    unsigned int sf_word_3 = 0;
                    unsigned int qa_3[4];
                    unsigned int qb_4[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qa_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_3[(0) + 3]))
                        : "r"(q_row_addr_3 + (0 ^ row_1 % 8) * 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qb_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_4[(0) + 3]))
                        : "r"(q_row_addr_3 + (1 ^ row_1 % 8) * 16));
                    float qv_4[16];
                    qv_4[0] = __uint_as_float(qa_3[0] << 16);
                    qv_4[1] = __uint_as_float(qa_3[0] & 4294901760u);
                    qv_4[8] = __uint_as_float(qb_4[0] << 16);
                    qv_4[9] = __uint_as_float(qb_4[0] & 4294901760u);
                    qv_4[2] = __uint_as_float(qa_3[1] << 16);
                    qv_4[3] = __uint_as_float(qa_3[1] & 4294901760u);
                    qv_4[10] = __uint_as_float(qb_4[1] << 16);
                    qv_4[11] = __uint_as_float(qb_4[1] & 4294901760u);
                    qv_4[4] = __uint_as_float(qa_3[2] << 16);
                    qv_4[5] = __uint_as_float(qa_3[2] & 4294901760u);
                    qv_4[12] = __uint_as_float(qb_4[2] << 16);
                    qv_4[13] = __uint_as_float(qb_4[2] & 4294901760u);
                    qv_4[6] = __uint_as_float(qa_3[3] << 16);
                    qv_4[7] = __uint_as_float(qa_3[3] & 4294901760u);
                    qv_4[14] = __uint_as_float(qb_4[3] << 16);
                    qv_4[15] = __uint_as_float(qb_4[3] & 4294901760u);
                    float m8_4[8];
                    float _fabs_192 = fabsf(qv_4[0]);
                    float _fabs_193 = fabsf(qv_4[1]);
                    float _max_218 = max_noftz(_fabs_192, _fabs_193);
                    m8_4[0] = _max_218;
                    float _fabs_194 = fabsf(qv_4[2]);
                    float _fabs_195 = fabsf(qv_4[3]);
                    float _max_219 = max_noftz(_fabs_194, _fabs_195);
                    m8_4[1] = _max_219;
                    float _fabs_196 = fabsf(qv_4[4]);
                    float _fabs_197 = fabsf(qv_4[5]);
                    float _max_220 = max_noftz(_fabs_196, _fabs_197);
                    m8_4[2] = _max_220;
                    float _fabs_198 = fabsf(qv_4[6]);
                    float _fabs_199 = fabsf(qv_4[7]);
                    float _max_221 = max_noftz(_fabs_198, _fabs_199);
                    m8_4[3] = _max_221;
                    float _fabs_200 = fabsf(qv_4[8]);
                    float _fabs_201 = fabsf(qv_4[9]);
                    float _max_222 = max_noftz(_fabs_200, _fabs_201);
                    m8_4[4] = _max_222;
                    float _fabs_202 = fabsf(qv_4[10]);
                    float _fabs_203 = fabsf(qv_4[11]);
                    float _max_223 = max_noftz(_fabs_202, _fabs_203);
                    m8_4[5] = _max_223;
                    float _fabs_204 = fabsf(qv_4[12]);
                    float _fabs_205 = fabsf(qv_4[13]);
                    float _max_224 = max_noftz(_fabs_204, _fabs_205);
                    m8_4[6] = _max_224;
                    float _fabs_206 = fabsf(qv_4[14]);
                    float _fabs_207 = fabsf(qv_4[15]);
                    float _max_225 = max_noftz(_fabs_206, _fabs_207);
                    m8_4[7] = _max_225;
                    float m4_3[4];
                    float _max_226 = max_noftz(m8_4[0], m8_4[1]);
                    m4_3[0] = _max_226;
                    float _max_227 = max_noftz(m8_4[2], m8_4[3]);
                    m4_3[1] = _max_227;
                    float _max_228 = max_noftz(m8_4[4], m8_4[5]);
                    m4_3[2] = _max_228;
                    float _max_229 = max_noftz(m8_4[6], m8_4[7]);
                    m4_3[3] = _max_229;
                    float _max_230 = max_noftz(m4_3[0], m4_3[1]);
                    float _max_231 = max_noftz(m4_3[2], m4_3[3]);
                    float _max_232 = max_noftz(_max_230, _max_231);
                    float amax_3 = _max_232;
                    float sc_3 = amax_3 * inv_six_1;
                    uint16_t _e4m3x2_f32_204;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_204) : "f"(0.0f), "f"(sc_3));
                    uint16_t sc_pair_3 = _e4m3x2_f32_204;
                    unsigned int sc_byte_3 = (unsigned int)sc_pair_3 & 255;
                    unsigned int sc_exp_3 = sc_byte_3 >> 3 & 15;
                    unsigned int sc_man_3 = sc_byte_3 & 7;
                    float sc_norm_3 = __uint_as_float(sc_exp_3 + 120 << 23 | sc_man_3 << 20);
                    float sc_sub_3 = (float)sc_man_3 * 0.001953125f;
                    float sc_dec_3 = ((sc_exp_3 == 0) ? sc_sub_3 : sc_norm_3);
                    float _rcp_13 = __frcp_rn(sc_dec_3);
                    float inv_3 = ((sc_dec_3 > 0.0f) ? _rcp_13 : 0.0f);
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_4 = {inv_3, inv_3};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_4)[_ls], _scale2_4);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        qv_4[_ls] = qv_4[_ls] * inv_3;
                    }
                    #endif
                    uint32_t _fp4_pair_96;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_96) : "f"(qv_4[0]), "f"(qv_4[1]));
                    uint32_t _fp4_pair_97;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_97) : "f"(qv_4[2]), "f"(qv_4[3]));
                    uint32_t _fp4_pair_98;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_98) : "f"(qv_4[4]), "f"(qv_4[5]));
                    uint32_t _fp4_pair_99;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_99) : "f"(qv_4[6]), "f"(qv_4[7]));
                    uint32_t _fp4_pair_100;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_100) : "f"(qv_4[8]), "f"(qv_4[9]));
                    uint32_t _fp4_pair_101;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_101) : "f"(qv_4[10]), "f"(qv_4[11]));
                    uint32_t _fp4_pair_102;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_102) : "f"(qv_4[12]), "f"(qv_4[13]));
                    uint32_t _fp4_pair_103;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_103) : "f"(qv_4[14]), "f"(qv_4[15]));
                    words_3[0] = _fp4_pair_96 | _fp4_pair_97 << 8 | _fp4_pair_98 << 16 | _fp4_pair_99 << 24;
                    words_3[1] = _fp4_pair_100 | _fp4_pair_101 << 8 | _fp4_pair_102 << 16 | _fp4_pair_103 << 24;
                    sf_word_3 = sf_word_3 | sc_byte_3;
                    unsigned int qa_0_3[4];
                    unsigned int qb_1_3[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qa_0_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0_3[(0) + 3]))
                        : "r"(q_row_addr_3 + (2 ^ row_1 % 8) * 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qb_1_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1_3[(0) + 3]))
                        : "r"(q_row_addr_3 + (3 ^ row_1 % 8) * 16));
                    float qv_2_3[16];
                    qv_2_3[0] = __uint_as_float(qa_0_3[0] << 16);
                    qv_2_3[1] = __uint_as_float(qa_0_3[0] & 4294901760u);
                    qv_2_3[8] = __uint_as_float(qb_1_3[0] << 16);
                    qv_2_3[9] = __uint_as_float(qb_1_3[0] & 4294901760u);
                    qv_2_3[2] = __uint_as_float(qa_0_3[1] << 16);
                    qv_2_3[3] = __uint_as_float(qa_0_3[1] & 4294901760u);
                    qv_2_3[10] = __uint_as_float(qb_1_3[1] << 16);
                    qv_2_3[11] = __uint_as_float(qb_1_3[1] & 4294901760u);
                    qv_2_3[4] = __uint_as_float(qa_0_3[2] << 16);
                    qv_2_3[5] = __uint_as_float(qa_0_3[2] & 4294901760u);
                    qv_2_3[12] = __uint_as_float(qb_1_3[2] << 16);
                    qv_2_3[13] = __uint_as_float(qb_1_3[2] & 4294901760u);
                    qv_2_3[6] = __uint_as_float(qa_0_3[3] << 16);
                    qv_2_3[7] = __uint_as_float(qa_0_3[3] & 4294901760u);
                    qv_2_3[14] = __uint_as_float(qb_1_3[3] << 16);
                    qv_2_3[15] = __uint_as_float(qb_1_3[3] & 4294901760u);
                    float m8_3_3[8];
                    float _fabs_208 = fabsf(qv_2_3[0]);
                    float _fabs_209 = fabsf(qv_2_3[1]);
                    float _max_233 = max_noftz(_fabs_208, _fabs_209);
                    m8_3_3[0] = _max_233;
                    float _fabs_210 = fabsf(qv_2_3[2]);
                    float _fabs_211 = fabsf(qv_2_3[3]);
                    float _max_234 = max_noftz(_fabs_210, _fabs_211);
                    m8_3_3[1] = _max_234;
                    float _fabs_212 = fabsf(qv_2_3[4]);
                    float _fabs_213 = fabsf(qv_2_3[5]);
                    float _max_235 = max_noftz(_fabs_212, _fabs_213);
                    m8_3_3[2] = _max_235;
                    float _fabs_214 = fabsf(qv_2_3[6]);
                    float _fabs_215 = fabsf(qv_2_3[7]);
                    float _max_236 = max_noftz(_fabs_214, _fabs_215);
                    m8_3_3[3] = _max_236;
                    float _fabs_216 = fabsf(qv_2_3[8]);
                    float _fabs_217 = fabsf(qv_2_3[9]);
                    float _max_237 = max_noftz(_fabs_216, _fabs_217);
                    m8_3_3[4] = _max_237;
                    float _fabs_218 = fabsf(qv_2_3[10]);
                    float _fabs_219 = fabsf(qv_2_3[11]);
                    float _max_238 = max_noftz(_fabs_218, _fabs_219);
                    m8_3_3[5] = _max_238;
                    float _fabs_220 = fabsf(qv_2_3[12]);
                    float _fabs_221 = fabsf(qv_2_3[13]);
                    float _max_239 = max_noftz(_fabs_220, _fabs_221);
                    m8_3_3[6] = _max_239;
                    float _fabs_222 = fabsf(qv_2_3[14]);
                    float _fabs_223 = fabsf(qv_2_3[15]);
                    float _max_240 = max_noftz(_fabs_222, _fabs_223);
                    m8_3_3[7] = _max_240;
                    float m4_4_3[4];
                    float _max_241 = max_noftz(m8_3_3[0], m8_3_3[1]);
                    m4_4_3[0] = _max_241;
                    float _max_242 = max_noftz(m8_3_3[2], m8_3_3[3]);
                    m4_4_3[1] = _max_242;
                    float _max_243 = max_noftz(m8_3_3[4], m8_3_3[5]);
                    m4_4_3[2] = _max_243;
                    float _max_244 = max_noftz(m8_3_3[6], m8_3_3[7]);
                    m4_4_3[3] = _max_244;
                    float _max_245 = max_noftz(m4_4_3[0], m4_4_3[1]);
                    float _max_246 = max_noftz(m4_4_3[2], m4_4_3[3]);
                    float _max_247 = max_noftz(_max_245, _max_246);
                    float amax_5_3 = _max_247;
                    float sc_6_3 = amax_5_3 * inv_six_1;
                    uint16_t _e4m3x2_f32_205;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_205) : "f"(0.0f), "f"(sc_6_3));
                    uint16_t sc_pair_7_3 = _e4m3x2_f32_205;
                    unsigned int sc_byte_8_3 = (unsigned int)sc_pair_7_3 & 255;
                    unsigned int sc_exp_9_3 = sc_byte_8_3 >> 3 & 15;
                    unsigned int sc_man_10_3 = sc_byte_8_3 & 7;
                    float sc_norm_11_3 = __uint_as_float(sc_exp_9_3 + 120 << 23 | sc_man_10_3 << 20);
                    float sc_sub_12_3 = (float)sc_man_10_3 * 0.001953125f;
                    float sc_dec_13_3 = ((sc_exp_9_3 == 0) ? sc_sub_12_3 : sc_norm_11_3);
                    float _rcp_14 = __frcp_rn(sc_dec_13_3);
                    float inv_14_3 = ((sc_dec_13_3 > 0.0f) ? _rcp_14 : 0.0f);
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_5 = {inv_14_3, inv_14_3};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_2_3)[_ls], _scale2_5);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        qv_2_3[_ls] = qv_2_3[_ls] * inv_14_3;
                    }
                    #endif
                    uint32_t _fp4_pair_104;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_104) : "f"(qv_2_3[0]), "f"(qv_2_3[1]));
                    uint32_t _fp4_pair_105;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_105) : "f"(qv_2_3[2]), "f"(qv_2_3[3]));
                    uint32_t _fp4_pair_106;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_106) : "f"(qv_2_3[4]), "f"(qv_2_3[5]));
                    uint32_t _fp4_pair_107;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_107) : "f"(qv_2_3[6]), "f"(qv_2_3[7]));
                    uint32_t _fp4_pair_108;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_108) : "f"(qv_2_3[8]), "f"(qv_2_3[9]));
                    uint32_t _fp4_pair_109;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_109) : "f"(qv_2_3[10]), "f"(qv_2_3[11]));
                    uint32_t _fp4_pair_110;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_110) : "f"(qv_2_3[12]), "f"(qv_2_3[13]));
                    uint32_t _fp4_pair_111;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_111) : "f"(qv_2_3[14]), "f"(qv_2_3[15]));
                    words_3[2] = _fp4_pair_104 | _fp4_pair_105 << 8 | _fp4_pair_106 << 16 | _fp4_pair_107 << 24;
                    words_3[3] = _fp4_pair_108 | _fp4_pair_109 << 8 | _fp4_pair_110 << 16 | _fp4_pair_111 << 24;
                    sf_word_3 = sf_word_3 | sc_byte_8_3 << 8;
                    unsigned int qa_15_3[4];
                    unsigned int qb_16_3[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qa_15_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_15_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_15_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_15_3[(0) + 3]))
                        : "r"(q_row_addr_3 + (4 ^ row_1 % 8) * 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qb_16_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_16_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_16_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_16_3[(0) + 3]))
                        : "r"(q_row_addr_3 + (5 ^ row_1 % 8) * 16));
                    float qv_17_3[16];
                    qv_17_3[0] = __uint_as_float(qa_15_3[0] << 16);
                    qv_17_3[1] = __uint_as_float(qa_15_3[0] & 4294901760u);
                    qv_17_3[8] = __uint_as_float(qb_16_3[0] << 16);
                    qv_17_3[9] = __uint_as_float(qb_16_3[0] & 4294901760u);
                    qv_17_3[2] = __uint_as_float(qa_15_3[1] << 16);
                    qv_17_3[3] = __uint_as_float(qa_15_3[1] & 4294901760u);
                    qv_17_3[10] = __uint_as_float(qb_16_3[1] << 16);
                    qv_17_3[11] = __uint_as_float(qb_16_3[1] & 4294901760u);
                    qv_17_3[4] = __uint_as_float(qa_15_3[2] << 16);
                    qv_17_3[5] = __uint_as_float(qa_15_3[2] & 4294901760u);
                    qv_17_3[12] = __uint_as_float(qb_16_3[2] << 16);
                    qv_17_3[13] = __uint_as_float(qb_16_3[2] & 4294901760u);
                    qv_17_3[6] = __uint_as_float(qa_15_3[3] << 16);
                    qv_17_3[7] = __uint_as_float(qa_15_3[3] & 4294901760u);
                    qv_17_3[14] = __uint_as_float(qb_16_3[3] << 16);
                    qv_17_3[15] = __uint_as_float(qb_16_3[3] & 4294901760u);
                    float m8_18_3[8];
                    float _fabs_224 = fabsf(qv_17_3[0]);
                    float _fabs_225 = fabsf(qv_17_3[1]);
                    float _max_248 = max_noftz(_fabs_224, _fabs_225);
                    m8_18_3[0] = _max_248;
                    float _fabs_226 = fabsf(qv_17_3[2]);
                    float _fabs_227 = fabsf(qv_17_3[3]);
                    float _max_249 = max_noftz(_fabs_226, _fabs_227);
                    m8_18_3[1] = _max_249;
                    float _fabs_228 = fabsf(qv_17_3[4]);
                    float _fabs_229 = fabsf(qv_17_3[5]);
                    float _max_250 = max_noftz(_fabs_228, _fabs_229);
                    m8_18_3[2] = _max_250;
                    float _fabs_230 = fabsf(qv_17_3[6]);
                    float _fabs_231 = fabsf(qv_17_3[7]);
                    float _max_251 = max_noftz(_fabs_230, _fabs_231);
                    m8_18_3[3] = _max_251;
                    float _fabs_232 = fabsf(qv_17_3[8]);
                    float _fabs_233 = fabsf(qv_17_3[9]);
                    float _max_252 = max_noftz(_fabs_232, _fabs_233);
                    m8_18_3[4] = _max_252;
                    float _fabs_234 = fabsf(qv_17_3[10]);
                    float _fabs_235 = fabsf(qv_17_3[11]);
                    float _max_253 = max_noftz(_fabs_234, _fabs_235);
                    m8_18_3[5] = _max_253;
                    float _fabs_236 = fabsf(qv_17_3[12]);
                    float _fabs_237 = fabsf(qv_17_3[13]);
                    float _max_254 = max_noftz(_fabs_236, _fabs_237);
                    m8_18_3[6] = _max_254;
                    float _fabs_238 = fabsf(qv_17_3[14]);
                    float _fabs_239 = fabsf(qv_17_3[15]);
                    float _max_255 = max_noftz(_fabs_238, _fabs_239);
                    m8_18_3[7] = _max_255;
                    float m4_19_3[4];
                    float _max_256 = max_noftz(m8_18_3[0], m8_18_3[1]);
                    m4_19_3[0] = _max_256;
                    float _max_257 = max_noftz(m8_18_3[2], m8_18_3[3]);
                    m4_19_3[1] = _max_257;
                    float _max_258 = max_noftz(m8_18_3[4], m8_18_3[5]);
                    m4_19_3[2] = _max_258;
                    float _max_259 = max_noftz(m8_18_3[6], m8_18_3[7]);
                    m4_19_3[3] = _max_259;
                    float _max_260 = max_noftz(m4_19_3[0], m4_19_3[1]);
                    float _max_261 = max_noftz(m4_19_3[2], m4_19_3[3]);
                    float _max_262 = max_noftz(_max_260, _max_261);
                    float amax_20_3 = _max_262;
                    float sc_21_3 = amax_20_3 * inv_six_1;
                    uint16_t _e4m3x2_f32_206;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_206) : "f"(0.0f), "f"(sc_21_3));
                    uint16_t sc_pair_22_3 = _e4m3x2_f32_206;
                    unsigned int sc_byte_23_3 = (unsigned int)sc_pair_22_3 & 255;
                    unsigned int sc_exp_24_3 = sc_byte_23_3 >> 3 & 15;
                    unsigned int sc_man_25_3 = sc_byte_23_3 & 7;
                    float sc_norm_26_3 = __uint_as_float(sc_exp_24_3 + 120 << 23 | sc_man_25_3 << 20);
                    float sc_sub_27_3 = (float)sc_man_25_3 * 0.001953125f;
                    float sc_dec_28_3 = ((sc_exp_24_3 == 0) ? sc_sub_27_3 : sc_norm_26_3);
                    float _rcp_15 = __frcp_rn(sc_dec_28_3);
                    float inv_29_3 = ((sc_dec_28_3 > 0.0f) ? _rcp_15 : 0.0f);
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_6 = {inv_29_3, inv_29_3};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_17_3)[_ls], _scale2_6);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        qv_17_3[_ls] = qv_17_3[_ls] * inv_29_3;
                    }
                    #endif
                    uint32_t _fp4_pair_112;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_112) : "f"(qv_17_3[0]), "f"(qv_17_3[1]));
                    uint32_t _fp4_pair_113;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_113) : "f"(qv_17_3[2]), "f"(qv_17_3[3]));
                    uint32_t _fp4_pair_114;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_114) : "f"(qv_17_3[4]), "f"(qv_17_3[5]));
                    uint32_t _fp4_pair_115;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_115) : "f"(qv_17_3[6]), "f"(qv_17_3[7]));
                    uint32_t _fp4_pair_116;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_116) : "f"(qv_17_3[8]), "f"(qv_17_3[9]));
                    uint32_t _fp4_pair_117;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_117) : "f"(qv_17_3[10]), "f"(qv_17_3[11]));
                    uint32_t _fp4_pair_118;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_118) : "f"(qv_17_3[12]), "f"(qv_17_3[13]));
                    uint32_t _fp4_pair_119;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_119) : "f"(qv_17_3[14]), "f"(qv_17_3[15]));
                    words_3[4] = _fp4_pair_112 | _fp4_pair_113 << 8 | _fp4_pair_114 << 16 | _fp4_pair_115 << 24;
                    words_3[5] = _fp4_pair_116 | _fp4_pair_117 << 8 | _fp4_pair_118 << 16 | _fp4_pair_119 << 24;
                    sf_word_3 = sf_word_3 | sc_byte_23_3 << 16;
                    unsigned int qa_30_3[4];
                    unsigned int qb_31_3[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qa_30_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_30_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_30_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_30_3[(0) + 3]))
                        : "r"(q_row_addr_3 + (6 ^ row_1 % 8) * 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qb_31_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_31_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_31_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_31_3[(0) + 3]))
                        : "r"(q_row_addr_3 + (7 ^ row_1 % 8) * 16));
                    float qv_32_3[16];
                    qv_32_3[0] = __uint_as_float(qa_30_3[0] << 16);
                    qv_32_3[1] = __uint_as_float(qa_30_3[0] & 4294901760u);
                    qv_32_3[8] = __uint_as_float(qb_31_3[0] << 16);
                    qv_32_3[9] = __uint_as_float(qb_31_3[0] & 4294901760u);
                    qv_32_3[2] = __uint_as_float(qa_30_3[1] << 16);
                    qv_32_3[3] = __uint_as_float(qa_30_3[1] & 4294901760u);
                    qv_32_3[10] = __uint_as_float(qb_31_3[1] << 16);
                    qv_32_3[11] = __uint_as_float(qb_31_3[1] & 4294901760u);
                    qv_32_3[4] = __uint_as_float(qa_30_3[2] << 16);
                    qv_32_3[5] = __uint_as_float(qa_30_3[2] & 4294901760u);
                    qv_32_3[12] = __uint_as_float(qb_31_3[2] << 16);
                    qv_32_3[13] = __uint_as_float(qb_31_3[2] & 4294901760u);
                    qv_32_3[6] = __uint_as_float(qa_30_3[3] << 16);
                    qv_32_3[7] = __uint_as_float(qa_30_3[3] & 4294901760u);
                    qv_32_3[14] = __uint_as_float(qb_31_3[3] << 16);
                    qv_32_3[15] = __uint_as_float(qb_31_3[3] & 4294901760u);
                    float m8_33_3[8];
                    float _fabs_240 = fabsf(qv_32_3[0]);
                    float _fabs_241 = fabsf(qv_32_3[1]);
                    float _max_263 = max_noftz(_fabs_240, _fabs_241);
                    m8_33_3[0] = _max_263;
                    float _fabs_242 = fabsf(qv_32_3[2]);
                    float _fabs_243 = fabsf(qv_32_3[3]);
                    float _max_264 = max_noftz(_fabs_242, _fabs_243);
                    m8_33_3[1] = _max_264;
                    float _fabs_244 = fabsf(qv_32_3[4]);
                    float _fabs_245 = fabsf(qv_32_3[5]);
                    float _max_265 = max_noftz(_fabs_244, _fabs_245);
                    m8_33_3[2] = _max_265;
                    float _fabs_246 = fabsf(qv_32_3[6]);
                    float _fabs_247 = fabsf(qv_32_3[7]);
                    float _max_266 = max_noftz(_fabs_246, _fabs_247);
                    m8_33_3[3] = _max_266;
                    float _fabs_248 = fabsf(qv_32_3[8]);
                    float _fabs_249 = fabsf(qv_32_3[9]);
                    float _max_267 = max_noftz(_fabs_248, _fabs_249);
                    m8_33_3[4] = _max_267;
                    float _fabs_250 = fabsf(qv_32_3[10]);
                    float _fabs_251 = fabsf(qv_32_3[11]);
                    float _max_268 = max_noftz(_fabs_250, _fabs_251);
                    m8_33_3[5] = _max_268;
                    float _fabs_252 = fabsf(qv_32_3[12]);
                    float _fabs_253 = fabsf(qv_32_3[13]);
                    float _max_269 = max_noftz(_fabs_252, _fabs_253);
                    m8_33_3[6] = _max_269;
                    float _fabs_254 = fabsf(qv_32_3[14]);
                    float _fabs_255 = fabsf(qv_32_3[15]);
                    float _max_270 = max_noftz(_fabs_254, _fabs_255);
                    m8_33_3[7] = _max_270;
                    float m4_34_3[4];
                    float _max_271 = max_noftz(m8_33_3[0], m8_33_3[1]);
                    m4_34_3[0] = _max_271;
                    float _max_272 = max_noftz(m8_33_3[2], m8_33_3[3]);
                    m4_34_3[1] = _max_272;
                    float _max_273 = max_noftz(m8_33_3[4], m8_33_3[5]);
                    m4_34_3[2] = _max_273;
                    float _max_274 = max_noftz(m8_33_3[6], m8_33_3[7]);
                    m4_34_3[3] = _max_274;
                    float _max_275 = max_noftz(m4_34_3[0], m4_34_3[1]);
                    float _max_276 = max_noftz(m4_34_3[2], m4_34_3[3]);
                    float _max_277 = max_noftz(_max_275, _max_276);
                    float amax_35_3 = _max_277;
                    float sc_36_3 = amax_35_3 * inv_six_1;
                    uint16_t _e4m3x2_f32_207;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_207) : "f"(0.0f), "f"(sc_36_3));
                    uint16_t sc_pair_37_3 = _e4m3x2_f32_207;
                    unsigned int sc_byte_38_3 = (unsigned int)sc_pair_37_3 & 255;
                    unsigned int sc_exp_39_3 = sc_byte_38_3 >> 3 & 15;
                    unsigned int sc_man_40_3 = sc_byte_38_3 & 7;
                    float sc_norm_41_3 = __uint_as_float(sc_exp_39_3 + 120 << 23 | sc_man_40_3 << 20);
                    float sc_sub_42_3 = (float)sc_man_40_3 * 0.001953125f;
                    float sc_dec_43_3 = ((sc_exp_39_3 == 0) ? sc_sub_42_3 : sc_norm_41_3);
                    float _rcp_16 = __frcp_rn(sc_dec_43_3);
                    float inv_44_3 = ((sc_dec_43_3 > 0.0f) ? _rcp_16 : 0.0f);
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_7 = {inv_44_3, inv_44_3};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_32_3)[_ls], _scale2_7);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        qv_32_3[_ls] = qv_32_3[_ls] * inv_44_3;
                    }
                    #endif
                    uint32_t _fp4_pair_120;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_120) : "f"(qv_32_3[0]), "f"(qv_32_3[1]));
                    uint32_t _fp4_pair_121;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_121) : "f"(qv_32_3[2]), "f"(qv_32_3[3]));
                    uint32_t _fp4_pair_122;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_122) : "f"(qv_32_3[4]), "f"(qv_32_3[5]));
                    uint32_t _fp4_pair_123;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_123) : "f"(qv_32_3[6]), "f"(qv_32_3[7]));
                    uint32_t _fp4_pair_124;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_124) : "f"(qv_32_3[8]), "f"(qv_32_3[9]));
                    uint32_t _fp4_pair_125;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_125) : "f"(qv_32_3[10]), "f"(qv_32_3[11]));
                    uint32_t _fp4_pair_126;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_126) : "f"(qv_32_3[12]), "f"(qv_32_3[13]));
                    uint32_t _fp4_pair_127;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_127) : "f"(qv_32_3[14]), "f"(qv_32_3[15]));
                    words_3[6] = _fp4_pair_120 | _fp4_pair_121 << 8 | _fp4_pair_122 << 16 | _fp4_pair_123 << 24;
                    words_3[7] = _fp4_pair_124 | _fp4_pair_125 << 8 | _fp4_pair_126 << 16 | _fp4_pair_127 << 24;
                    sf_word_3 = sf_word_3 | sc_byte_38_3 << 24;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(exch_u_3), "r"(*reinterpret_cast<uint32_t*>(&words_3[0])), "r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(exch_u_3 + 16), "r"(*reinterpret_cast<uint32_t*>(&words_3[4])), "r"(*reinterpret_cast<uint32_t*>(&words_3[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words_3[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words_3[(4) + 3])));
                    smem_qsf32[kset_u_0_1 / 4 * 2048 + row_1 % 32 / 8 * 512 + kset_u_0_1 % 4 * 128 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = sf_word_3;
                    smem_qsf32[kset_u_0_1 / 4 * 2048 + (row_1 ^ 64) % 32 / 8 * 512 + kset_u_0_1 % 4 * 128 + (row_1 ^ 64) % 8 * 16 + (row_1 ^ 64) / 32 % 4 * 4 >> 2] = sf_word_3;
                } else {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(exch_u_3), "r"(*reinterpret_cast<uint32_t*>(&zero8_1[0])), "r"(*reinterpret_cast<uint32_t*>(&zero8_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero8_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero8_1[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(exch_u_3 + 16), "r"(*reinterpret_cast<uint32_t*>(&zero8_1[4])), "r"(*reinterpret_cast<uint32_t*>(&zero8_1[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero8_1[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero8_1[(4) + 3])));
                    smem_qsf32[kset_u_0_1 / 4 * 2048 + row_1 % 32 / 8 * 512 + kset_u_0_1 % 4 * 128 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = 0;
                    smem_qsf32[kset_u_0_1 / 4 * 2048 + (row_1 ^ 64) % 32 / 8 * 512 + kset_u_0_1 % 4 * 128 + (row_1 ^ 64) % 8 * 16 + (row_1 ^ 64) / 32 % 4 * 4 >> 2] = 0;
                }
            }
            asm volatile("barrier.sync 10, 384;" ::: "memory");
            unsigned int qw_1[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&qw_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_1[(0) + 3]))
                : "r"(exch_lane_1 + q_par_1 * 7 * 1024));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&qw_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_1[(4) + 3]))
                : "r"(exch_lane_1 + q_par_1 * 7 * 1024 + 16));
            tmem_st_x8_u32(q_taddr_1, (const uint32_t*)qw_1);
            unsigned int qw_2_1[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_2_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&qw_2_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_2_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_2_1[(0) + 3]))
                : "r"(exch_lane_1 + (q_par_1 * 7 + 1) * 1024));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_2_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&qw_2_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_2_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_2_1[(4) + 3]))
                : "r"(exch_lane_1 + (q_par_1 * 7 + 1) * 1024 + 16));
            tmem_st_x8_u32(q_taddr_1 + 8, (const uint32_t*)qw_2_1);
            unsigned int qw_3_1[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_3_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&qw_3_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_3_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_3_1[(0) + 3]))
                : "r"(exch_lane_1 + (q_par_1 * 7 + 2) * 1024));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_3_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&qw_3_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_3_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_3_1[(4) + 3]))
                : "r"(exch_lane_1 + (q_par_1 * 7 + 2) * 1024 + 16));
            tmem_st_x8_u32(q_taddr_1 + 16, (const uint32_t*)qw_3_1);
            unsigned int qw_4_1[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&qw_4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_4_1[(0) + 3]))
                : "r"(exch_lane_1 + (q_par_1 * 7 + 3) * 1024));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_4_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&qw_4_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_4_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_4_1[(4) + 3]))
                : "r"(exch_lane_1 + (q_par_1 * 7 + 3) * 1024 + 16));
            tmem_st_x8_u32(q_taddr_1 + 24, (const uint32_t*)qw_4_1);
            unsigned int qw_5_1[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_5_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&qw_5_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_5_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_5_1[(0) + 3]))
                : "r"(exch_lane_1 + (q_par_1 * 7 + 4) * 1024));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_5_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&qw_5_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_5_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_5_1[(4) + 3]))
                : "r"(exch_lane_1 + (q_par_1 * 7 + 4) * 1024 + 16));
            tmem_st_x8_u32(q_taddr_1 + 32, (const uint32_t*)qw_5_1);
            unsigned int qw_6_1[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_6_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&qw_6_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_6_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_6_1[(0) + 3]))
                : "r"(exch_lane_1 + (q_par_1 * 7 + 5) * 1024));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_6_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&qw_6_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_6_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_6_1[(4) + 3]))
                : "r"(exch_lane_1 + (q_par_1 * 7 + 5) * 1024 + 16));
            tmem_st_x8_u32(q_taddr_1 + 40, (const uint32_t*)qw_6_1);
            unsigned int qw_7_1[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_7_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&qw_7_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_7_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_7_1[(0) + 3]))
                : "r"(exch_lane_1 + (q_par_1 * 7 + 6) * 1024));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_7_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&qw_7_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_7_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_7_1[(4) + 3]))
                : "r"(exch_lane_1 + (q_par_1 * 7 + 6) * 1024 + 16));
            tmem_st_x8_u32(q_taddr_1 + 48, (const uint32_t*)qw_7_1);
            tmem_st_x8_u32(q_taddr_1 + 56, (const uint32_t*)zero8_1);
            smem_qsf32[2048 + row_1 % 32 / 8 * 512 + 384 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = 0;
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(q_ready_addr);
            asm volatile("barrier.sync 9, 384;" ::: "memory");
            float softmax_scale_log2_1 = bmm1_scale[0] * 1.4426950408889634f;
            float output_scale_1 = bmm2_scale[0];
            float sm_1[3];
            sm_1[0] = -CAKE_INF;
            sm_1[1] = 0.0f;
            sm_1[2] = 0.0f;
            float sink_lane_1 = -CAKE_INF;
            if (has_sinks != 0 && split_idx_1 == 0 && head_base_1 + head_1 < num_heads) {
                sink_lane_1 = sinks[head_base_1 + head_1] * 1.4426950408889634f;
            }
            for (int it_1 = 0; it_1 < tiles_per_split; it_1++) {
                int buf_1 = it_1 & 1;
                int par_1 = it_1 >> 1 & 1;
                int kbase_1 = smem_kf4_0_addr + (unsigned int)(buf_1 * 53248);
                int kz_off_1 = buf_1 * 53248;
                mbarrier_wait_hint(tok_full_addr + (buf_1) * 8, par_1, 10000000);
                int tok_off_1 = buf_1 * 256;
                int pbase_1 = smem_p_0_addr + (unsigned int)(buf_1 * 8192);
                int raw_index_1 = smem_tok32v[tok_off_1 + row_1];
                unsigned int mask_word_1 = (unsigned int)smem_tok32v[tok_off_1 + ((half_1 == 0) ? 129 : 131)];
                int valid_1 = 1;
                if (raw_index_1 < 0) {
                    valid_1 = 0;
                }
                if (valid_1 != 0) {
                    {
                        unsigned int sfw_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&sfw_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&sfw_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sfw_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sfw_1[(0) + 3]))
                            : "r"(kbase_1 + 49152 + row_1 * 32 + 16));
                        smem_ksf32[2048 + row_1 % 32 / 8 * 512 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = sfw_1[0];
                        smem_ksf32[2048 + row_1 % 32 / 8 * 512 + 128 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = sfw_1[1];
                        smem_ksf32[2048 + row_1 % 32 / 8 * 512 + 256 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = sfw_1[2];
                        {
                            smem_ksf32[2048 + row_1 % 32 / 8 * 512 + 384 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = 0;
                        }
                    }
                } else if (1) {
                    smem_ksf32[2048 + row_1 % 32 / 8 * 512 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = 0;
                    smem_ksf32[2048 + row_1 % 32 / 8 * 512 + 128 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = 0;
                    smem_ksf32[2048 + row_1 % 32 / 8 * 512 + 256 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = 0;
                    smem_ksf32[2048 + row_1 % 32 / 8 * 512 + 384 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = 0;
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(kv_full_addr);
                if (it_1 > 0) {
                    mbarrier_wait_hint(o_full_addr, it_1 - 1 & 1, 10000000);
                    mbarrier_wait_hint(o_full_addr + 8, it_1 - 1 & 1, 10000000);
                    mbarrier_wait_hint(o_full_addr + 16, it_1 - 1 & 1, 10000000);
                    mbarrier_wait_hint(o_full_addr + 24, it_1 - 1 & 1, 10000000);
                }
                if (valid_1 != 0) {
                    unsigned int kraw4_1[4];
                    unsigned int sfw32_1 = 0;
                    int vblock_1 = 32 * o_chunk_1 + 11;
                    unsigned int v8_2[4];
                    {
                        {
                            int vchunk_6 = 16 * o_chunk_1 + 5;
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 3]))
                                : "r"(kbase_1 + vchunk_6 / 8 * 16384 + (row_1 * 128 + (vchunk_6 % 8 * 16 ^ row_1 % 8 * 16))));
                        }
                        {
                            sfw32_1 = smem_kz32[(kz_off_1 + 49152 + row_1 * 32 >> 2) + 8 * o_chunk_1 + 2];
                        }
                        unsigned int scale_11 = sfw32_1 >> 24 & 255;
                        {
                            v8_2[0] = cake_dsv4_qmul4<5>(kraw4_1[2], scale_11);
                        }
                        {
                            v8_2[1] = cake_dsv4_qmul4<6>(kraw4_1[2], scale_11);
                        }
                        {
                            v8_2[2] = cake_dsv4_qmul4<5>(kraw4_1[3], scale_11);
                        }
                        {
                            v8_2[3] = cake_dsv4_qmul4<6>(kraw4_1[3], scale_11);
                        }
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(smem_v_addr + (unsigned int)(16384 + (row_1 * 128 + (48 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_2[(0) + 3])));
                    int vblock_0_1 = 32 * o_chunk_1 + 12;
                    unsigned int v8_1_1[4];
                    {
                        {
                            int vchunk_7 = 16 * o_chunk_1 + 6;
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 3]))
                                : "r"(kbase_1 + vchunk_7 / 8 * 16384 + (row_1 * 128 + (vchunk_7 % 8 * 16 ^ row_1 % 8 * 16))));
                        }
                        {
                            sfw32_1 = smem_kz32[(kz_off_1 + 49152 + row_1 * 32 >> 2) + 8 * o_chunk_1 + 3];
                        }
                        unsigned int scale_12 = sfw32_1 & 255;
                        {
                            v8_1_1[0] = cake_dsv4_qmul4<5>(kraw4_1[0], scale_12);
                        }
                        {
                            v8_1_1[1] = cake_dsv4_qmul4<6>(kraw4_1[0], scale_12);
                        }
                        {
                            v8_1_1[2] = cake_dsv4_qmul4<5>(kraw4_1[1], scale_12);
                        }
                        {
                            v8_1_1[3] = cake_dsv4_qmul4<6>(kraw4_1[1], scale_12);
                        }
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(smem_v_addr + (unsigned int)(16384 + (row_1 * 128 + (64 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_1_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_1_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_1_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_1_1[(0) + 3])));
                    int vblock_2_1 = 32 * o_chunk_1 + 13;
                    unsigned int v8_3_1[4];
                    {
                        unsigned int scale_13 = sfw32_1 >> 8 & 255;
                        {
                            v8_3_1[0] = cake_dsv4_qmul4<5>(kraw4_1[2], scale_13);
                        }
                        {
                            v8_3_1[1] = cake_dsv4_qmul4<6>(kraw4_1[2], scale_13);
                        }
                        {
                            v8_3_1[2] = cake_dsv4_qmul4<5>(kraw4_1[3], scale_13);
                        }
                        {
                            v8_3_1[3] = cake_dsv4_qmul4<6>(kraw4_1[3], scale_13);
                        }
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(smem_v_addr + (unsigned int)(16384 + (row_1 * 128 + (80 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_3_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_3_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_3_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_3_1[(0) + 3])));
                    int vblock_4_1 = 32 * o_chunk_1 + 14;
                    unsigned int v8_5_1[4];
                    {
                        {
                            int vchunk_8 = 16 * o_chunk_1 + 7;
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 3]))
                                : "r"(kbase_1 + vchunk_8 / 8 * 16384 + (row_1 * 128 + (vchunk_8 % 8 * 16 ^ row_1 % 8 * 16))));
                        }
                        unsigned int scale_14 = sfw32_1 >> 16 & 255;
                        {
                            v8_5_1[0] = cake_dsv4_qmul4<5>(kraw4_1[0], scale_14);
                        }
                        {
                            v8_5_1[1] = cake_dsv4_qmul4<6>(kraw4_1[0], scale_14);
                        }
                        {
                            v8_5_1[2] = cake_dsv4_qmul4<5>(kraw4_1[1], scale_14);
                        }
                        {
                            v8_5_1[3] = cake_dsv4_qmul4<6>(kraw4_1[1], scale_14);
                        }
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(smem_v_addr + (unsigned int)(16384 + (row_1 * 128 + (96 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_5_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_5_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_5_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_5_1[(0) + 3])));
                    int vblock_6_1 = 32 * o_chunk_1 + 15;
                    unsigned int v8_7_1[4];
                    {
                        unsigned int scale_15 = sfw32_1 >> 24 & 255;
                        {
                            v8_7_1[0] = cake_dsv4_qmul4<5>(kraw4_1[2], scale_15);
                        }
                        {
                            v8_7_1[1] = cake_dsv4_qmul4<6>(kraw4_1[2], scale_15);
                        }
                        {
                            v8_7_1[2] = cake_dsv4_qmul4<5>(kraw4_1[3], scale_15);
                        }
                        {
                            v8_7_1[3] = cake_dsv4_qmul4<6>(kraw4_1[3], scale_15);
                        }
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(smem_v_addr + (unsigned int)(16384 + (row_1 * 128 + (112 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_7_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_7_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_7_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_7_1[(0) + 3])));
                    int vblock_8_1 = 32 * o_chunk_1 + 16;
                    unsigned int v8_9_1[4];
                    {
                        {
                            int vchunk_9 = 16 * o_chunk_1 + 8;
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 3]))
                                : "r"(kbase_1 + vchunk_9 / 8 * 16384 + (row_1 * 128 + (vchunk_9 % 8 * 16 ^ row_1 % 8 * 16))));
                        }
                        {
                            sfw32_1 = smem_kz32[(kz_off_1 + 49152 + row_1 * 32 >> 2) + 8 * o_chunk_1 + 4];
                        }
                        unsigned int scale_16 = sfw32_1 & 255;
                        {
                            v8_9_1[0] = cake_dsv4_qmul4<5>(kraw4_1[0], scale_16);
                        }
                        {
                            v8_9_1[1] = cake_dsv4_qmul4<6>(kraw4_1[0], scale_16);
                        }
                        {
                            v8_9_1[2] = cake_dsv4_qmul4<5>(kraw4_1[1], scale_16);
                        }
                        {
                            v8_9_1[3] = cake_dsv4_qmul4<6>(kraw4_1[1], scale_16);
                        }
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (0 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_9_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_9_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_9_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_9_1[(0) + 3])));
                    int vblock_10_1 = 32 * o_chunk_1 + 17;
                    unsigned int v8_11_1[4];
                    {
                        unsigned int scale_17 = sfw32_1 >> 8 & 255;
                        {
                            v8_11_1[0] = cake_dsv4_qmul4<5>(kraw4_1[2], scale_17);
                        }
                        {
                            v8_11_1[1] = cake_dsv4_qmul4<6>(kraw4_1[2], scale_17);
                        }
                        {
                            v8_11_1[2] = cake_dsv4_qmul4<5>(kraw4_1[3], scale_17);
                        }
                        {
                            v8_11_1[3] = cake_dsv4_qmul4<6>(kraw4_1[3], scale_17);
                        }
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (16 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_11_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_11_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_11_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_11_1[(0) + 3])));
                    int vblock_12_1 = 32 * o_chunk_1 + 18;
                    unsigned int v8_13_1[4];
                    {
                        {
                            int vchunk_10 = 16 * o_chunk_1 + 9;
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 3]))
                                : "r"(kbase_1 + vchunk_10 / 8 * 16384 + (row_1 * 128 + (vchunk_10 % 8 * 16 ^ row_1 % 8 * 16))));
                        }
                        unsigned int scale_18 = sfw32_1 >> 16 & 255;
                        {
                            v8_13_1[0] = cake_dsv4_qmul4<5>(kraw4_1[0], scale_18);
                        }
                        {
                            v8_13_1[1] = cake_dsv4_qmul4<6>(kraw4_1[0], scale_18);
                        }
                        {
                            v8_13_1[2] = cake_dsv4_qmul4<5>(kraw4_1[1], scale_18);
                        }
                        {
                            v8_13_1[3] = cake_dsv4_qmul4<6>(kraw4_1[1], scale_18);
                        }
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (32 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_13_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_13_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_13_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_13_1[(0) + 3])));
                    int vblock_14_1 = 32 * o_chunk_1 + 19;
                    unsigned int v8_15_1[4];
                    {
                        unsigned int scale_19 = sfw32_1 >> 24 & 255;
                        {
                            v8_15_1[0] = cake_dsv4_qmul4<5>(kraw4_1[2], scale_19);
                        }
                        {
                            v8_15_1[1] = cake_dsv4_qmul4<6>(kraw4_1[2], scale_19);
                        }
                        {
                            v8_15_1[2] = cake_dsv4_qmul4<5>(kraw4_1[3], scale_19);
                        }
                        {
                            v8_15_1[3] = cake_dsv4_qmul4<6>(kraw4_1[3], scale_19);
                        }
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (48 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_15_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_15_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_15_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_15_1[(0) + 3])));
                    int vblock_16_1 = 32 * o_chunk_1 + 20;
                    unsigned int v8_17_1[4];
                    {
                        {
                            int vchunk_11 = 16 * o_chunk_1 + 10;
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 3]))
                                : "r"(kbase_1 + vchunk_11 / 8 * 16384 + (row_1 * 128 + (vchunk_11 % 8 * 16 ^ row_1 % 8 * 16))));
                        }
                        {
                            sfw32_1 = smem_kz32[(kz_off_1 + 49152 + row_1 * 32 >> 2) + 8 * o_chunk_1 + 5];
                        }
                        unsigned int scale_20 = sfw32_1 & 255;
                        {
                            v8_17_1[0] = cake_dsv4_qmul4<5>(kraw4_1[0], scale_20);
                        }
                        {
                            v8_17_1[1] = cake_dsv4_qmul4<6>(kraw4_1[0], scale_20);
                        }
                        {
                            v8_17_1[2] = cake_dsv4_qmul4<5>(kraw4_1[1], scale_20);
                        }
                        {
                            v8_17_1[3] = cake_dsv4_qmul4<6>(kraw4_1[1], scale_20);
                        }
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (64 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_17_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_17_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_17_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_17_1[(0) + 3])));
                    int vblock_18_1 = 32 * o_chunk_1 + 21;
                    unsigned int v8_19_1[4];
                    {
                        unsigned int scale_21 = sfw32_1 >> 8 & 255;
                        {
                            v8_19_1[0] = cake_dsv4_qmul4<5>(kraw4_1[2], scale_21);
                        }
                        {
                            v8_19_1[1] = cake_dsv4_qmul4<6>(kraw4_1[2], scale_21);
                        }
                        {
                            v8_19_1[2] = cake_dsv4_qmul4<5>(kraw4_1[3], scale_21);
                        }
                        {
                            v8_19_1[3] = cake_dsv4_qmul4<6>(kraw4_1[3], scale_21);
                        }
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (80 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_19_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_19_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_19_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_19_1[(0) + 3])));
                } else {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(16384 + (row_1 * 128 + (48 ^ row_1 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(16384 + (row_1 * 128 + (64 ^ row_1 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(16384 + (row_1 * 128 + (80 ^ row_1 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(16384 + (row_1 * 128 + (96 ^ row_1 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(16384 + (row_1 * 128 + (112 ^ row_1 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (0 ^ row_1 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (16 ^ row_1 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (32 ^ row_1 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (48 ^ row_1 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (64 ^ row_1 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (80 ^ row_1 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(v_full_addr);
                mbarrier_wait_hint(s_full_addr, it_1 & 1, 10000000);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int col_lo_1 = 32 + 64 * half_1;
                float sv_1[16];
                tmem_ld_x16(&sv_1[0], taddr + (unsigned int)col_lo_1 + (unsigned int)(tmem_row_origin_1 << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int mword_1 = mask_word_1;
                sv_1[0] = (((mword_1 & 1) != 0) ? sv_1[0] : -CAKE_INF);
                sv_1[1] = (((mword_1 >> 1 & 1) != 0) ? sv_1[1] : -CAKE_INF);
                sv_1[2] = (((mword_1 >> 2 & 1) != 0) ? sv_1[2] : -CAKE_INF);
                sv_1[3] = (((mword_1 >> 3 & 1) != 0) ? sv_1[3] : -CAKE_INF);
                sv_1[4] = (((mword_1 >> 4 & 1) != 0) ? sv_1[4] : -CAKE_INF);
                sv_1[5] = (((mword_1 >> 5 & 1) != 0) ? sv_1[5] : -CAKE_INF);
                sv_1[6] = (((mword_1 >> 6 & 1) != 0) ? sv_1[6] : -CAKE_INF);
                sv_1[7] = (((mword_1 >> 7 & 1) != 0) ? sv_1[7] : -CAKE_INF);
                sv_1[8] = (((mword_1 >> 8 & 1) != 0) ? sv_1[8] : -CAKE_INF);
                sv_1[9] = (((mword_1 >> 9 & 1) != 0) ? sv_1[9] : -CAKE_INF);
                sv_1[10] = (((mword_1 >> 10 & 1) != 0) ? sv_1[10] : -CAKE_INF);
                sv_1[11] = (((mword_1 >> 11 & 1) != 0) ? sv_1[11] : -CAKE_INF);
                sv_1[12] = (((mword_1 >> 12 & 1) != 0) ? sv_1[12] : -CAKE_INF);
                sv_1[13] = (((mword_1 >> 13 & 1) != 0) ? sv_1[13] : -CAKE_INF);
                sv_1[14] = (((mword_1 >> 14 & 1) != 0) ? sv_1[14] : -CAKE_INF);
                sv_1[15] = (((mword_1 >> 15 & 1) != 0) ? sv_1[15] : -CAKE_INF);
                float mx_1[16];
                mx_1[0] = sv_1[0];
                mx_1[1] = sv_1[1];
                mx_1[2] = sv_1[2];
                mx_1[3] = sv_1[3];
                mx_1[4] = sv_1[4];
                mx_1[5] = sv_1[5];
                mx_1[6] = sv_1[6];
                mx_1[7] = sv_1[7];
                mx_1[8] = sv_1[8];
                mx_1[9] = sv_1[9];
                mx_1[10] = sv_1[10];
                mx_1[11] = sv_1[11];
                mx_1[12] = sv_1[12];
                mx_1[13] = sv_1[13];
                mx_1[14] = sv_1[14];
                mx_1[15] = sv_1[15];
                float _max_278 = max_noftz(mx_1[0], mx_1[8]);
                mx_1[0] = _max_278;
                float _max_279 = max_noftz(mx_1[1], mx_1[9]);
                mx_1[1] = _max_279;
                float _max_280 = max_noftz(mx_1[2], mx_1[10]);
                mx_1[2] = _max_280;
                float _max_281 = max_noftz(mx_1[3], mx_1[11]);
                mx_1[3] = _max_281;
                float _max_282 = max_noftz(mx_1[4], mx_1[12]);
                mx_1[4] = _max_282;
                float _max_283 = max_noftz(mx_1[5], mx_1[13]);
                mx_1[5] = _max_283;
                float _max_284 = max_noftz(mx_1[6], mx_1[14]);
                mx_1[6] = _max_284;
                float _max_285 = max_noftz(mx_1[7], mx_1[15]);
                mx_1[7] = _max_285;
                float _max_286 = max_noftz(mx_1[0], mx_1[4]);
                mx_1[0] = _max_286;
                float _max_287 = max_noftz(mx_1[1], mx_1[5]);
                mx_1[1] = _max_287;
                float _max_288 = max_noftz(mx_1[2], mx_1[6]);
                mx_1[2] = _max_288;
                float _max_289 = max_noftz(mx_1[3], mx_1[7]);
                mx_1[3] = _max_289;
                float _max_290 = max_noftz(mx_1[0], mx_1[2]);
                mx_1[0] = _max_290;
                float _max_291 = max_noftz(mx_1[1], mx_1[3]);
                mx_1[1] = _max_291;
                float _max_292 = max_noftz(mx_1[0], mx_1[1]);
                mx_1[0] = _max_292;
                int slice_id_2 = 1 + 3 * half_1;
                smem_pmax[slice_id_2 * 64 + head_1] = mx_1[0];
                asm volatile("barrier.sync 10, 384;" ::: "memory");
                float m_tile_1 = smem_pmax[head_1];
                float _max_293 = max_noftz(m_tile_1, smem_pmax[64 + head_1]);
                m_tile_1 = _max_293;
                float _max_294 = max_noftz(m_tile_1, smem_pmax[128 + head_1]);
                m_tile_1 = _max_294;
                float _max_295 = max_noftz(m_tile_1, smem_pmax[192 + head_1]);
                m_tile_1 = _max_295;
                float _max_296 = max_noftz(m_tile_1, smem_pmax[256 + head_1]);
                m_tile_1 = _max_296;
                float _max_297 = max_noftz(m_tile_1, smem_pmax[320 + head_1]);
                m_tile_1 = _max_297;
                float cand_1 = m_tile_1 * softmax_scale_log2_1;
                if (it_1 == 0) {
                    float _max_298 = max_noftz(cand_1, sink_lane_1);
                    cand_1 = _max_298;
                }
                float _max_299 = max_noftz(cand_1, sm_1[0]);
                cand_1 = _max_299;
                int grow_1 = 0;
                float alpha_1 = 1.0f;
                if (it_1 == 0) {
                    grow_1 = 1;
                }
                if (cand_1 - sm_1[0] > 8.0f) {
                    grow_1 = 1;
                }
                if (grow_1 != 0) {
                    float _exp2_34 = approx_exp2(sm_1[0] - cand_1);
                    alpha_1 = ((sm_1[0] > -CAKE_INF) ? _exp2_34 : 0.0f);
                    sm_1[1] = sm_1[1] * alpha_1;
                    sm_1[2] = sm_1[2] * alpha_1;
                    sm_1[0] = cand_1;
                }
                float m_scaled_1 = ((sm_1[0] > -CAKE_INF) ? sm_1[0] : 0.0f);
                unsigned int _vote_5 = __ballot_sync(0xFFFFFFFF, grow_1 != 0);
                unsigned int grow_bits_1 = _vote_5;
                asm volatile("barrier.sync 10, 384;" ::: "memory");
                if (it_1 > 0) {
                    unsigned int any_grow_1 = smem_flag[0] | smem_flag[1];
                    if (any_grow_1 != 0) {
                        float alpha_c_1[16];
                        alpha_c_1[0] = smem_alpha[32];
                        alpha_c_1[1] = smem_alpha[33];
                        alpha_c_1[2] = smem_alpha[34];
                        alpha_c_1[3] = smem_alpha[35];
                        alpha_c_1[4] = smem_alpha[36];
                        alpha_c_1[5] = smem_alpha[37];
                        alpha_c_1[6] = smem_alpha[38];
                        alpha_c_1[7] = smem_alpha[39];
                        alpha_c_1[8] = smem_alpha[40];
                        alpha_c_1[9] = smem_alpha[41];
                        alpha_c_1[10] = smem_alpha[42];
                        alpha_c_1[11] = smem_alpha[43];
                        alpha_c_1[12] = smem_alpha[44];
                        alpha_c_1[13] = smem_alpha[45];
                        alpha_c_1[14] = smem_alpha[46];
                        alpha_c_1[15] = smem_alpha[47];
                        float ov_1[16];
                        tmem_ld_x16(&ov_1[0], taddr + 128 + 32 + (unsigned int)(tmem_row_origin_1 << 16));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        ov_1[0] = ov_1[0] * alpha_c_1[0];
                        ov_1[1] = ov_1[1] * alpha_c_1[1];
                        ov_1[2] = ov_1[2] * alpha_c_1[2];
                        ov_1[3] = ov_1[3] * alpha_c_1[3];
                        ov_1[4] = ov_1[4] * alpha_c_1[4];
                        ov_1[5] = ov_1[5] * alpha_c_1[5];
                        ov_1[6] = ov_1[6] * alpha_c_1[6];
                        ov_1[7] = ov_1[7] * alpha_c_1[7];
                        ov_1[8] = ov_1[8] * alpha_c_1[8];
                        ov_1[9] = ov_1[9] * alpha_c_1[9];
                        ov_1[10] = ov_1[10] * alpha_c_1[10];
                        ov_1[11] = ov_1[11] * alpha_c_1[11];
                        ov_1[12] = ov_1[12] * alpha_c_1[12];
                        ov_1[13] = ov_1[13] * alpha_c_1[13];
                        ov_1[14] = ov_1[14] * alpha_c_1[14];
                        ov_1[15] = ov_1[15] * alpha_c_1[15];
                        tmem_st_x16_f32(taddr + 128 + 32 + (unsigned int)(tmem_row_origin_1 << 16), ov_1);
                        tmem_ld_x16(&ov_1[0], taddr + 192 + 32 + (unsigned int)(tmem_row_origin_1 << 16));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        ov_1[0] = ov_1[0] * alpha_c_1[0];
                        ov_1[1] = ov_1[1] * alpha_c_1[1];
                        ov_1[2] = ov_1[2] * alpha_c_1[2];
                        ov_1[3] = ov_1[3] * alpha_c_1[3];
                        ov_1[4] = ov_1[4] * alpha_c_1[4];
                        ov_1[5] = ov_1[5] * alpha_c_1[5];
                        ov_1[6] = ov_1[6] * alpha_c_1[6];
                        ov_1[7] = ov_1[7] * alpha_c_1[7];
                        ov_1[8] = ov_1[8] * alpha_c_1[8];
                        ov_1[9] = ov_1[9] * alpha_c_1[9];
                        ov_1[10] = ov_1[10] * alpha_c_1[10];
                        ov_1[11] = ov_1[11] * alpha_c_1[11];
                        ov_1[12] = ov_1[12] * alpha_c_1[12];
                        ov_1[13] = ov_1[13] * alpha_c_1[13];
                        ov_1[14] = ov_1[14] * alpha_c_1[14];
                        ov_1[15] = ov_1[15] * alpha_c_1[15];
                        tmem_st_x16_f32(taddr + 192 + 32 + (unsigned int)(tmem_row_origin_1 << 16), ov_1);
                        tmem_ld_x16(&ov_1[0], taddr + 256 + 32 + (unsigned int)(tmem_row_origin_1 << 16));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        ov_1[0] = ov_1[0] * alpha_c_1[0];
                        ov_1[1] = ov_1[1] * alpha_c_1[1];
                        ov_1[2] = ov_1[2] * alpha_c_1[2];
                        ov_1[3] = ov_1[3] * alpha_c_1[3];
                        ov_1[4] = ov_1[4] * alpha_c_1[4];
                        ov_1[5] = ov_1[5] * alpha_c_1[5];
                        ov_1[6] = ov_1[6] * alpha_c_1[6];
                        ov_1[7] = ov_1[7] * alpha_c_1[7];
                        ov_1[8] = ov_1[8] * alpha_c_1[8];
                        ov_1[9] = ov_1[9] * alpha_c_1[9];
                        ov_1[10] = ov_1[10] * alpha_c_1[10];
                        ov_1[11] = ov_1[11] * alpha_c_1[11];
                        ov_1[12] = ov_1[12] * alpha_c_1[12];
                        ov_1[13] = ov_1[13] * alpha_c_1[13];
                        ov_1[14] = ov_1[14] * alpha_c_1[14];
                        ov_1[15] = ov_1[15] * alpha_c_1[15];
                        tmem_st_x16_f32(taddr + 256 + 32 + (unsigned int)(tmem_row_origin_1 << 16), ov_1);
                        tmem_ld_x16(&ov_1[0], taddr + 320 + 32 + (unsigned int)(tmem_row_origin_1 << 16));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        ov_1[0] = ov_1[0] * alpha_c_1[0];
                        ov_1[1] = ov_1[1] * alpha_c_1[1];
                        ov_1[2] = ov_1[2] * alpha_c_1[2];
                        ov_1[3] = ov_1[3] * alpha_c_1[3];
                        ov_1[4] = ov_1[4] * alpha_c_1[4];
                        ov_1[5] = ov_1[5] * alpha_c_1[5];
                        ov_1[6] = ov_1[6] * alpha_c_1[6];
                        ov_1[7] = ov_1[7] * alpha_c_1[7];
                        ov_1[8] = ov_1[8] * alpha_c_1[8];
                        ov_1[9] = ov_1[9] * alpha_c_1[9];
                        ov_1[10] = ov_1[10] * alpha_c_1[10];
                        ov_1[11] = ov_1[11] * alpha_c_1[11];
                        ov_1[12] = ov_1[12] * alpha_c_1[12];
                        ov_1[13] = ov_1[13] * alpha_c_1[13];
                        ov_1[14] = ov_1[14] * alpha_c_1[14];
                        ov_1[15] = ov_1[15] * alpha_c_1[15];
                        tmem_st_x16_f32(taddr + 320 + 32 + (unsigned int)(tmem_row_origin_1 << 16), ov_1);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                }
                float psum_1 = 0.0f;
                float rsum_1 = 0.0f;
                float _exp2_35 = approx_exp2(sv_1[0] * softmax_scale_log2_1 - m_scaled_1);
                sv_1[0] = _exp2_35;
                float _exp2_36 = approx_exp2(sv_1[1] * softmax_scale_log2_1 - m_scaled_1);
                sv_1[1] = _exp2_36;
                float _exp2_37 = approx_exp2(sv_1[2] * softmax_scale_log2_1 - m_scaled_1);
                sv_1[2] = _exp2_37;
                float _exp2_38 = approx_exp2(sv_1[3] * softmax_scale_log2_1 - m_scaled_1);
                sv_1[3] = _exp2_38;
                float _exp2_39 = approx_exp2(sv_1[4] * softmax_scale_log2_1 - m_scaled_1);
                sv_1[4] = _exp2_39;
                float _exp2_40 = approx_exp2(sv_1[5] * softmax_scale_log2_1 - m_scaled_1);
                sv_1[5] = _exp2_40;
                float _exp2_41 = approx_exp2(sv_1[6] * softmax_scale_log2_1 - m_scaled_1);
                sv_1[6] = _exp2_41;
                float _exp2_42 = approx_exp2(sv_1[7] * softmax_scale_log2_1 - m_scaled_1);
                sv_1[7] = _exp2_42;
                float _exp2_43 = approx_exp2(sv_1[8] * softmax_scale_log2_1 - m_scaled_1);
                sv_1[8] = _exp2_43;
                float _exp2_44 = approx_exp2(sv_1[9] * softmax_scale_log2_1 - m_scaled_1);
                sv_1[9] = _exp2_44;
                float _exp2_45 = approx_exp2(sv_1[10] * softmax_scale_log2_1 - m_scaled_1);
                sv_1[10] = _exp2_45;
                float _exp2_46 = approx_exp2(sv_1[11] * softmax_scale_log2_1 - m_scaled_1);
                sv_1[11] = _exp2_46;
                float _exp2_47 = approx_exp2(sv_1[12] * softmax_scale_log2_1 - m_scaled_1);
                sv_1[12] = _exp2_47;
                float _exp2_48 = approx_exp2(sv_1[13] * softmax_scale_log2_1 - m_scaled_1);
                sv_1[13] = _exp2_48;
                float _exp2_49 = approx_exp2(sv_1[14] * softmax_scale_log2_1 - m_scaled_1);
                sv_1[14] = _exp2_49;
                float _exp2_50 = approx_exp2(sv_1[15] * softmax_scale_log2_1 - m_scaled_1);
                sv_1[15] = _exp2_50;
                unsigned int pw_1[4];
                uint16_t _e4m3x2_f32_384;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_384) : "f"(sv_1[1]), "f"(sv_1[0]));
                uint16_t pair0_1 = _e4m3x2_f32_384;
                uint16_t _e4m3x2_f32_385;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_385) : "f"(sv_1[3]), "f"(sv_1[2]));
                uint16_t pair1_2 = _e4m3x2_f32_385;
                pw_1[0] = (unsigned int)pair0_1 | (unsigned int)pair1_2 << 16;
                uint16_t _e4m3x2_f32_386;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_386) : "f"(sv_1[5]), "f"(sv_1[4]));
                uint16_t pair0_0_1 = _e4m3x2_f32_386;
                uint16_t _e4m3x2_f32_387;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_387) : "f"(sv_1[7]), "f"(sv_1[6]));
                uint16_t pair1_1_1 = _e4m3x2_f32_387;
                pw_1[1] = (unsigned int)pair0_0_1 | (unsigned int)pair1_1_1 << 16;
                uint16_t _e4m3x2_f32_388;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_388) : "f"(sv_1[9]), "f"(sv_1[8]));
                uint16_t pair0_2_1 = _e4m3x2_f32_388;
                uint16_t _e4m3x2_f32_389;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_389) : "f"(sv_1[11]), "f"(sv_1[10]));
                uint16_t pair1_3_1 = _e4m3x2_f32_389;
                pw_1[2] = (unsigned int)pair0_2_1 | (unsigned int)pair1_3_1 << 16;
                uint16_t _e4m3x2_f32_390;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_390) : "f"(sv_1[13]), "f"(sv_1[12]));
                uint16_t pair0_4_1 = _e4m3x2_f32_390;
                uint16_t _e4m3x2_f32_391;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_391) : "f"(sv_1[15]), "f"(sv_1[14]));
                uint16_t pair1_5_1 = _e4m3x2_f32_391;
                pw_1[3] = (unsigned int)pair0_4_1 | (unsigned int)pair1_5_1 << 16;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(pbase_1 + (head_1 * 128 + ((col_lo_1 >> 4) * 16 ^ head_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&pw_1[0])), "r"(*reinterpret_cast<uint32_t*>(&pw_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&pw_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&pw_1[(0) + 3])));
                psum_1 = psum_1 + sv_1[0];
                float _fp8_rt_32;
                uint16_t _e4m3x2_52;
                uint32_t _f16x2_52;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_52) : "f"(0.0f), "f"(sv_1[0]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_52) : "h"(_e4m3x2_52));
                uint16_t _fp8_h0_52 = (uint16_t)(_f16x2_52 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_32) : "h"(_fp8_h0_52));
                rsum_1 = rsum_1 + _fp8_rt_32;
                psum_1 = psum_1 + sv_1[1];
                float _fp8_rt_33;
                uint16_t _e4m3x2_53;
                uint32_t _f16x2_53;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_53) : "f"(0.0f), "f"(sv_1[1]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_53) : "h"(_e4m3x2_53));
                uint16_t _fp8_h0_53 = (uint16_t)(_f16x2_53 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_33) : "h"(_fp8_h0_53));
                rsum_1 = rsum_1 + _fp8_rt_33;
                psum_1 = psum_1 + sv_1[2];
                float _fp8_rt_34;
                uint16_t _e4m3x2_54;
                uint32_t _f16x2_54;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_54) : "f"(0.0f), "f"(sv_1[2]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_54) : "h"(_e4m3x2_54));
                uint16_t _fp8_h0_54 = (uint16_t)(_f16x2_54 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_34) : "h"(_fp8_h0_54));
                rsum_1 = rsum_1 + _fp8_rt_34;
                psum_1 = psum_1 + sv_1[3];
                float _fp8_rt_35;
                uint16_t _e4m3x2_55;
                uint32_t _f16x2_55;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_55) : "f"(0.0f), "f"(sv_1[3]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_55) : "h"(_e4m3x2_55));
                uint16_t _fp8_h0_55 = (uint16_t)(_f16x2_55 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_35) : "h"(_fp8_h0_55));
                rsum_1 = rsum_1 + _fp8_rt_35;
                psum_1 = psum_1 + sv_1[4];
                float _fp8_rt_36;
                uint16_t _e4m3x2_56;
                uint32_t _f16x2_56;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_56) : "f"(0.0f), "f"(sv_1[4]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_56) : "h"(_e4m3x2_56));
                uint16_t _fp8_h0_56 = (uint16_t)(_f16x2_56 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_36) : "h"(_fp8_h0_56));
                rsum_1 = rsum_1 + _fp8_rt_36;
                psum_1 = psum_1 + sv_1[5];
                float _fp8_rt_37;
                uint16_t _e4m3x2_57;
                uint32_t _f16x2_57;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_57) : "f"(0.0f), "f"(sv_1[5]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_57) : "h"(_e4m3x2_57));
                uint16_t _fp8_h0_57 = (uint16_t)(_f16x2_57 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_37) : "h"(_fp8_h0_57));
                rsum_1 = rsum_1 + _fp8_rt_37;
                psum_1 = psum_1 + sv_1[6];
                float _fp8_rt_38;
                uint16_t _e4m3x2_58;
                uint32_t _f16x2_58;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_58) : "f"(0.0f), "f"(sv_1[6]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_58) : "h"(_e4m3x2_58));
                uint16_t _fp8_h0_58 = (uint16_t)(_f16x2_58 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_38) : "h"(_fp8_h0_58));
                rsum_1 = rsum_1 + _fp8_rt_38;
                psum_1 = psum_1 + sv_1[7];
                float _fp8_rt_39;
                uint16_t _e4m3x2_59;
                uint32_t _f16x2_59;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_59) : "f"(0.0f), "f"(sv_1[7]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_59) : "h"(_e4m3x2_59));
                uint16_t _fp8_h0_59 = (uint16_t)(_f16x2_59 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_39) : "h"(_fp8_h0_59));
                rsum_1 = rsum_1 + _fp8_rt_39;
                psum_1 = psum_1 + sv_1[8];
                float _fp8_rt_40;
                uint16_t _e4m3x2_60;
                uint32_t _f16x2_60;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_60) : "f"(0.0f), "f"(sv_1[8]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_60) : "h"(_e4m3x2_60));
                uint16_t _fp8_h0_60 = (uint16_t)(_f16x2_60 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_40) : "h"(_fp8_h0_60));
                rsum_1 = rsum_1 + _fp8_rt_40;
                psum_1 = psum_1 + sv_1[9];
                float _fp8_rt_41;
                uint16_t _e4m3x2_61;
                uint32_t _f16x2_61;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_61) : "f"(0.0f), "f"(sv_1[9]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_61) : "h"(_e4m3x2_61));
                uint16_t _fp8_h0_61 = (uint16_t)(_f16x2_61 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_41) : "h"(_fp8_h0_61));
                rsum_1 = rsum_1 + _fp8_rt_41;
                psum_1 = psum_1 + sv_1[10];
                float _fp8_rt_42;
                uint16_t _e4m3x2_62;
                uint32_t _f16x2_62;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_62) : "f"(0.0f), "f"(sv_1[10]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_62) : "h"(_e4m3x2_62));
                uint16_t _fp8_h0_62 = (uint16_t)(_f16x2_62 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_42) : "h"(_fp8_h0_62));
                rsum_1 = rsum_1 + _fp8_rt_42;
                psum_1 = psum_1 + sv_1[11];
                float _fp8_rt_43;
                uint16_t _e4m3x2_63;
                uint32_t _f16x2_63;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_63) : "f"(0.0f), "f"(sv_1[11]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_63) : "h"(_e4m3x2_63));
                uint16_t _fp8_h0_63 = (uint16_t)(_f16x2_63 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_43) : "h"(_fp8_h0_63));
                rsum_1 = rsum_1 + _fp8_rt_43;
                psum_1 = psum_1 + sv_1[12];
                float _fp8_rt_44;
                uint16_t _e4m3x2_64;
                uint32_t _f16x2_64;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_64) : "f"(0.0f), "f"(sv_1[12]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_64) : "h"(_e4m3x2_64));
                uint16_t _fp8_h0_64 = (uint16_t)(_f16x2_64 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_44) : "h"(_fp8_h0_64));
                rsum_1 = rsum_1 + _fp8_rt_44;
                psum_1 = psum_1 + sv_1[13];
                float _fp8_rt_45;
                uint16_t _e4m3x2_65;
                uint32_t _f16x2_65;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_65) : "f"(0.0f), "f"(sv_1[13]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_65) : "h"(_e4m3x2_65));
                uint16_t _fp8_h0_65 = (uint16_t)(_f16x2_65 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_45) : "h"(_fp8_h0_65));
                rsum_1 = rsum_1 + _fp8_rt_45;
                psum_1 = psum_1 + sv_1[14];
                float _fp8_rt_46;
                uint16_t _e4m3x2_66;
                uint32_t _f16x2_66;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_66) : "f"(0.0f), "f"(sv_1[14]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_66) : "h"(_e4m3x2_66));
                uint16_t _fp8_h0_66 = (uint16_t)(_f16x2_66 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_46) : "h"(_fp8_h0_66));
                rsum_1 = rsum_1 + _fp8_rt_46;
                psum_1 = psum_1 + sv_1[15];
                float _fp8_rt_47;
                uint16_t _e4m3x2_67;
                uint32_t _f16x2_67;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_67) : "f"(0.0f), "f"(sv_1[15]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_67) : "h"(_e4m3x2_67));
                uint16_t _fp8_h0_67 = (uint16_t)(_f16x2_67 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_47) : "h"(_fp8_h0_67));
                rsum_1 = rsum_1 + _fp8_rt_47;
                if (it_1 == 0) {
                    float sink_term_1 = 0.0f;
                    sm_1[1] = sink_term_1;
                    sm_1[2] = sink_term_1;
                }
                sm_1[1] = sm_1[1] + psum_1;
                sm_1[2] = sm_1[2] + rsum_1;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(p_full_addr);
            }
            int last_par_1 = tiles_per_split - 1 & 1;
            mbarrier_wait_hint(o_full_addr, last_par_1, 10000000);
            mbarrier_wait_hint(o_full_addr + 8, last_par_1, 10000000);
            mbarrier_wait_hint(o_full_addr + 16, last_par_1, 10000000);
            mbarrier_wait_hint(o_full_addr + 24, last_par_1, 10000000);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int slice_id_3 = 1 + 3 * half_1;
            smem_xsum[slice_id_3 * 64 + head_1] = sm_1[1];
            smem_xsum[384 + slice_id_3 * 64 + head_1] = sm_1[2];
            asm volatile("barrier.sync 10, 384;" ::: "memory");
            asm volatile("barrier.sync 10, 384;" ::: "memory");
            float norm_c_1[16];
            norm_c_1[0] = smem_norm[32];
            norm_c_1[1] = smem_norm[33];
            norm_c_1[2] = smem_norm[34];
            norm_c_1[3] = smem_norm[35];
            norm_c_1[4] = smem_norm[36];
            norm_c_1[5] = smem_norm[37];
            norm_c_1[6] = smem_norm[38];
            norm_c_1[7] = smem_norm[39];
            norm_c_1[8] = smem_norm[40];
            norm_c_1[9] = smem_norm[41];
            norm_c_1[10] = smem_norm[42];
            norm_c_1[11] = smem_norm[43];
            norm_c_1[12] = smem_norm[44];
            norm_c_1[13] = smem_norm[45];
            norm_c_1[14] = smem_norm[46];
            norm_c_1[15] = smem_norm[47];
            float o_values_1[16];
            float o_scaled_1[16];
            int is_odd_1 = lane & 1;
            tmem_ld_x16(&o_values_1[0], taddr + 128 + 32 + (unsigned int)(tmem_row_origin_1 << 16));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            o_scaled_1[0] = o_values_1[0] * norm_c_1[0];
            o_scaled_1[1] = o_values_1[1] * norm_c_1[1];
            o_scaled_1[2] = o_values_1[2] * norm_c_1[2];
            o_scaled_1[3] = o_values_1[3] * norm_c_1[3];
            o_scaled_1[4] = o_values_1[4] * norm_c_1[4];
            o_scaled_1[5] = o_values_1[5] * norm_c_1[5];
            o_scaled_1[6] = o_values_1[6] * norm_c_1[6];
            o_scaled_1[7] = o_values_1[7] * norm_c_1[7];
            o_scaled_1[8] = o_values_1[8] * norm_c_1[8];
            o_scaled_1[9] = o_values_1[9] * norm_c_1[9];
            o_scaled_1[10] = o_values_1[10] * norm_c_1[10];
            o_scaled_1[11] = o_values_1[11] * norm_c_1[11];
            o_scaled_1[12] = o_values_1[12] * norm_c_1[12];
            o_scaled_1[13] = o_values_1[13] * norm_c_1[13];
            o_scaled_1[14] = o_values_1[14] * norm_c_1[14];
            o_scaled_1[15] = o_values_1[15] * norm_c_1[15];
            uint32_t o_scaled_bf16_1[8];
            #pragma unroll
            for (int _lp = 0; _lp < 8; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_scaled_1[_lp*2 + 0], o_scaled_1[_lp*2+1 + 0]));
                o_scaled_bf16_1[_lp] = *(uint32_t*)&_bf2;
            }
            int dim_own_1 = row_1;
            smem_o16[16384 + dim_own_1] = (uint16_t)(o_scaled_bf16_1[0] & 65535);
            smem_o16[16896 + dim_own_1] = (uint16_t)(o_scaled_bf16_1[0] >> 16);
            smem_o16[17408 + dim_own_1] = (uint16_t)(o_scaled_bf16_1[1] & 65535);
            smem_o16[17920 + dim_own_1] = (uint16_t)(o_scaled_bf16_1[1] >> 16);
            smem_o16[18432 + dim_own_1] = (uint16_t)(o_scaled_bf16_1[2] & 65535);
            smem_o16[18944 + dim_own_1] = (uint16_t)(o_scaled_bf16_1[2] >> 16);
            smem_o16[19456 + dim_own_1] = (uint16_t)(o_scaled_bf16_1[3] & 65535);
            smem_o16[19968 + dim_own_1] = (uint16_t)(o_scaled_bf16_1[3] >> 16);
            smem_o16[20480 + dim_own_1] = (uint16_t)(o_scaled_bf16_1[4] & 65535);
            smem_o16[20992 + dim_own_1] = (uint16_t)(o_scaled_bf16_1[4] >> 16);
            smem_o16[21504 + dim_own_1] = (uint16_t)(o_scaled_bf16_1[5] & 65535);
            smem_o16[22016 + dim_own_1] = (uint16_t)(o_scaled_bf16_1[5] >> 16);
            smem_o16[22528 + dim_own_1] = (uint16_t)(o_scaled_bf16_1[6] & 65535);
            smem_o16[23040 + dim_own_1] = (uint16_t)(o_scaled_bf16_1[6] >> 16);
            smem_o16[23552 + dim_own_1] = (uint16_t)(o_scaled_bf16_1[7] & 65535);
            smem_o16[24064 + dim_own_1] = (uint16_t)(o_scaled_bf16_1[7] >> 16);
            tmem_ld_x16(&o_values_1[0], taddr + 192 + 32 + (unsigned int)(tmem_row_origin_1 << 16));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            o_scaled_1[0] = o_values_1[0] * norm_c_1[0];
            o_scaled_1[1] = o_values_1[1] * norm_c_1[1];
            o_scaled_1[2] = o_values_1[2] * norm_c_1[2];
            o_scaled_1[3] = o_values_1[3] * norm_c_1[3];
            o_scaled_1[4] = o_values_1[4] * norm_c_1[4];
            o_scaled_1[5] = o_values_1[5] * norm_c_1[5];
            o_scaled_1[6] = o_values_1[6] * norm_c_1[6];
            o_scaled_1[7] = o_values_1[7] * norm_c_1[7];
            o_scaled_1[8] = o_values_1[8] * norm_c_1[8];
            o_scaled_1[9] = o_values_1[9] * norm_c_1[9];
            o_scaled_1[10] = o_values_1[10] * norm_c_1[10];
            o_scaled_1[11] = o_values_1[11] * norm_c_1[11];
            o_scaled_1[12] = o_values_1[12] * norm_c_1[12];
            o_scaled_1[13] = o_values_1[13] * norm_c_1[13];
            o_scaled_1[14] = o_values_1[14] * norm_c_1[14];
            o_scaled_1[15] = o_values_1[15] * norm_c_1[15];
            uint32_t o_scaled_bf16_8_1[8];
            #pragma unroll
            for (int _lp = 0; _lp < 8; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_scaled_1[_lp*2 + 0], o_scaled_1[_lp*2+1 + 0]));
                o_scaled_bf16_8_1[_lp] = *(uint32_t*)&_bf2;
            }
            int dim_own_9_1 = 128 + row_1;
            smem_o16[16384 + dim_own_9_1] = (uint16_t)(o_scaled_bf16_8_1[0] & 65535);
            smem_o16[16896 + dim_own_9_1] = (uint16_t)(o_scaled_bf16_8_1[0] >> 16);
            smem_o16[17408 + dim_own_9_1] = (uint16_t)(o_scaled_bf16_8_1[1] & 65535);
            smem_o16[17920 + dim_own_9_1] = (uint16_t)(o_scaled_bf16_8_1[1] >> 16);
            smem_o16[18432 + dim_own_9_1] = (uint16_t)(o_scaled_bf16_8_1[2] & 65535);
            smem_o16[18944 + dim_own_9_1] = (uint16_t)(o_scaled_bf16_8_1[2] >> 16);
            smem_o16[19456 + dim_own_9_1] = (uint16_t)(o_scaled_bf16_8_1[3] & 65535);
            smem_o16[19968 + dim_own_9_1] = (uint16_t)(o_scaled_bf16_8_1[3] >> 16);
            smem_o16[20480 + dim_own_9_1] = (uint16_t)(o_scaled_bf16_8_1[4] & 65535);
            smem_o16[20992 + dim_own_9_1] = (uint16_t)(o_scaled_bf16_8_1[4] >> 16);
            smem_o16[21504 + dim_own_9_1] = (uint16_t)(o_scaled_bf16_8_1[5] & 65535);
            smem_o16[22016 + dim_own_9_1] = (uint16_t)(o_scaled_bf16_8_1[5] >> 16);
            smem_o16[22528 + dim_own_9_1] = (uint16_t)(o_scaled_bf16_8_1[6] & 65535);
            smem_o16[23040 + dim_own_9_1] = (uint16_t)(o_scaled_bf16_8_1[6] >> 16);
            smem_o16[23552 + dim_own_9_1] = (uint16_t)(o_scaled_bf16_8_1[7] & 65535);
            smem_o16[24064 + dim_own_9_1] = (uint16_t)(o_scaled_bf16_8_1[7] >> 16);
            tmem_ld_x16(&o_values_1[0], taddr + 256 + 32 + (unsigned int)(tmem_row_origin_1 << 16));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            o_scaled_1[0] = o_values_1[0] * norm_c_1[0];
            o_scaled_1[1] = o_values_1[1] * norm_c_1[1];
            o_scaled_1[2] = o_values_1[2] * norm_c_1[2];
            o_scaled_1[3] = o_values_1[3] * norm_c_1[3];
            o_scaled_1[4] = o_values_1[4] * norm_c_1[4];
            o_scaled_1[5] = o_values_1[5] * norm_c_1[5];
            o_scaled_1[6] = o_values_1[6] * norm_c_1[6];
            o_scaled_1[7] = o_values_1[7] * norm_c_1[7];
            o_scaled_1[8] = o_values_1[8] * norm_c_1[8];
            o_scaled_1[9] = o_values_1[9] * norm_c_1[9];
            o_scaled_1[10] = o_values_1[10] * norm_c_1[10];
            o_scaled_1[11] = o_values_1[11] * norm_c_1[11];
            o_scaled_1[12] = o_values_1[12] * norm_c_1[12];
            o_scaled_1[13] = o_values_1[13] * norm_c_1[13];
            o_scaled_1[14] = o_values_1[14] * norm_c_1[14];
            o_scaled_1[15] = o_values_1[15] * norm_c_1[15];
            uint32_t o_scaled_bf16_10_1[8];
            #pragma unroll
            for (int _lp = 0; _lp < 8; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_scaled_1[_lp*2 + 0], o_scaled_1[_lp*2+1 + 0]));
                o_scaled_bf16_10_1[_lp] = *(uint32_t*)&_bf2;
            }
            int dim_own_11_1 = 256 + row_1;
            smem_o16[16384 + dim_own_11_1] = (uint16_t)(o_scaled_bf16_10_1[0] & 65535);
            smem_o16[16896 + dim_own_11_1] = (uint16_t)(o_scaled_bf16_10_1[0] >> 16);
            smem_o16[17408 + dim_own_11_1] = (uint16_t)(o_scaled_bf16_10_1[1] & 65535);
            smem_o16[17920 + dim_own_11_1] = (uint16_t)(o_scaled_bf16_10_1[1] >> 16);
            smem_o16[18432 + dim_own_11_1] = (uint16_t)(o_scaled_bf16_10_1[2] & 65535);
            smem_o16[18944 + dim_own_11_1] = (uint16_t)(o_scaled_bf16_10_1[2] >> 16);
            smem_o16[19456 + dim_own_11_1] = (uint16_t)(o_scaled_bf16_10_1[3] & 65535);
            smem_o16[19968 + dim_own_11_1] = (uint16_t)(o_scaled_bf16_10_1[3] >> 16);
            smem_o16[20480 + dim_own_11_1] = (uint16_t)(o_scaled_bf16_10_1[4] & 65535);
            smem_o16[20992 + dim_own_11_1] = (uint16_t)(o_scaled_bf16_10_1[4] >> 16);
            smem_o16[21504 + dim_own_11_1] = (uint16_t)(o_scaled_bf16_10_1[5] & 65535);
            smem_o16[22016 + dim_own_11_1] = (uint16_t)(o_scaled_bf16_10_1[5] >> 16);
            smem_o16[22528 + dim_own_11_1] = (uint16_t)(o_scaled_bf16_10_1[6] & 65535);
            smem_o16[23040 + dim_own_11_1] = (uint16_t)(o_scaled_bf16_10_1[6] >> 16);
            smem_o16[23552 + dim_own_11_1] = (uint16_t)(o_scaled_bf16_10_1[7] & 65535);
            smem_o16[24064 + dim_own_11_1] = (uint16_t)(o_scaled_bf16_10_1[7] >> 16);
            tmem_ld_x16(&o_values_1[0], taddr + 320 + 32 + (unsigned int)(tmem_row_origin_1 << 16));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            o_scaled_1[0] = o_values_1[0] * norm_c_1[0];
            o_scaled_1[1] = o_values_1[1] * norm_c_1[1];
            o_scaled_1[2] = o_values_1[2] * norm_c_1[2];
            o_scaled_1[3] = o_values_1[3] * norm_c_1[3];
            o_scaled_1[4] = o_values_1[4] * norm_c_1[4];
            o_scaled_1[5] = o_values_1[5] * norm_c_1[5];
            o_scaled_1[6] = o_values_1[6] * norm_c_1[6];
            o_scaled_1[7] = o_values_1[7] * norm_c_1[7];
            o_scaled_1[8] = o_values_1[8] * norm_c_1[8];
            o_scaled_1[9] = o_values_1[9] * norm_c_1[9];
            o_scaled_1[10] = o_values_1[10] * norm_c_1[10];
            o_scaled_1[11] = o_values_1[11] * norm_c_1[11];
            o_scaled_1[12] = o_values_1[12] * norm_c_1[12];
            o_scaled_1[13] = o_values_1[13] * norm_c_1[13];
            o_scaled_1[14] = o_values_1[14] * norm_c_1[14];
            o_scaled_1[15] = o_values_1[15] * norm_c_1[15];
            uint32_t o_scaled_bf16_12_1[8];
            #pragma unroll
            for (int _lp = 0; _lp < 8; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_scaled_1[_lp*2 + 0], o_scaled_1[_lp*2+1 + 0]));
                o_scaled_bf16_12_1[_lp] = *(uint32_t*)&_bf2;
            }
            int dim_own_13_1 = 384 + row_1;
            smem_o16[16384 + dim_own_13_1] = (uint16_t)(o_scaled_bf16_12_1[0] & 65535);
            smem_o16[16896 + dim_own_13_1] = (uint16_t)(o_scaled_bf16_12_1[0] >> 16);
            smem_o16[17408 + dim_own_13_1] = (uint16_t)(o_scaled_bf16_12_1[1] & 65535);
            smem_o16[17920 + dim_own_13_1] = (uint16_t)(o_scaled_bf16_12_1[1] >> 16);
            smem_o16[18432 + dim_own_13_1] = (uint16_t)(o_scaled_bf16_12_1[2] & 65535);
            smem_o16[18944 + dim_own_13_1] = (uint16_t)(o_scaled_bf16_12_1[2] >> 16);
            smem_o16[19456 + dim_own_13_1] = (uint16_t)(o_scaled_bf16_12_1[3] & 65535);
            smem_o16[19968 + dim_own_13_1] = (uint16_t)(o_scaled_bf16_12_1[3] >> 16);
            smem_o16[20480 + dim_own_13_1] = (uint16_t)(o_scaled_bf16_12_1[4] & 65535);
            smem_o16[20992 + dim_own_13_1] = (uint16_t)(o_scaled_bf16_12_1[4] >> 16);
            smem_o16[21504 + dim_own_13_1] = (uint16_t)(o_scaled_bf16_12_1[5] & 65535);
            smem_o16[22016 + dim_own_13_1] = (uint16_t)(o_scaled_bf16_12_1[5] >> 16);
            smem_o16[22528 + dim_own_13_1] = (uint16_t)(o_scaled_bf16_12_1[6] & 65535);
            smem_o16[23040 + dim_own_13_1] = (uint16_t)(o_scaled_bf16_12_1[6] >> 16);
            smem_o16[23552 + dim_own_13_1] = (uint16_t)(o_scaled_bf16_12_1[7] & 65535);
            smem_o16[24064 + dim_own_13_1] = (uint16_t)(o_scaled_bf16_12_1[7] >> 16);
            asm volatile("barrier.sync 10, 384;" ::: "memory");
            int tid384_1 = 128 + row_1;
            int ci_1 = tid384_1;
            if (ci_1 < 4096) {
                int h_11 = ci_1 / 64;
                int ch_11 = ci_1 - h_11 * 64;
                if (head_base_1 + h_11 < num_heads) {
                    unsigned int w4_11[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_11[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_11[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_11[(0) + 3]))
                        : "r"(smem_o32_addr + (unsigned int)(h_11 * 1024) + (unsigned int)(ch_11 * 16)));
                    long long out_off_11 = ((long long)(query_idx_1 * num_heads + head_base_1 + h_11) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)(o_chunk_1 * 512) + (long long)(ch_11 * 8);
                    reinterpret_cast<int4*>(partial_O + out_off_11)[0] = reinterpret_cast<int4*>(w4_11)[0];
                }
            }
            int ci_14_1 = tid384_1 + 384;
            if (ci_14_1 < 4096) {
                int h_12 = ci_14_1 / 64;
                int ch_12 = ci_14_1 - h_12 * 64;
                if (head_base_1 + h_12 < num_heads) {
                    unsigned int w4_12[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12[(0) + 3]))
                        : "r"(smem_o32_addr + (unsigned int)(h_12 * 1024) + (unsigned int)(ch_12 * 16)));
                    long long out_off_12 = ((long long)(query_idx_1 * num_heads + head_base_1 + h_12) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)(o_chunk_1 * 512) + (long long)(ch_12 * 8);
                    reinterpret_cast<int4*>(partial_O + out_off_12)[0] = reinterpret_cast<int4*>(w4_12)[0];
                }
            }
            int ci_15_1 = tid384_1 + 768;
            if (ci_15_1 < 4096) {
                int h_13 = ci_15_1 / 64;
                int ch_13 = ci_15_1 - h_13 * 64;
                if (head_base_1 + h_13 < num_heads) {
                    unsigned int w4_13[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_13[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_13[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_13[(0) + 3]))
                        : "r"(smem_o32_addr + (unsigned int)(h_13 * 1024) + (unsigned int)(ch_13 * 16)));
                    long long out_off_13 = ((long long)(query_idx_1 * num_heads + head_base_1 + h_13) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)(o_chunk_1 * 512) + (long long)(ch_13 * 8);
                    reinterpret_cast<int4*>(partial_O + out_off_13)[0] = reinterpret_cast<int4*>(w4_13)[0];
                }
            }
            int ci_16_1 = tid384_1 + 1152;
            if (ci_16_1 < 4096) {
                int h_14 = ci_16_1 / 64;
                int ch_14 = ci_16_1 - h_14 * 64;
                if (head_base_1 + h_14 < num_heads) {
                    unsigned int w4_14[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_14[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_14[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_14[(0) + 3]))
                        : "r"(smem_o32_addr + (unsigned int)(h_14 * 1024) + (unsigned int)(ch_14 * 16)));
                    long long out_off_14 = ((long long)(query_idx_1 * num_heads + head_base_1 + h_14) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)(o_chunk_1 * 512) + (long long)(ch_14 * 8);
                    reinterpret_cast<int4*>(partial_O + out_off_14)[0] = reinterpret_cast<int4*>(w4_14)[0];
                }
            }
            int ci_17_1 = tid384_1 + 1536;
            if (ci_17_1 < 4096) {
                int h_15 = ci_17_1 / 64;
                int ch_15 = ci_17_1 - h_15 * 64;
                if (head_base_1 + h_15 < num_heads) {
                    unsigned int w4_15[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_15[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_15[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_15[(0) + 3]))
                        : "r"(smem_o32_addr + (unsigned int)(h_15 * 1024) + (unsigned int)(ch_15 * 16)));
                    long long out_off_15 = ((long long)(query_idx_1 * num_heads + head_base_1 + h_15) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)(o_chunk_1 * 512) + (long long)(ch_15 * 8);
                    reinterpret_cast<int4*>(partial_O + out_off_15)[0] = reinterpret_cast<int4*>(w4_15)[0];
                }
            }
            int ci_18_1 = tid384_1 + 1920;
            if (ci_18_1 < 4096) {
                int h_16 = ci_18_1 / 64;
                int ch_16 = ci_18_1 - h_16 * 64;
                if (head_base_1 + h_16 < num_heads) {
                    unsigned int w4_16[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_16[(0) + 3]))
                        : "r"(smem_o32_addr + (unsigned int)(h_16 * 1024) + (unsigned int)(ch_16 * 16)));
                    long long out_off_16 = ((long long)(query_idx_1 * num_heads + head_base_1 + h_16) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)(o_chunk_1 * 512) + (long long)(ch_16 * 8);
                    reinterpret_cast<int4*>(partial_O + out_off_16)[0] = reinterpret_cast<int4*>(w4_16)[0];
                }
            }
            int ci_19_1 = tid384_1 + 2304;
            if (ci_19_1 < 4096) {
                int h_17 = ci_19_1 / 64;
                int ch_17 = ci_19_1 - h_17 * 64;
                if (head_base_1 + h_17 < num_heads) {
                    unsigned int w4_17[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_17[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_17[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_17[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_17[(0) + 3]))
                        : "r"(smem_o32_addr + (unsigned int)(h_17 * 1024) + (unsigned int)(ch_17 * 16)));
                    long long out_off_17 = ((long long)(query_idx_1 * num_heads + head_base_1 + h_17) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)(o_chunk_1 * 512) + (long long)(ch_17 * 8);
                    reinterpret_cast<int4*>(partial_O + out_off_17)[0] = reinterpret_cast<int4*>(w4_17)[0];
                }
            }
            int ci_20_1 = tid384_1 + 2688;
            if (ci_20_1 < 4096) {
                int h_18 = ci_20_1 / 64;
                int ch_18 = ci_20_1 - h_18 * 64;
                if (head_base_1 + h_18 < num_heads) {
                    unsigned int w4_18[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_18[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_18[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_18[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_18[(0) + 3]))
                        : "r"(smem_o32_addr + (unsigned int)(h_18 * 1024) + (unsigned int)(ch_18 * 16)));
                    long long out_off_18 = ((long long)(query_idx_1 * num_heads + head_base_1 + h_18) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)(o_chunk_1 * 512) + (long long)(ch_18 * 8);
                    reinterpret_cast<int4*>(partial_O + out_off_18)[0] = reinterpret_cast<int4*>(w4_18)[0];
                }
            }
            int ci_21_1 = tid384_1 + 3072;
            if (ci_21_1 < 4096) {
                int h_19 = ci_21_1 / 64;
                int ch_19 = ci_21_1 - h_19 * 64;
                if (head_base_1 + h_19 < num_heads) {
                    unsigned int w4_19[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_19[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_19[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_19[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_19[(0) + 3]))
                        : "r"(smem_o32_addr + (unsigned int)(h_19 * 1024) + (unsigned int)(ch_19 * 16)));
                    long long out_off_19 = ((long long)(query_idx_1 * num_heads + head_base_1 + h_19) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)(o_chunk_1 * 512) + (long long)(ch_19 * 8);
                    reinterpret_cast<int4*>(partial_O + out_off_19)[0] = reinterpret_cast<int4*>(w4_19)[0];
                }
            }
            int ci_22_1 = tid384_1 + 3456;
            if (ci_22_1 < 4096) {
                int h_20 = ci_22_1 / 64;
                int ch_20 = ci_22_1 - h_20 * 64;
                if (head_base_1 + h_20 < num_heads) {
                    unsigned int w4_20[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_20[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_20[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_20[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_20[(0) + 3]))
                        : "r"(smem_o32_addr + (unsigned int)(h_20 * 1024) + (unsigned int)(ch_20 * 16)));
                    long long out_off_20 = ((long long)(query_idx_1 * num_heads + head_base_1 + h_20) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)(o_chunk_1 * 512) + (long long)(ch_20 * 8);
                    reinterpret_cast<int4*>(partial_O + out_off_20)[0] = reinterpret_cast<int4*>(w4_20)[0];
                }
            }
            int ci_23_1 = tid384_1 + 3840;
            if (ci_23_1 < 4096) {
                int h_21 = ci_23_1 / 64;
                int ch_21 = ci_23_1 - h_21 * 64;
                if (head_base_1 + h_21 < num_heads) {
                    unsigned int w4_21[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_21[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_21[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_21[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_21[(0) + 3]))
                        : "r"(smem_o32_addr + (unsigned int)(h_21 * 1024) + (unsigned int)(ch_21 * 16)));
                    long long out_off_21 = ((long long)(query_idx_1 * num_heads + head_base_1 + h_21) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)(o_chunk_1 * 512) + (long long)(ch_21 * 8);
                    reinterpret_cast<int4*>(partial_O + out_off_21)[0] = reinterpret_cast<int4*>(w4_21)[0];
                }
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: compute2 ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 144;");
        { // compute2_main
            const int local_warp_2 = warp - 8;
            const int half_2 = local_warp_2 / 2;
            int o_chunk_2 = 0;
            int work_idx_2 = blockIdx.x;
            int head_tile_2 = work_idx_2 % num_head_tiles;
            int split_work_2 = work_idx_2 / num_head_tiles;
            int split_idx_2 = split_work_2 % num_splits;
            int query_idx_2 = split_work_2 / num_splits;
            int head_base_2 = head_tile_2 * 64;
            const int row_2 = local_warp_2 * 32 + lane;
            const int head_2 = row_2 & 63;
            const int tmem_row_origin_2 = local_warp_2 * 32;
            float inv_six_2 = 0.16666666666666666f;
            unsigned int _phase_q_nope_full0_0_2 = 0;
            mbarrier_wait_hint(q_nope_full0_addr, _phase_q_nope_full0_0_2, 10000000);
            _phase_q_nope_full0_0_2 ^= 1;
            unsigned int _phase_q_nope_full1_0_2 = 0;
            mbarrier_wait_hint(q_nope_full1_addr, _phase_q_nope_full1_0_2, 10000000);
            _phase_q_nope_full1_0_2 ^= 1;
            unsigned int _phase_q_nope_full2_0_2 = 0;
            mbarrier_wait_hint(q_nope_full2_addr, _phase_q_nope_full2_0_2, 10000000);
            _phase_q_nope_full2_0_2 ^= 1;
            int q_taddr_2 = taddr + 448 + (unsigned int)(tmem_row_origin_2 << 16);
            const int q_par_2 = local_warp_2 % 2;
            const int q_slot_2 = 4 + local_warp_2 / 2;
            int q_block_live_2 = ((head_base_2 + q_par_2 * 32 < num_heads) ? 1 : 0);
            int q_head_2 = q_par_2 * 32 + lane;
            int exch_lane_2 = smem_p_0_addr + (unsigned int)(lane * 32);
            unsigned int zero8_2[8];
            zero8_2[0] = 0;
            zero8_2[1] = 0;
            zero8_2[2] = 0;
            zero8_2[3] = 0;
            zero8_2[4] = 0;
            zero8_2[5] = 0;
            zero8_2[6] = 0;
            zero8_2[7] = 0;
            int kset_u_2 = ((1) ? q_slot_2 : 6);
            int do_u_3 = ((1) ? 1 : ((q_slot_2 == 0) ? 1 : 0));
            if (do_u_3 != 0) {
                int exch_u_4 = exch_lane_2 + (q_par_2 * 7 + kset_u_2) * 1024;
                if (q_block_live_2 != 0) {
                    int q_row_addr_4 = smem_qstage_addr + (unsigned int)(kset_u_2 * 8192) + (unsigned int)(q_head_2 * 128);
                    unsigned int words_4[8];
                    unsigned int sf_word_4 = 0;
                    unsigned int qa_4[4];
                    unsigned int qb_5[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qa_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_4[(0) + 3]))
                        : "r"(q_row_addr_4 + (0 ^ row_2 % 8) * 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qb_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_5[(0) + 3]))
                        : "r"(q_row_addr_4 + (1 ^ row_2 % 8) * 16));
                    float qv_5[16];
                    qv_5[0] = __uint_as_float(qa_4[0] << 16);
                    qv_5[1] = __uint_as_float(qa_4[0] & 4294901760u);
                    qv_5[8] = __uint_as_float(qb_5[0] << 16);
                    qv_5[9] = __uint_as_float(qb_5[0] & 4294901760u);
                    qv_5[2] = __uint_as_float(qa_4[1] << 16);
                    qv_5[3] = __uint_as_float(qa_4[1] & 4294901760u);
                    qv_5[10] = __uint_as_float(qb_5[1] << 16);
                    qv_5[11] = __uint_as_float(qb_5[1] & 4294901760u);
                    qv_5[4] = __uint_as_float(qa_4[2] << 16);
                    qv_5[5] = __uint_as_float(qa_4[2] & 4294901760u);
                    qv_5[12] = __uint_as_float(qb_5[2] << 16);
                    qv_5[13] = __uint_as_float(qb_5[2] & 4294901760u);
                    qv_5[6] = __uint_as_float(qa_4[3] << 16);
                    qv_5[7] = __uint_as_float(qa_4[3] & 4294901760u);
                    qv_5[14] = __uint_as_float(qb_5[3] << 16);
                    qv_5[15] = __uint_as_float(qb_5[3] & 4294901760u);
                    float m8_5[8];
                    float _fabs_256 = fabsf(qv_5[0]);
                    float _fabs_257 = fabsf(qv_5[1]);
                    float _max_300 = max_noftz(_fabs_256, _fabs_257);
                    m8_5[0] = _max_300;
                    float _fabs_258 = fabsf(qv_5[2]);
                    float _fabs_259 = fabsf(qv_5[3]);
                    float _max_301 = max_noftz(_fabs_258, _fabs_259);
                    m8_5[1] = _max_301;
                    float _fabs_260 = fabsf(qv_5[4]);
                    float _fabs_261 = fabsf(qv_5[5]);
                    float _max_302 = max_noftz(_fabs_260, _fabs_261);
                    m8_5[2] = _max_302;
                    float _fabs_262 = fabsf(qv_5[6]);
                    float _fabs_263 = fabsf(qv_5[7]);
                    float _max_303 = max_noftz(_fabs_262, _fabs_263);
                    m8_5[3] = _max_303;
                    float _fabs_264 = fabsf(qv_5[8]);
                    float _fabs_265 = fabsf(qv_5[9]);
                    float _max_304 = max_noftz(_fabs_264, _fabs_265);
                    m8_5[4] = _max_304;
                    float _fabs_266 = fabsf(qv_5[10]);
                    float _fabs_267 = fabsf(qv_5[11]);
                    float _max_305 = max_noftz(_fabs_266, _fabs_267);
                    m8_5[5] = _max_305;
                    float _fabs_268 = fabsf(qv_5[12]);
                    float _fabs_269 = fabsf(qv_5[13]);
                    float _max_306 = max_noftz(_fabs_268, _fabs_269);
                    m8_5[6] = _max_306;
                    float _fabs_270 = fabsf(qv_5[14]);
                    float _fabs_271 = fabsf(qv_5[15]);
                    float _max_307 = max_noftz(_fabs_270, _fabs_271);
                    m8_5[7] = _max_307;
                    float m4_5[4];
                    float _max_308 = max_noftz(m8_5[0], m8_5[1]);
                    m4_5[0] = _max_308;
                    float _max_309 = max_noftz(m8_5[2], m8_5[3]);
                    m4_5[1] = _max_309;
                    float _max_310 = max_noftz(m8_5[4], m8_5[5]);
                    m4_5[2] = _max_310;
                    float _max_311 = max_noftz(m8_5[6], m8_5[7]);
                    m4_5[3] = _max_311;
                    float _max_312 = max_noftz(m4_5[0], m4_5[1]);
                    float _max_313 = max_noftz(m4_5[2], m4_5[3]);
                    float _max_314 = max_noftz(_max_312, _max_313);
                    float amax_4 = _max_314;
                    float sc_4 = amax_4 * inv_six_2;
                    uint16_t _e4m3x2_f32_392;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_392) : "f"(0.0f), "f"(sc_4));
                    uint16_t sc_pair_4 = _e4m3x2_f32_392;
                    unsigned int sc_byte_4 = (unsigned int)sc_pair_4 & 255;
                    unsigned int sc_exp_4 = sc_byte_4 >> 3 & 15;
                    unsigned int sc_man_4 = sc_byte_4 & 7;
                    float sc_norm_4 = __uint_as_float(sc_exp_4 + 120 << 23 | sc_man_4 << 20);
                    float sc_sub_4 = (float)sc_man_4 * 0.001953125f;
                    float sc_dec_4 = ((sc_exp_4 == 0) ? sc_sub_4 : sc_norm_4);
                    float _rcp_18 = __frcp_rn(sc_dec_4);
                    float inv_4 = ((sc_dec_4 > 0.0f) ? _rcp_18 : 0.0f);
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_0 = {inv_4, inv_4};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_5)[_ls], _scale2_0);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        qv_5[_ls] = qv_5[_ls] * inv_4;
                    }
                    #endif
                    uint32_t _fp4_pair_128;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_128) : "f"(qv_5[0]), "f"(qv_5[1]));
                    uint32_t _fp4_pair_129;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_129) : "f"(qv_5[2]), "f"(qv_5[3]));
                    uint32_t _fp4_pair_130;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_130) : "f"(qv_5[4]), "f"(qv_5[5]));
                    uint32_t _fp4_pair_131;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_131) : "f"(qv_5[6]), "f"(qv_5[7]));
                    uint32_t _fp4_pair_132;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_132) : "f"(qv_5[8]), "f"(qv_5[9]));
                    uint32_t _fp4_pair_133;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_133) : "f"(qv_5[10]), "f"(qv_5[11]));
                    uint32_t _fp4_pair_134;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_134) : "f"(qv_5[12]), "f"(qv_5[13]));
                    uint32_t _fp4_pair_135;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_135) : "f"(qv_5[14]), "f"(qv_5[15]));
                    words_4[0] = _fp4_pair_128 | _fp4_pair_129 << 8 | _fp4_pair_130 << 16 | _fp4_pair_131 << 24;
                    words_4[1] = _fp4_pair_132 | _fp4_pair_133 << 8 | _fp4_pair_134 << 16 | _fp4_pair_135 << 24;
                    sf_word_4 = sf_word_4 | sc_byte_4;
                    unsigned int qa_0_4[4];
                    unsigned int qb_1_4[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qa_0_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0_4[(0) + 3]))
                        : "r"(q_row_addr_4 + (2 ^ row_2 % 8) * 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qb_1_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1_4[(0) + 3]))
                        : "r"(q_row_addr_4 + (3 ^ row_2 % 8) * 16));
                    float qv_2_4[16];
                    qv_2_4[0] = __uint_as_float(qa_0_4[0] << 16);
                    qv_2_4[1] = __uint_as_float(qa_0_4[0] & 4294901760u);
                    qv_2_4[8] = __uint_as_float(qb_1_4[0] << 16);
                    qv_2_4[9] = __uint_as_float(qb_1_4[0] & 4294901760u);
                    qv_2_4[2] = __uint_as_float(qa_0_4[1] << 16);
                    qv_2_4[3] = __uint_as_float(qa_0_4[1] & 4294901760u);
                    qv_2_4[10] = __uint_as_float(qb_1_4[1] << 16);
                    qv_2_4[11] = __uint_as_float(qb_1_4[1] & 4294901760u);
                    qv_2_4[4] = __uint_as_float(qa_0_4[2] << 16);
                    qv_2_4[5] = __uint_as_float(qa_0_4[2] & 4294901760u);
                    qv_2_4[12] = __uint_as_float(qb_1_4[2] << 16);
                    qv_2_4[13] = __uint_as_float(qb_1_4[2] & 4294901760u);
                    qv_2_4[6] = __uint_as_float(qa_0_4[3] << 16);
                    qv_2_4[7] = __uint_as_float(qa_0_4[3] & 4294901760u);
                    qv_2_4[14] = __uint_as_float(qb_1_4[3] << 16);
                    qv_2_4[15] = __uint_as_float(qb_1_4[3] & 4294901760u);
                    float m8_3_4[8];
                    float _fabs_272 = fabsf(qv_2_4[0]);
                    float _fabs_273 = fabsf(qv_2_4[1]);
                    float _max_315 = max_noftz(_fabs_272, _fabs_273);
                    m8_3_4[0] = _max_315;
                    float _fabs_274 = fabsf(qv_2_4[2]);
                    float _fabs_275 = fabsf(qv_2_4[3]);
                    float _max_316 = max_noftz(_fabs_274, _fabs_275);
                    m8_3_4[1] = _max_316;
                    float _fabs_276 = fabsf(qv_2_4[4]);
                    float _fabs_277 = fabsf(qv_2_4[5]);
                    float _max_317 = max_noftz(_fabs_276, _fabs_277);
                    m8_3_4[2] = _max_317;
                    float _fabs_278 = fabsf(qv_2_4[6]);
                    float _fabs_279 = fabsf(qv_2_4[7]);
                    float _max_318 = max_noftz(_fabs_278, _fabs_279);
                    m8_3_4[3] = _max_318;
                    float _fabs_280 = fabsf(qv_2_4[8]);
                    float _fabs_281 = fabsf(qv_2_4[9]);
                    float _max_319 = max_noftz(_fabs_280, _fabs_281);
                    m8_3_4[4] = _max_319;
                    float _fabs_282 = fabsf(qv_2_4[10]);
                    float _fabs_283 = fabsf(qv_2_4[11]);
                    float _max_320 = max_noftz(_fabs_282, _fabs_283);
                    m8_3_4[5] = _max_320;
                    float _fabs_284 = fabsf(qv_2_4[12]);
                    float _fabs_285 = fabsf(qv_2_4[13]);
                    float _max_321 = max_noftz(_fabs_284, _fabs_285);
                    m8_3_4[6] = _max_321;
                    float _fabs_286 = fabsf(qv_2_4[14]);
                    float _fabs_287 = fabsf(qv_2_4[15]);
                    float _max_322 = max_noftz(_fabs_286, _fabs_287);
                    m8_3_4[7] = _max_322;
                    float m4_4_4[4];
                    float _max_323 = max_noftz(m8_3_4[0], m8_3_4[1]);
                    m4_4_4[0] = _max_323;
                    float _max_324 = max_noftz(m8_3_4[2], m8_3_4[3]);
                    m4_4_4[1] = _max_324;
                    float _max_325 = max_noftz(m8_3_4[4], m8_3_4[5]);
                    m4_4_4[2] = _max_325;
                    float _max_326 = max_noftz(m8_3_4[6], m8_3_4[7]);
                    m4_4_4[3] = _max_326;
                    float _max_327 = max_noftz(m4_4_4[0], m4_4_4[1]);
                    float _max_328 = max_noftz(m4_4_4[2], m4_4_4[3]);
                    float _max_329 = max_noftz(_max_327, _max_328);
                    float amax_5_4 = _max_329;
                    float sc_6_4 = amax_5_4 * inv_six_2;
                    uint16_t _e4m3x2_f32_393;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_393) : "f"(0.0f), "f"(sc_6_4));
                    uint16_t sc_pair_7_4 = _e4m3x2_f32_393;
                    unsigned int sc_byte_8_4 = (unsigned int)sc_pair_7_4 & 255;
                    unsigned int sc_exp_9_4 = sc_byte_8_4 >> 3 & 15;
                    unsigned int sc_man_10_4 = sc_byte_8_4 & 7;
                    float sc_norm_11_4 = __uint_as_float(sc_exp_9_4 + 120 << 23 | sc_man_10_4 << 20);
                    float sc_sub_12_4 = (float)sc_man_10_4 * 0.001953125f;
                    float sc_dec_13_4 = ((sc_exp_9_4 == 0) ? sc_sub_12_4 : sc_norm_11_4);
                    float _rcp_19 = __frcp_rn(sc_dec_13_4);
                    float inv_14_4 = ((sc_dec_13_4 > 0.0f) ? _rcp_19 : 0.0f);
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_1 = {inv_14_4, inv_14_4};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_2_4)[_ls], _scale2_1);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        qv_2_4[_ls] = qv_2_4[_ls] * inv_14_4;
                    }
                    #endif
                    uint32_t _fp4_pair_136;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_136) : "f"(qv_2_4[0]), "f"(qv_2_4[1]));
                    uint32_t _fp4_pair_137;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_137) : "f"(qv_2_4[2]), "f"(qv_2_4[3]));
                    uint32_t _fp4_pair_138;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_138) : "f"(qv_2_4[4]), "f"(qv_2_4[5]));
                    uint32_t _fp4_pair_139;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_139) : "f"(qv_2_4[6]), "f"(qv_2_4[7]));
                    uint32_t _fp4_pair_140;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_140) : "f"(qv_2_4[8]), "f"(qv_2_4[9]));
                    uint32_t _fp4_pair_141;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_141) : "f"(qv_2_4[10]), "f"(qv_2_4[11]));
                    uint32_t _fp4_pair_142;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_142) : "f"(qv_2_4[12]), "f"(qv_2_4[13]));
                    uint32_t _fp4_pair_143;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_143) : "f"(qv_2_4[14]), "f"(qv_2_4[15]));
                    words_4[2] = _fp4_pair_136 | _fp4_pair_137 << 8 | _fp4_pair_138 << 16 | _fp4_pair_139 << 24;
                    words_4[3] = _fp4_pair_140 | _fp4_pair_141 << 8 | _fp4_pair_142 << 16 | _fp4_pair_143 << 24;
                    sf_word_4 = sf_word_4 | sc_byte_8_4 << 8;
                    unsigned int qa_15_4[4];
                    unsigned int qb_16_4[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qa_15_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_15_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_15_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_15_4[(0) + 3]))
                        : "r"(q_row_addr_4 + (4 ^ row_2 % 8) * 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qb_16_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_16_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_16_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_16_4[(0) + 3]))
                        : "r"(q_row_addr_4 + (5 ^ row_2 % 8) * 16));
                    float qv_17_4[16];
                    qv_17_4[0] = __uint_as_float(qa_15_4[0] << 16);
                    qv_17_4[1] = __uint_as_float(qa_15_4[0] & 4294901760u);
                    qv_17_4[8] = __uint_as_float(qb_16_4[0] << 16);
                    qv_17_4[9] = __uint_as_float(qb_16_4[0] & 4294901760u);
                    qv_17_4[2] = __uint_as_float(qa_15_4[1] << 16);
                    qv_17_4[3] = __uint_as_float(qa_15_4[1] & 4294901760u);
                    qv_17_4[10] = __uint_as_float(qb_16_4[1] << 16);
                    qv_17_4[11] = __uint_as_float(qb_16_4[1] & 4294901760u);
                    qv_17_4[4] = __uint_as_float(qa_15_4[2] << 16);
                    qv_17_4[5] = __uint_as_float(qa_15_4[2] & 4294901760u);
                    qv_17_4[12] = __uint_as_float(qb_16_4[2] << 16);
                    qv_17_4[13] = __uint_as_float(qb_16_4[2] & 4294901760u);
                    qv_17_4[6] = __uint_as_float(qa_15_4[3] << 16);
                    qv_17_4[7] = __uint_as_float(qa_15_4[3] & 4294901760u);
                    qv_17_4[14] = __uint_as_float(qb_16_4[3] << 16);
                    qv_17_4[15] = __uint_as_float(qb_16_4[3] & 4294901760u);
                    float m8_18_4[8];
                    float _fabs_288 = fabsf(qv_17_4[0]);
                    float _fabs_289 = fabsf(qv_17_4[1]);
                    float _max_330 = max_noftz(_fabs_288, _fabs_289);
                    m8_18_4[0] = _max_330;
                    float _fabs_290 = fabsf(qv_17_4[2]);
                    float _fabs_291 = fabsf(qv_17_4[3]);
                    float _max_331 = max_noftz(_fabs_290, _fabs_291);
                    m8_18_4[1] = _max_331;
                    float _fabs_292 = fabsf(qv_17_4[4]);
                    float _fabs_293 = fabsf(qv_17_4[5]);
                    float _max_332 = max_noftz(_fabs_292, _fabs_293);
                    m8_18_4[2] = _max_332;
                    float _fabs_294 = fabsf(qv_17_4[6]);
                    float _fabs_295 = fabsf(qv_17_4[7]);
                    float _max_333 = max_noftz(_fabs_294, _fabs_295);
                    m8_18_4[3] = _max_333;
                    float _fabs_296 = fabsf(qv_17_4[8]);
                    float _fabs_297 = fabsf(qv_17_4[9]);
                    float _max_334 = max_noftz(_fabs_296, _fabs_297);
                    m8_18_4[4] = _max_334;
                    float _fabs_298 = fabsf(qv_17_4[10]);
                    float _fabs_299 = fabsf(qv_17_4[11]);
                    float _max_335 = max_noftz(_fabs_298, _fabs_299);
                    m8_18_4[5] = _max_335;
                    float _fabs_300 = fabsf(qv_17_4[12]);
                    float _fabs_301 = fabsf(qv_17_4[13]);
                    float _max_336 = max_noftz(_fabs_300, _fabs_301);
                    m8_18_4[6] = _max_336;
                    float _fabs_302 = fabsf(qv_17_4[14]);
                    float _fabs_303 = fabsf(qv_17_4[15]);
                    float _max_337 = max_noftz(_fabs_302, _fabs_303);
                    m8_18_4[7] = _max_337;
                    float m4_19_4[4];
                    float _max_338 = max_noftz(m8_18_4[0], m8_18_4[1]);
                    m4_19_4[0] = _max_338;
                    float _max_339 = max_noftz(m8_18_4[2], m8_18_4[3]);
                    m4_19_4[1] = _max_339;
                    float _max_340 = max_noftz(m8_18_4[4], m8_18_4[5]);
                    m4_19_4[2] = _max_340;
                    float _max_341 = max_noftz(m8_18_4[6], m8_18_4[7]);
                    m4_19_4[3] = _max_341;
                    float _max_342 = max_noftz(m4_19_4[0], m4_19_4[1]);
                    float _max_343 = max_noftz(m4_19_4[2], m4_19_4[3]);
                    float _max_344 = max_noftz(_max_342, _max_343);
                    float amax_20_4 = _max_344;
                    float sc_21_4 = amax_20_4 * inv_six_2;
                    uint16_t _e4m3x2_f32_394;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_394) : "f"(0.0f), "f"(sc_21_4));
                    uint16_t sc_pair_22_4 = _e4m3x2_f32_394;
                    unsigned int sc_byte_23_4 = (unsigned int)sc_pair_22_4 & 255;
                    unsigned int sc_exp_24_4 = sc_byte_23_4 >> 3 & 15;
                    unsigned int sc_man_25_4 = sc_byte_23_4 & 7;
                    float sc_norm_26_4 = __uint_as_float(sc_exp_24_4 + 120 << 23 | sc_man_25_4 << 20);
                    float sc_sub_27_4 = (float)sc_man_25_4 * 0.001953125f;
                    float sc_dec_28_4 = ((sc_exp_24_4 == 0) ? sc_sub_27_4 : sc_norm_26_4);
                    float _rcp_20 = __frcp_rn(sc_dec_28_4);
                    float inv_29_4 = ((sc_dec_28_4 > 0.0f) ? _rcp_20 : 0.0f);
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_2 = {inv_29_4, inv_29_4};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_17_4)[_ls], _scale2_2);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        qv_17_4[_ls] = qv_17_4[_ls] * inv_29_4;
                    }
                    #endif
                    uint32_t _fp4_pair_144;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_144) : "f"(qv_17_4[0]), "f"(qv_17_4[1]));
                    uint32_t _fp4_pair_145;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_145) : "f"(qv_17_4[2]), "f"(qv_17_4[3]));
                    uint32_t _fp4_pair_146;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_146) : "f"(qv_17_4[4]), "f"(qv_17_4[5]));
                    uint32_t _fp4_pair_147;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_147) : "f"(qv_17_4[6]), "f"(qv_17_4[7]));
                    uint32_t _fp4_pair_148;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_148) : "f"(qv_17_4[8]), "f"(qv_17_4[9]));
                    uint32_t _fp4_pair_149;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_149) : "f"(qv_17_4[10]), "f"(qv_17_4[11]));
                    uint32_t _fp4_pair_150;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_150) : "f"(qv_17_4[12]), "f"(qv_17_4[13]));
                    uint32_t _fp4_pair_151;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_151) : "f"(qv_17_4[14]), "f"(qv_17_4[15]));
                    words_4[4] = _fp4_pair_144 | _fp4_pair_145 << 8 | _fp4_pair_146 << 16 | _fp4_pair_147 << 24;
                    words_4[5] = _fp4_pair_148 | _fp4_pair_149 << 8 | _fp4_pair_150 << 16 | _fp4_pair_151 << 24;
                    sf_word_4 = sf_word_4 | sc_byte_23_4 << 16;
                    unsigned int qa_30_4[4];
                    unsigned int qb_31_4[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qa_30_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_30_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_30_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_30_4[(0) + 3]))
                        : "r"(q_row_addr_4 + (6 ^ row_2 % 8) * 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qb_31_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_31_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_31_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_31_4[(0) + 3]))
                        : "r"(q_row_addr_4 + (7 ^ row_2 % 8) * 16));
                    float qv_32_4[16];
                    qv_32_4[0] = __uint_as_float(qa_30_4[0] << 16);
                    qv_32_4[1] = __uint_as_float(qa_30_4[0] & 4294901760u);
                    qv_32_4[8] = __uint_as_float(qb_31_4[0] << 16);
                    qv_32_4[9] = __uint_as_float(qb_31_4[0] & 4294901760u);
                    qv_32_4[2] = __uint_as_float(qa_30_4[1] << 16);
                    qv_32_4[3] = __uint_as_float(qa_30_4[1] & 4294901760u);
                    qv_32_4[10] = __uint_as_float(qb_31_4[1] << 16);
                    qv_32_4[11] = __uint_as_float(qb_31_4[1] & 4294901760u);
                    qv_32_4[4] = __uint_as_float(qa_30_4[2] << 16);
                    qv_32_4[5] = __uint_as_float(qa_30_4[2] & 4294901760u);
                    qv_32_4[12] = __uint_as_float(qb_31_4[2] << 16);
                    qv_32_4[13] = __uint_as_float(qb_31_4[2] & 4294901760u);
                    qv_32_4[6] = __uint_as_float(qa_30_4[3] << 16);
                    qv_32_4[7] = __uint_as_float(qa_30_4[3] & 4294901760u);
                    qv_32_4[14] = __uint_as_float(qb_31_4[3] << 16);
                    qv_32_4[15] = __uint_as_float(qb_31_4[3] & 4294901760u);
                    float m8_33_4[8];
                    float _fabs_304 = fabsf(qv_32_4[0]);
                    float _fabs_305 = fabsf(qv_32_4[1]);
                    float _max_345 = max_noftz(_fabs_304, _fabs_305);
                    m8_33_4[0] = _max_345;
                    float _fabs_306 = fabsf(qv_32_4[2]);
                    float _fabs_307 = fabsf(qv_32_4[3]);
                    float _max_346 = max_noftz(_fabs_306, _fabs_307);
                    m8_33_4[1] = _max_346;
                    float _fabs_308 = fabsf(qv_32_4[4]);
                    float _fabs_309 = fabsf(qv_32_4[5]);
                    float _max_347 = max_noftz(_fabs_308, _fabs_309);
                    m8_33_4[2] = _max_347;
                    float _fabs_310 = fabsf(qv_32_4[6]);
                    float _fabs_311 = fabsf(qv_32_4[7]);
                    float _max_348 = max_noftz(_fabs_310, _fabs_311);
                    m8_33_4[3] = _max_348;
                    float _fabs_312 = fabsf(qv_32_4[8]);
                    float _fabs_313 = fabsf(qv_32_4[9]);
                    float _max_349 = max_noftz(_fabs_312, _fabs_313);
                    m8_33_4[4] = _max_349;
                    float _fabs_314 = fabsf(qv_32_4[10]);
                    float _fabs_315 = fabsf(qv_32_4[11]);
                    float _max_350 = max_noftz(_fabs_314, _fabs_315);
                    m8_33_4[5] = _max_350;
                    float _fabs_316 = fabsf(qv_32_4[12]);
                    float _fabs_317 = fabsf(qv_32_4[13]);
                    float _max_351 = max_noftz(_fabs_316, _fabs_317);
                    m8_33_4[6] = _max_351;
                    float _fabs_318 = fabsf(qv_32_4[14]);
                    float _fabs_319 = fabsf(qv_32_4[15]);
                    float _max_352 = max_noftz(_fabs_318, _fabs_319);
                    m8_33_4[7] = _max_352;
                    float m4_34_4[4];
                    float _max_353 = max_noftz(m8_33_4[0], m8_33_4[1]);
                    m4_34_4[0] = _max_353;
                    float _max_354 = max_noftz(m8_33_4[2], m8_33_4[3]);
                    m4_34_4[1] = _max_354;
                    float _max_355 = max_noftz(m8_33_4[4], m8_33_4[5]);
                    m4_34_4[2] = _max_355;
                    float _max_356 = max_noftz(m8_33_4[6], m8_33_4[7]);
                    m4_34_4[3] = _max_356;
                    float _max_357 = max_noftz(m4_34_4[0], m4_34_4[1]);
                    float _max_358 = max_noftz(m4_34_4[2], m4_34_4[3]);
                    float _max_359 = max_noftz(_max_357, _max_358);
                    float amax_35_4 = _max_359;
                    float sc_36_4 = amax_35_4 * inv_six_2;
                    uint16_t _e4m3x2_f32_395;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_395) : "f"(0.0f), "f"(sc_36_4));
                    uint16_t sc_pair_37_4 = _e4m3x2_f32_395;
                    unsigned int sc_byte_38_4 = (unsigned int)sc_pair_37_4 & 255;
                    unsigned int sc_exp_39_4 = sc_byte_38_4 >> 3 & 15;
                    unsigned int sc_man_40_4 = sc_byte_38_4 & 7;
                    float sc_norm_41_4 = __uint_as_float(sc_exp_39_4 + 120 << 23 | sc_man_40_4 << 20);
                    float sc_sub_42_4 = (float)sc_man_40_4 * 0.001953125f;
                    float sc_dec_43_4 = ((sc_exp_39_4 == 0) ? sc_sub_42_4 : sc_norm_41_4);
                    float _rcp_21 = __frcp_rn(sc_dec_43_4);
                    float inv_44_4 = ((sc_dec_43_4 > 0.0f) ? _rcp_21 : 0.0f);
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_3 = {inv_44_4, inv_44_4};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_32_4)[_ls], _scale2_3);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        qv_32_4[_ls] = qv_32_4[_ls] * inv_44_4;
                    }
                    #endif
                    uint32_t _fp4_pair_152;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_152) : "f"(qv_32_4[0]), "f"(qv_32_4[1]));
                    uint32_t _fp4_pair_153;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_153) : "f"(qv_32_4[2]), "f"(qv_32_4[3]));
                    uint32_t _fp4_pair_154;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_154) : "f"(qv_32_4[4]), "f"(qv_32_4[5]));
                    uint32_t _fp4_pair_155;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_155) : "f"(qv_32_4[6]), "f"(qv_32_4[7]));
                    uint32_t _fp4_pair_156;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_156) : "f"(qv_32_4[8]), "f"(qv_32_4[9]));
                    uint32_t _fp4_pair_157;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_157) : "f"(qv_32_4[10]), "f"(qv_32_4[11]));
                    uint32_t _fp4_pair_158;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_158) : "f"(qv_32_4[12]), "f"(qv_32_4[13]));
                    uint32_t _fp4_pair_159;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_159) : "f"(qv_32_4[14]), "f"(qv_32_4[15]));
                    words_4[6] = _fp4_pair_152 | _fp4_pair_153 << 8 | _fp4_pair_154 << 16 | _fp4_pair_155 << 24;
                    words_4[7] = _fp4_pair_156 | _fp4_pair_157 << 8 | _fp4_pair_158 << 16 | _fp4_pair_159 << 24;
                    sf_word_4 = sf_word_4 | sc_byte_38_4 << 24;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(exch_u_4), "r"(*reinterpret_cast<uint32_t*>(&words_4[0])), "r"(*reinterpret_cast<uint32_t*>(&words_4[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words_4[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words_4[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(exch_u_4 + 16), "r"(*reinterpret_cast<uint32_t*>(&words_4[4])), "r"(*reinterpret_cast<uint32_t*>(&words_4[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words_4[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words_4[(4) + 3])));
                    smem_qsf32[kset_u_2 / 4 * 2048 + row_2 % 32 / 8 * 512 + kset_u_2 % 4 * 128 + row_2 % 8 * 16 + row_2 / 32 % 4 * 4 >> 2] = sf_word_4;
                    smem_qsf32[kset_u_2 / 4 * 2048 + (row_2 ^ 64) % 32 / 8 * 512 + kset_u_2 % 4 * 128 + (row_2 ^ 64) % 8 * 16 + (row_2 ^ 64) / 32 % 4 * 4 >> 2] = sf_word_4;
                } else {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(exch_u_4), "r"(*reinterpret_cast<uint32_t*>(&zero8_2[0])), "r"(*reinterpret_cast<uint32_t*>(&zero8_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero8_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero8_2[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(exch_u_4 + 16), "r"(*reinterpret_cast<uint32_t*>(&zero8_2[4])), "r"(*reinterpret_cast<uint32_t*>(&zero8_2[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero8_2[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero8_2[(4) + 3])));
                    smem_qsf32[kset_u_2 / 4 * 2048 + row_2 % 32 / 8 * 512 + kset_u_2 % 4 * 128 + row_2 % 8 * 16 + row_2 / 32 % 4 * 4 >> 2] = 0;
                    smem_qsf32[kset_u_2 / 4 * 2048 + (row_2 ^ 64) % 32 / 8 * 512 + kset_u_2 % 4 * 128 + (row_2 ^ 64) % 8 * 16 + (row_2 ^ 64) / 32 % 4 * 4 >> 2] = 0;
                }
            }
            int kset_u_0_2 = ((0) ? q_slot_2 : 6);
            int do_u_1_2 = ((0) ? 1 : ((q_slot_2 == 0) ? 1 : 0));
            if (do_u_1_2 != 0) {
                int exch_u_5 = exch_lane_2 + (q_par_2 * 7 + kset_u_0_2) * 1024;
                if (q_block_live_2 != 0) {
                    int q_row_addr_5 = smem_qstage_addr + (unsigned int)(kset_u_0_2 * 8192) + (unsigned int)(q_head_2 * 128);
                    unsigned int words_5[8];
                    unsigned int sf_word_5 = 0;
                    unsigned int qa_5[4];
                    unsigned int qb_6[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qa_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_5[(0) + 3]))
                        : "r"(q_row_addr_5 + (0 ^ row_2 % 8) * 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qb_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_6[(0) + 3]))
                        : "r"(q_row_addr_5 + (1 ^ row_2 % 8) * 16));
                    float qv_6[16];
                    qv_6[0] = __uint_as_float(qa_5[0] << 16);
                    qv_6[1] = __uint_as_float(qa_5[0] & 4294901760u);
                    qv_6[8] = __uint_as_float(qb_6[0] << 16);
                    qv_6[9] = __uint_as_float(qb_6[0] & 4294901760u);
                    qv_6[2] = __uint_as_float(qa_5[1] << 16);
                    qv_6[3] = __uint_as_float(qa_5[1] & 4294901760u);
                    qv_6[10] = __uint_as_float(qb_6[1] << 16);
                    qv_6[11] = __uint_as_float(qb_6[1] & 4294901760u);
                    qv_6[4] = __uint_as_float(qa_5[2] << 16);
                    qv_6[5] = __uint_as_float(qa_5[2] & 4294901760u);
                    qv_6[12] = __uint_as_float(qb_6[2] << 16);
                    qv_6[13] = __uint_as_float(qb_6[2] & 4294901760u);
                    qv_6[6] = __uint_as_float(qa_5[3] << 16);
                    qv_6[7] = __uint_as_float(qa_5[3] & 4294901760u);
                    qv_6[14] = __uint_as_float(qb_6[3] << 16);
                    qv_6[15] = __uint_as_float(qb_6[3] & 4294901760u);
                    float m8_6[8];
                    float _fabs_320 = fabsf(qv_6[0]);
                    float _fabs_321 = fabsf(qv_6[1]);
                    float _max_360 = max_noftz(_fabs_320, _fabs_321);
                    m8_6[0] = _max_360;
                    float _fabs_322 = fabsf(qv_6[2]);
                    float _fabs_323 = fabsf(qv_6[3]);
                    float _max_361 = max_noftz(_fabs_322, _fabs_323);
                    m8_6[1] = _max_361;
                    float _fabs_324 = fabsf(qv_6[4]);
                    float _fabs_325 = fabsf(qv_6[5]);
                    float _max_362 = max_noftz(_fabs_324, _fabs_325);
                    m8_6[2] = _max_362;
                    float _fabs_326 = fabsf(qv_6[6]);
                    float _fabs_327 = fabsf(qv_6[7]);
                    float _max_363 = max_noftz(_fabs_326, _fabs_327);
                    m8_6[3] = _max_363;
                    float _fabs_328 = fabsf(qv_6[8]);
                    float _fabs_329 = fabsf(qv_6[9]);
                    float _max_364 = max_noftz(_fabs_328, _fabs_329);
                    m8_6[4] = _max_364;
                    float _fabs_330 = fabsf(qv_6[10]);
                    float _fabs_331 = fabsf(qv_6[11]);
                    float _max_365 = max_noftz(_fabs_330, _fabs_331);
                    m8_6[5] = _max_365;
                    float _fabs_332 = fabsf(qv_6[12]);
                    float _fabs_333 = fabsf(qv_6[13]);
                    float _max_366 = max_noftz(_fabs_332, _fabs_333);
                    m8_6[6] = _max_366;
                    float _fabs_334 = fabsf(qv_6[14]);
                    float _fabs_335 = fabsf(qv_6[15]);
                    float _max_367 = max_noftz(_fabs_334, _fabs_335);
                    m8_6[7] = _max_367;
                    float m4_6[4];
                    float _max_368 = max_noftz(m8_6[0], m8_6[1]);
                    m4_6[0] = _max_368;
                    float _max_369 = max_noftz(m8_6[2], m8_6[3]);
                    m4_6[1] = _max_369;
                    float _max_370 = max_noftz(m8_6[4], m8_6[5]);
                    m4_6[2] = _max_370;
                    float _max_371 = max_noftz(m8_6[6], m8_6[7]);
                    m4_6[3] = _max_371;
                    float _max_372 = max_noftz(m4_6[0], m4_6[1]);
                    float _max_373 = max_noftz(m4_6[2], m4_6[3]);
                    float _max_374 = max_noftz(_max_372, _max_373);
                    float amax_6 = _max_374;
                    float sc_5 = amax_6 * inv_six_2;
                    uint16_t _e4m3x2_f32_396;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_396) : "f"(0.0f), "f"(sc_5));
                    uint16_t sc_pair_5 = _e4m3x2_f32_396;
                    unsigned int sc_byte_5 = (unsigned int)sc_pair_5 & 255;
                    unsigned int sc_exp_5 = sc_byte_5 >> 3 & 15;
                    unsigned int sc_man_5 = sc_byte_5 & 7;
                    float sc_norm_5 = __uint_as_float(sc_exp_5 + 120 << 23 | sc_man_5 << 20);
                    float sc_sub_5 = (float)sc_man_5 * 0.001953125f;
                    float sc_dec_5 = ((sc_exp_5 == 0) ? sc_sub_5 : sc_norm_5);
                    float _rcp_22 = __frcp_rn(sc_dec_5);
                    float inv_5 = ((sc_dec_5 > 0.0f) ? _rcp_22 : 0.0f);
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_4 = {inv_5, inv_5};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_6)[_ls], _scale2_4);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        qv_6[_ls] = qv_6[_ls] * inv_5;
                    }
                    #endif
                    uint32_t _fp4_pair_160;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_160) : "f"(qv_6[0]), "f"(qv_6[1]));
                    uint32_t _fp4_pair_161;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_161) : "f"(qv_6[2]), "f"(qv_6[3]));
                    uint32_t _fp4_pair_162;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_162) : "f"(qv_6[4]), "f"(qv_6[5]));
                    uint32_t _fp4_pair_163;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_163) : "f"(qv_6[6]), "f"(qv_6[7]));
                    uint32_t _fp4_pair_164;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_164) : "f"(qv_6[8]), "f"(qv_6[9]));
                    uint32_t _fp4_pair_165;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_165) : "f"(qv_6[10]), "f"(qv_6[11]));
                    uint32_t _fp4_pair_166;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_166) : "f"(qv_6[12]), "f"(qv_6[13]));
                    uint32_t _fp4_pair_167;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_167) : "f"(qv_6[14]), "f"(qv_6[15]));
                    words_5[0] = _fp4_pair_160 | _fp4_pair_161 << 8 | _fp4_pair_162 << 16 | _fp4_pair_163 << 24;
                    words_5[1] = _fp4_pair_164 | _fp4_pair_165 << 8 | _fp4_pair_166 << 16 | _fp4_pair_167 << 24;
                    sf_word_5 = sf_word_5 | sc_byte_5;
                    unsigned int qa_0_5[4];
                    unsigned int qb_1_5[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qa_0_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0_5[(0) + 3]))
                        : "r"(q_row_addr_5 + (2 ^ row_2 % 8) * 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qb_1_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1_5[(0) + 3]))
                        : "r"(q_row_addr_5 + (3 ^ row_2 % 8) * 16));
                    float qv_2_5[16];
                    qv_2_5[0] = __uint_as_float(qa_0_5[0] << 16);
                    qv_2_5[1] = __uint_as_float(qa_0_5[0] & 4294901760u);
                    qv_2_5[8] = __uint_as_float(qb_1_5[0] << 16);
                    qv_2_5[9] = __uint_as_float(qb_1_5[0] & 4294901760u);
                    qv_2_5[2] = __uint_as_float(qa_0_5[1] << 16);
                    qv_2_5[3] = __uint_as_float(qa_0_5[1] & 4294901760u);
                    qv_2_5[10] = __uint_as_float(qb_1_5[1] << 16);
                    qv_2_5[11] = __uint_as_float(qb_1_5[1] & 4294901760u);
                    qv_2_5[4] = __uint_as_float(qa_0_5[2] << 16);
                    qv_2_5[5] = __uint_as_float(qa_0_5[2] & 4294901760u);
                    qv_2_5[12] = __uint_as_float(qb_1_5[2] << 16);
                    qv_2_5[13] = __uint_as_float(qb_1_5[2] & 4294901760u);
                    qv_2_5[6] = __uint_as_float(qa_0_5[3] << 16);
                    qv_2_5[7] = __uint_as_float(qa_0_5[3] & 4294901760u);
                    qv_2_5[14] = __uint_as_float(qb_1_5[3] << 16);
                    qv_2_5[15] = __uint_as_float(qb_1_5[3] & 4294901760u);
                    float m8_3_5[8];
                    float _fabs_336 = fabsf(qv_2_5[0]);
                    float _fabs_337 = fabsf(qv_2_5[1]);
                    float _max_375 = max_noftz(_fabs_336, _fabs_337);
                    m8_3_5[0] = _max_375;
                    float _fabs_338 = fabsf(qv_2_5[2]);
                    float _fabs_339 = fabsf(qv_2_5[3]);
                    float _max_376 = max_noftz(_fabs_338, _fabs_339);
                    m8_3_5[1] = _max_376;
                    float _fabs_340 = fabsf(qv_2_5[4]);
                    float _fabs_341 = fabsf(qv_2_5[5]);
                    float _max_377 = max_noftz(_fabs_340, _fabs_341);
                    m8_3_5[2] = _max_377;
                    float _fabs_342 = fabsf(qv_2_5[6]);
                    float _fabs_343 = fabsf(qv_2_5[7]);
                    float _max_378 = max_noftz(_fabs_342, _fabs_343);
                    m8_3_5[3] = _max_378;
                    float _fabs_344 = fabsf(qv_2_5[8]);
                    float _fabs_345 = fabsf(qv_2_5[9]);
                    float _max_379 = max_noftz(_fabs_344, _fabs_345);
                    m8_3_5[4] = _max_379;
                    float _fabs_346 = fabsf(qv_2_5[10]);
                    float _fabs_347 = fabsf(qv_2_5[11]);
                    float _max_380 = max_noftz(_fabs_346, _fabs_347);
                    m8_3_5[5] = _max_380;
                    float _fabs_348 = fabsf(qv_2_5[12]);
                    float _fabs_349 = fabsf(qv_2_5[13]);
                    float _max_381 = max_noftz(_fabs_348, _fabs_349);
                    m8_3_5[6] = _max_381;
                    float _fabs_350 = fabsf(qv_2_5[14]);
                    float _fabs_351 = fabsf(qv_2_5[15]);
                    float _max_382 = max_noftz(_fabs_350, _fabs_351);
                    m8_3_5[7] = _max_382;
                    float m4_4_5[4];
                    float _max_383 = max_noftz(m8_3_5[0], m8_3_5[1]);
                    m4_4_5[0] = _max_383;
                    float _max_384 = max_noftz(m8_3_5[2], m8_3_5[3]);
                    m4_4_5[1] = _max_384;
                    float _max_385 = max_noftz(m8_3_5[4], m8_3_5[5]);
                    m4_4_5[2] = _max_385;
                    float _max_386 = max_noftz(m8_3_5[6], m8_3_5[7]);
                    m4_4_5[3] = _max_386;
                    float _max_387 = max_noftz(m4_4_5[0], m4_4_5[1]);
                    float _max_388 = max_noftz(m4_4_5[2], m4_4_5[3]);
                    float _max_389 = max_noftz(_max_387, _max_388);
                    float amax_5_5 = _max_389;
                    float sc_6_5 = amax_5_5 * inv_six_2;
                    uint16_t _e4m3x2_f32_397;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_397) : "f"(0.0f), "f"(sc_6_5));
                    uint16_t sc_pair_7_5 = _e4m3x2_f32_397;
                    unsigned int sc_byte_8_5 = (unsigned int)sc_pair_7_5 & 255;
                    unsigned int sc_exp_9_5 = sc_byte_8_5 >> 3 & 15;
                    unsigned int sc_man_10_5 = sc_byte_8_5 & 7;
                    float sc_norm_11_5 = __uint_as_float(sc_exp_9_5 + 120 << 23 | sc_man_10_5 << 20);
                    float sc_sub_12_5 = (float)sc_man_10_5 * 0.001953125f;
                    float sc_dec_13_5 = ((sc_exp_9_5 == 0) ? sc_sub_12_5 : sc_norm_11_5);
                    float _rcp_23 = __frcp_rn(sc_dec_13_5);
                    float inv_14_5 = ((sc_dec_13_5 > 0.0f) ? _rcp_23 : 0.0f);
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_5 = {inv_14_5, inv_14_5};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_2_5)[_ls], _scale2_5);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        qv_2_5[_ls] = qv_2_5[_ls] * inv_14_5;
                    }
                    #endif
                    uint32_t _fp4_pair_168;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_168) : "f"(qv_2_5[0]), "f"(qv_2_5[1]));
                    uint32_t _fp4_pair_169;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_169) : "f"(qv_2_5[2]), "f"(qv_2_5[3]));
                    uint32_t _fp4_pair_170;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_170) : "f"(qv_2_5[4]), "f"(qv_2_5[5]));
                    uint32_t _fp4_pair_171;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_171) : "f"(qv_2_5[6]), "f"(qv_2_5[7]));
                    uint32_t _fp4_pair_172;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_172) : "f"(qv_2_5[8]), "f"(qv_2_5[9]));
                    uint32_t _fp4_pair_173;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_173) : "f"(qv_2_5[10]), "f"(qv_2_5[11]));
                    uint32_t _fp4_pair_174;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_174) : "f"(qv_2_5[12]), "f"(qv_2_5[13]));
                    uint32_t _fp4_pair_175;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_175) : "f"(qv_2_5[14]), "f"(qv_2_5[15]));
                    words_5[2] = _fp4_pair_168 | _fp4_pair_169 << 8 | _fp4_pair_170 << 16 | _fp4_pair_171 << 24;
                    words_5[3] = _fp4_pair_172 | _fp4_pair_173 << 8 | _fp4_pair_174 << 16 | _fp4_pair_175 << 24;
                    sf_word_5 = sf_word_5 | sc_byte_8_5 << 8;
                    unsigned int qa_15_5[4];
                    unsigned int qb_16_5[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qa_15_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_15_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_15_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_15_5[(0) + 3]))
                        : "r"(q_row_addr_5 + (4 ^ row_2 % 8) * 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qb_16_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_16_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_16_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_16_5[(0) + 3]))
                        : "r"(q_row_addr_5 + (5 ^ row_2 % 8) * 16));
                    float qv_17_5[16];
                    qv_17_5[0] = __uint_as_float(qa_15_5[0] << 16);
                    qv_17_5[1] = __uint_as_float(qa_15_5[0] & 4294901760u);
                    qv_17_5[8] = __uint_as_float(qb_16_5[0] << 16);
                    qv_17_5[9] = __uint_as_float(qb_16_5[0] & 4294901760u);
                    qv_17_5[2] = __uint_as_float(qa_15_5[1] << 16);
                    qv_17_5[3] = __uint_as_float(qa_15_5[1] & 4294901760u);
                    qv_17_5[10] = __uint_as_float(qb_16_5[1] << 16);
                    qv_17_5[11] = __uint_as_float(qb_16_5[1] & 4294901760u);
                    qv_17_5[4] = __uint_as_float(qa_15_5[2] << 16);
                    qv_17_5[5] = __uint_as_float(qa_15_5[2] & 4294901760u);
                    qv_17_5[12] = __uint_as_float(qb_16_5[2] << 16);
                    qv_17_5[13] = __uint_as_float(qb_16_5[2] & 4294901760u);
                    qv_17_5[6] = __uint_as_float(qa_15_5[3] << 16);
                    qv_17_5[7] = __uint_as_float(qa_15_5[3] & 4294901760u);
                    qv_17_5[14] = __uint_as_float(qb_16_5[3] << 16);
                    qv_17_5[15] = __uint_as_float(qb_16_5[3] & 4294901760u);
                    float m8_18_5[8];
                    float _fabs_352 = fabsf(qv_17_5[0]);
                    float _fabs_353 = fabsf(qv_17_5[1]);
                    float _max_390 = max_noftz(_fabs_352, _fabs_353);
                    m8_18_5[0] = _max_390;
                    float _fabs_354 = fabsf(qv_17_5[2]);
                    float _fabs_355 = fabsf(qv_17_5[3]);
                    float _max_391 = max_noftz(_fabs_354, _fabs_355);
                    m8_18_5[1] = _max_391;
                    float _fabs_356 = fabsf(qv_17_5[4]);
                    float _fabs_357 = fabsf(qv_17_5[5]);
                    float _max_392 = max_noftz(_fabs_356, _fabs_357);
                    m8_18_5[2] = _max_392;
                    float _fabs_358 = fabsf(qv_17_5[6]);
                    float _fabs_359 = fabsf(qv_17_5[7]);
                    float _max_393 = max_noftz(_fabs_358, _fabs_359);
                    m8_18_5[3] = _max_393;
                    float _fabs_360 = fabsf(qv_17_5[8]);
                    float _fabs_361 = fabsf(qv_17_5[9]);
                    float _max_394 = max_noftz(_fabs_360, _fabs_361);
                    m8_18_5[4] = _max_394;
                    float _fabs_362 = fabsf(qv_17_5[10]);
                    float _fabs_363 = fabsf(qv_17_5[11]);
                    float _max_395 = max_noftz(_fabs_362, _fabs_363);
                    m8_18_5[5] = _max_395;
                    float _fabs_364 = fabsf(qv_17_5[12]);
                    float _fabs_365 = fabsf(qv_17_5[13]);
                    float _max_396 = max_noftz(_fabs_364, _fabs_365);
                    m8_18_5[6] = _max_396;
                    float _fabs_366 = fabsf(qv_17_5[14]);
                    float _fabs_367 = fabsf(qv_17_5[15]);
                    float _max_397 = max_noftz(_fabs_366, _fabs_367);
                    m8_18_5[7] = _max_397;
                    float m4_19_5[4];
                    float _max_398 = max_noftz(m8_18_5[0], m8_18_5[1]);
                    m4_19_5[0] = _max_398;
                    float _max_399 = max_noftz(m8_18_5[2], m8_18_5[3]);
                    m4_19_5[1] = _max_399;
                    float _max_400 = max_noftz(m8_18_5[4], m8_18_5[5]);
                    m4_19_5[2] = _max_400;
                    float _max_401 = max_noftz(m8_18_5[6], m8_18_5[7]);
                    m4_19_5[3] = _max_401;
                    float _max_402 = max_noftz(m4_19_5[0], m4_19_5[1]);
                    float _max_403 = max_noftz(m4_19_5[2], m4_19_5[3]);
                    float _max_404 = max_noftz(_max_402, _max_403);
                    float amax_20_5 = _max_404;
                    float sc_21_5 = amax_20_5 * inv_six_2;
                    uint16_t _e4m3x2_f32_398;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_398) : "f"(0.0f), "f"(sc_21_5));
                    uint16_t sc_pair_22_5 = _e4m3x2_f32_398;
                    unsigned int sc_byte_23_5 = (unsigned int)sc_pair_22_5 & 255;
                    unsigned int sc_exp_24_5 = sc_byte_23_5 >> 3 & 15;
                    unsigned int sc_man_25_5 = sc_byte_23_5 & 7;
                    float sc_norm_26_5 = __uint_as_float(sc_exp_24_5 + 120 << 23 | sc_man_25_5 << 20);
                    float sc_sub_27_5 = (float)sc_man_25_5 * 0.001953125f;
                    float sc_dec_28_5 = ((sc_exp_24_5 == 0) ? sc_sub_27_5 : sc_norm_26_5);
                    float _rcp_24 = __frcp_rn(sc_dec_28_5);
                    float inv_29_5 = ((sc_dec_28_5 > 0.0f) ? _rcp_24 : 0.0f);
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_6 = {inv_29_5, inv_29_5};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_17_5)[_ls], _scale2_6);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        qv_17_5[_ls] = qv_17_5[_ls] * inv_29_5;
                    }
                    #endif
                    uint32_t _fp4_pair_176;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_176) : "f"(qv_17_5[0]), "f"(qv_17_5[1]));
                    uint32_t _fp4_pair_177;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_177) : "f"(qv_17_5[2]), "f"(qv_17_5[3]));
                    uint32_t _fp4_pair_178;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_178) : "f"(qv_17_5[4]), "f"(qv_17_5[5]));
                    uint32_t _fp4_pair_179;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_179) : "f"(qv_17_5[6]), "f"(qv_17_5[7]));
                    uint32_t _fp4_pair_180;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_180) : "f"(qv_17_5[8]), "f"(qv_17_5[9]));
                    uint32_t _fp4_pair_181;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_181) : "f"(qv_17_5[10]), "f"(qv_17_5[11]));
                    uint32_t _fp4_pair_182;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_182) : "f"(qv_17_5[12]), "f"(qv_17_5[13]));
                    uint32_t _fp4_pair_183;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_183) : "f"(qv_17_5[14]), "f"(qv_17_5[15]));
                    words_5[4] = _fp4_pair_176 | _fp4_pair_177 << 8 | _fp4_pair_178 << 16 | _fp4_pair_179 << 24;
                    words_5[5] = _fp4_pair_180 | _fp4_pair_181 << 8 | _fp4_pair_182 << 16 | _fp4_pair_183 << 24;
                    sf_word_5 = sf_word_5 | sc_byte_23_5 << 16;
                    unsigned int qa_30_5[4];
                    unsigned int qb_31_5[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qa_30_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_30_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_30_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_30_5[(0) + 3]))
                        : "r"(q_row_addr_5 + (6 ^ row_2 % 8) * 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qb_31_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_31_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_31_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_31_5[(0) + 3]))
                        : "r"(q_row_addr_5 + (7 ^ row_2 % 8) * 16));
                    float qv_32_5[16];
                    qv_32_5[0] = __uint_as_float(qa_30_5[0] << 16);
                    qv_32_5[1] = __uint_as_float(qa_30_5[0] & 4294901760u);
                    qv_32_5[8] = __uint_as_float(qb_31_5[0] << 16);
                    qv_32_5[9] = __uint_as_float(qb_31_5[0] & 4294901760u);
                    qv_32_5[2] = __uint_as_float(qa_30_5[1] << 16);
                    qv_32_5[3] = __uint_as_float(qa_30_5[1] & 4294901760u);
                    qv_32_5[10] = __uint_as_float(qb_31_5[1] << 16);
                    qv_32_5[11] = __uint_as_float(qb_31_5[1] & 4294901760u);
                    qv_32_5[4] = __uint_as_float(qa_30_5[2] << 16);
                    qv_32_5[5] = __uint_as_float(qa_30_5[2] & 4294901760u);
                    qv_32_5[12] = __uint_as_float(qb_31_5[2] << 16);
                    qv_32_5[13] = __uint_as_float(qb_31_5[2] & 4294901760u);
                    qv_32_5[6] = __uint_as_float(qa_30_5[3] << 16);
                    qv_32_5[7] = __uint_as_float(qa_30_5[3] & 4294901760u);
                    qv_32_5[14] = __uint_as_float(qb_31_5[3] << 16);
                    qv_32_5[15] = __uint_as_float(qb_31_5[3] & 4294901760u);
                    float m8_33_5[8];
                    float _fabs_368 = fabsf(qv_32_5[0]);
                    float _fabs_369 = fabsf(qv_32_5[1]);
                    float _max_405 = max_noftz(_fabs_368, _fabs_369);
                    m8_33_5[0] = _max_405;
                    float _fabs_370 = fabsf(qv_32_5[2]);
                    float _fabs_371 = fabsf(qv_32_5[3]);
                    float _max_406 = max_noftz(_fabs_370, _fabs_371);
                    m8_33_5[1] = _max_406;
                    float _fabs_372 = fabsf(qv_32_5[4]);
                    float _fabs_373 = fabsf(qv_32_5[5]);
                    float _max_407 = max_noftz(_fabs_372, _fabs_373);
                    m8_33_5[2] = _max_407;
                    float _fabs_374 = fabsf(qv_32_5[6]);
                    float _fabs_375 = fabsf(qv_32_5[7]);
                    float _max_408 = max_noftz(_fabs_374, _fabs_375);
                    m8_33_5[3] = _max_408;
                    float _fabs_376 = fabsf(qv_32_5[8]);
                    float _fabs_377 = fabsf(qv_32_5[9]);
                    float _max_409 = max_noftz(_fabs_376, _fabs_377);
                    m8_33_5[4] = _max_409;
                    float _fabs_378 = fabsf(qv_32_5[10]);
                    float _fabs_379 = fabsf(qv_32_5[11]);
                    float _max_410 = max_noftz(_fabs_378, _fabs_379);
                    m8_33_5[5] = _max_410;
                    float _fabs_380 = fabsf(qv_32_5[12]);
                    float _fabs_381 = fabsf(qv_32_5[13]);
                    float _max_411 = max_noftz(_fabs_380, _fabs_381);
                    m8_33_5[6] = _max_411;
                    float _fabs_382 = fabsf(qv_32_5[14]);
                    float _fabs_383 = fabsf(qv_32_5[15]);
                    float _max_412 = max_noftz(_fabs_382, _fabs_383);
                    m8_33_5[7] = _max_412;
                    float m4_34_5[4];
                    float _max_413 = max_noftz(m8_33_5[0], m8_33_5[1]);
                    m4_34_5[0] = _max_413;
                    float _max_414 = max_noftz(m8_33_5[2], m8_33_5[3]);
                    m4_34_5[1] = _max_414;
                    float _max_415 = max_noftz(m8_33_5[4], m8_33_5[5]);
                    m4_34_5[2] = _max_415;
                    float _max_416 = max_noftz(m8_33_5[6], m8_33_5[7]);
                    m4_34_5[3] = _max_416;
                    float _max_417 = max_noftz(m4_34_5[0], m4_34_5[1]);
                    float _max_418 = max_noftz(m4_34_5[2], m4_34_5[3]);
                    float _max_419 = max_noftz(_max_417, _max_418);
                    float amax_35_5 = _max_419;
                    float sc_36_5 = amax_35_5 * inv_six_2;
                    uint16_t _e4m3x2_f32_399;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_399) : "f"(0.0f), "f"(sc_36_5));
                    uint16_t sc_pair_37_5 = _e4m3x2_f32_399;
                    unsigned int sc_byte_38_5 = (unsigned int)sc_pair_37_5 & 255;
                    unsigned int sc_exp_39_5 = sc_byte_38_5 >> 3 & 15;
                    unsigned int sc_man_40_5 = sc_byte_38_5 & 7;
                    float sc_norm_41_5 = __uint_as_float(sc_exp_39_5 + 120 << 23 | sc_man_40_5 << 20);
                    float sc_sub_42_5 = (float)sc_man_40_5 * 0.001953125f;
                    float sc_dec_43_5 = ((sc_exp_39_5 == 0) ? sc_sub_42_5 : sc_norm_41_5);
                    float _rcp_25 = __frcp_rn(sc_dec_43_5);
                    float inv_44_5 = ((sc_dec_43_5 > 0.0f) ? _rcp_25 : 0.0f);
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_7 = {inv_44_5, inv_44_5};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_32_5)[_ls], _scale2_7);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        qv_32_5[_ls] = qv_32_5[_ls] * inv_44_5;
                    }
                    #endif
                    uint32_t _fp4_pair_184;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_184) : "f"(qv_32_5[0]), "f"(qv_32_5[1]));
                    uint32_t _fp4_pair_185;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_185) : "f"(qv_32_5[2]), "f"(qv_32_5[3]));
                    uint32_t _fp4_pair_186;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_186) : "f"(qv_32_5[4]), "f"(qv_32_5[5]));
                    uint32_t _fp4_pair_187;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_187) : "f"(qv_32_5[6]), "f"(qv_32_5[7]));
                    uint32_t _fp4_pair_188;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_188) : "f"(qv_32_5[8]), "f"(qv_32_5[9]));
                    uint32_t _fp4_pair_189;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_189) : "f"(qv_32_5[10]), "f"(qv_32_5[11]));
                    uint32_t _fp4_pair_190;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_190) : "f"(qv_32_5[12]), "f"(qv_32_5[13]));
                    uint32_t _fp4_pair_191;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_191) : "f"(qv_32_5[14]), "f"(qv_32_5[15]));
                    words_5[6] = _fp4_pair_184 | _fp4_pair_185 << 8 | _fp4_pair_186 << 16 | _fp4_pair_187 << 24;
                    words_5[7] = _fp4_pair_188 | _fp4_pair_189 << 8 | _fp4_pair_190 << 16 | _fp4_pair_191 << 24;
                    sf_word_5 = sf_word_5 | sc_byte_38_5 << 24;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(exch_u_5), "r"(*reinterpret_cast<uint32_t*>(&words_5[0])), "r"(*reinterpret_cast<uint32_t*>(&words_5[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words_5[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words_5[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(exch_u_5 + 16), "r"(*reinterpret_cast<uint32_t*>(&words_5[4])), "r"(*reinterpret_cast<uint32_t*>(&words_5[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words_5[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words_5[(4) + 3])));
                    smem_qsf32[kset_u_0_2 / 4 * 2048 + row_2 % 32 / 8 * 512 + kset_u_0_2 % 4 * 128 + row_2 % 8 * 16 + row_2 / 32 % 4 * 4 >> 2] = sf_word_5;
                    smem_qsf32[kset_u_0_2 / 4 * 2048 + (row_2 ^ 64) % 32 / 8 * 512 + kset_u_0_2 % 4 * 128 + (row_2 ^ 64) % 8 * 16 + (row_2 ^ 64) / 32 % 4 * 4 >> 2] = sf_word_5;
                } else {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(exch_u_5), "r"(*reinterpret_cast<uint32_t*>(&zero8_2[0])), "r"(*reinterpret_cast<uint32_t*>(&zero8_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero8_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero8_2[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(exch_u_5 + 16), "r"(*reinterpret_cast<uint32_t*>(&zero8_2[4])), "r"(*reinterpret_cast<uint32_t*>(&zero8_2[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero8_2[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero8_2[(4) + 3])));
                    smem_qsf32[kset_u_0_2 / 4 * 2048 + row_2 % 32 / 8 * 512 + kset_u_0_2 % 4 * 128 + row_2 % 8 * 16 + row_2 / 32 % 4 * 4 >> 2] = 0;
                    smem_qsf32[kset_u_0_2 / 4 * 2048 + (row_2 ^ 64) % 32 / 8 * 512 + kset_u_0_2 % 4 * 128 + (row_2 ^ 64) % 8 * 16 + (row_2 ^ 64) / 32 % 4 * 4 >> 2] = 0;
                }
            }
            asm volatile("barrier.sync 10, 384;" ::: "memory");
            unsigned int qw_8[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&qw_8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_8[(0) + 3]))
                : "r"(exch_lane_2 + q_par_2 * 7 * 1024));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_8[4])), "=r"(*reinterpret_cast<uint32_t*>(&qw_8[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_8[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_8[(4) + 3]))
                : "r"(exch_lane_2 + q_par_2 * 7 * 1024 + 16));
            tmem_st_x8_u32(q_taddr_2, (const uint32_t*)qw_8);
            unsigned int qw_2_2[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_2_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&qw_2_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_2_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_2_2[(0) + 3]))
                : "r"(exch_lane_2 + (q_par_2 * 7 + 1) * 1024));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_2_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&qw_2_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_2_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_2_2[(4) + 3]))
                : "r"(exch_lane_2 + (q_par_2 * 7 + 1) * 1024 + 16));
            tmem_st_x8_u32(q_taddr_2 + 8, (const uint32_t*)qw_2_2);
            unsigned int qw_3_2[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_3_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&qw_3_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_3_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_3_2[(0) + 3]))
                : "r"(exch_lane_2 + (q_par_2 * 7 + 2) * 1024));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_3_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&qw_3_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_3_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_3_2[(4) + 3]))
                : "r"(exch_lane_2 + (q_par_2 * 7 + 2) * 1024 + 16));
            tmem_st_x8_u32(q_taddr_2 + 16, (const uint32_t*)qw_3_2);
            unsigned int qw_4_2[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_4_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&qw_4_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_4_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_4_2[(0) + 3]))
                : "r"(exch_lane_2 + (q_par_2 * 7 + 3) * 1024));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_4_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&qw_4_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_4_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_4_2[(4) + 3]))
                : "r"(exch_lane_2 + (q_par_2 * 7 + 3) * 1024 + 16));
            tmem_st_x8_u32(q_taddr_2 + 24, (const uint32_t*)qw_4_2);
            unsigned int qw_5_2[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_5_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&qw_5_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_5_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_5_2[(0) + 3]))
                : "r"(exch_lane_2 + (q_par_2 * 7 + 4) * 1024));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_5_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&qw_5_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_5_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_5_2[(4) + 3]))
                : "r"(exch_lane_2 + (q_par_2 * 7 + 4) * 1024 + 16));
            tmem_st_x8_u32(q_taddr_2 + 32, (const uint32_t*)qw_5_2);
            unsigned int qw_6_2[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_6_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&qw_6_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_6_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_6_2[(0) + 3]))
                : "r"(exch_lane_2 + (q_par_2 * 7 + 5) * 1024));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_6_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&qw_6_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_6_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_6_2[(4) + 3]))
                : "r"(exch_lane_2 + (q_par_2 * 7 + 5) * 1024 + 16));
            tmem_st_x8_u32(q_taddr_2 + 40, (const uint32_t*)qw_6_2);
            unsigned int qw_7_2[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_7_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&qw_7_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_7_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_7_2[(0) + 3]))
                : "r"(exch_lane_2 + (q_par_2 * 7 + 6) * 1024));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&qw_7_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&qw_7_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qw_7_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qw_7_2[(4) + 3]))
                : "r"(exch_lane_2 + (q_par_2 * 7 + 6) * 1024 + 16));
            tmem_st_x8_u32(q_taddr_2 + 48, (const uint32_t*)qw_7_2);
            tmem_st_x8_u32(q_taddr_2 + 56, (const uint32_t*)zero8_2);
            smem_qsf32[2048 + row_2 % 32 / 8 * 512 + 384 + row_2 % 8 * 16 + row_2 / 32 % 4 * 4 >> 2] = 0;
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(q_ready_addr);
            asm volatile("barrier.sync 9, 384;" ::: "memory");
            float softmax_scale_log2_2 = bmm1_scale[0] * 1.4426950408889634f;
            float output_scale_2 = bmm2_scale[0];
            float sm_2[3];
            sm_2[0] = -CAKE_INF;
            sm_2[1] = 0.0f;
            sm_2[2] = 0.0f;
            float sink_lane_2 = -CAKE_INF;
            if (has_sinks != 0 && split_idx_2 == 0 && head_base_2 + head_2 < num_heads) {
                sink_lane_2 = sinks[head_base_2 + head_2] * 1.4426950408889634f;
            }
            for (int it_2 = 0; it_2 < tiles_per_split; it_2++) {
                int buf_2 = it_2 & 1;
                int par_2 = it_2 >> 1 & 1;
                int kbase_2 = smem_kf4_0_addr + (unsigned int)(buf_2 * 53248);
                int kz_off_2 = buf_2 * 53248;
                mbarrier_wait_hint(tok_full_addr + (buf_2) * 8, par_2, 10000000);
                int tok_off_2 = buf_2 * 256;
                int pbase_2 = smem_p_0_addr + (unsigned int)(buf_2 * 8192);
                int raw_index_2 = smem_tok32v[tok_off_2 + row_2];
                unsigned int mask_word_2 = (unsigned int)smem_tok32v[tok_off_2 + ((half_2 == 0) ? 129 : 131)];
                int valid_2 = 1;
                if (raw_index_2 < 0) {
                    valid_2 = 0;
                }
                if (valid_2 != 0) {
                } else if (0) {
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(kv_full_addr);
                if (it_2 > 0) {
                    mbarrier_wait_hint(o_full_addr, it_2 - 1 & 1, 10000000);
                    mbarrier_wait_hint(o_full_addr + 8, it_2 - 1 & 1, 10000000);
                    mbarrier_wait_hint(o_full_addr + 16, it_2 - 1 & 1, 10000000);
                    mbarrier_wait_hint(o_full_addr + 24, it_2 - 1 & 1, 10000000);
                }
                if (valid_2 != 0) {
                    unsigned int kraw4_2[4];
                    unsigned int sfw32_2 = 0;
                    int vblock_3 = 32 * o_chunk_2 + 22;
                    unsigned int v8_4[4];
                    {
                        {
                            int vchunk_12 = 16 * o_chunk_2 + 11;
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[(0) + 3]))
                                : "r"(kbase_2 + vchunk_12 / 8 * 16384 + (row_2 * 128 + (vchunk_12 % 8 * 16 ^ row_2 % 8 * 16))));
                        }
                        {
                            sfw32_2 = smem_kz32[(kz_off_2 + 49152 + row_2 * 32 >> 2) + 8 * o_chunk_2 + 5];
                        }
                        unsigned int scale_22 = sfw32_2 >> 16 & 255;
                        {
                            v8_4[0] = cake_dsv4_qmul4<5>(kraw4_2[0], scale_22);
                        }
                        {
                            v8_4[1] = cake_dsv4_qmul4<6>(kraw4_2[0], scale_22);
                        }
                        {
                            v8_4[2] = cake_dsv4_qmul4<5>(kraw4_2[1], scale_22);
                        }
                        {
                            v8_4[3] = cake_dsv4_qmul4<6>(kraw4_2[1], scale_22);
                        }
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(smem_v_addr + (unsigned int)(32768 + (row_2 * 128 + (96 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_4[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_4[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_4[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_4[(0) + 3])));
                    int vblock_0_2 = 32 * o_chunk_2 + 23;
                    unsigned int v8_1_2[4];
                    {
                        unsigned int scale_23 = sfw32_2 >> 24 & 255;
                        {
                            v8_1_2[0] = cake_dsv4_qmul4<5>(kraw4_2[2], scale_23);
                        }
                        {
                            v8_1_2[1] = cake_dsv4_qmul4<6>(kraw4_2[2], scale_23);
                        }
                        {
                            v8_1_2[2] = cake_dsv4_qmul4<5>(kraw4_2[3], scale_23);
                        }
                        {
                            v8_1_2[3] = cake_dsv4_qmul4<6>(kraw4_2[3], scale_23);
                        }
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(smem_v_addr + (unsigned int)(32768 + (row_2 * 128 + (112 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_1_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_1_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_1_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_1_2[(0) + 3])));
                    int vblock_2_2 = 32 * o_chunk_2 + 24;
                    unsigned int v8_3_2[4];
                    {
                        {
                            int vchunk_13 = 16 * o_chunk_2 + 12;
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[(0) + 3]))
                                : "r"(kbase_2 + vchunk_13 / 8 * 16384 + (row_2 * 128 + (vchunk_13 % 8 * 16 ^ row_2 % 8 * 16))));
                        }
                        {
                            sfw32_2 = smem_kz32[(kz_off_2 + 49152 + row_2 * 32 >> 2) + 8 * o_chunk_2 + 6];
                        }
                        unsigned int scale_24 = sfw32_2 & 255;
                        {
                            v8_3_2[0] = cake_dsv4_qmul4<5>(kraw4_2[0], scale_24);
                        }
                        {
                            v8_3_2[1] = cake_dsv4_qmul4<6>(kraw4_2[0], scale_24);
                        }
                        {
                            v8_3_2[2] = cake_dsv4_qmul4<5>(kraw4_2[1], scale_24);
                        }
                        {
                            v8_3_2[3] = cake_dsv4_qmul4<6>(kraw4_2[1], scale_24);
                        }
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (0 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_3_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_3_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_3_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_3_2[(0) + 3])));
                    int vblock_4_2 = 32 * o_chunk_2 + 25;
                    unsigned int v8_5_2[4];
                    {
                        unsigned int scale_25 = sfw32_2 >> 8 & 255;
                        {
                            v8_5_2[0] = cake_dsv4_qmul4<5>(kraw4_2[2], scale_25);
                        }
                        {
                            v8_5_2[1] = cake_dsv4_qmul4<6>(kraw4_2[2], scale_25);
                        }
                        {
                            v8_5_2[2] = cake_dsv4_qmul4<5>(kraw4_2[3], scale_25);
                        }
                        {
                            v8_5_2[3] = cake_dsv4_qmul4<6>(kraw4_2[3], scale_25);
                        }
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (16 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_5_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_5_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_5_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_5_2[(0) + 3])));
                    int vblock_6_2 = 32 * o_chunk_2 + 26;
                    unsigned int v8_7_2[4];
                    {
                        {
                            int vchunk_14 = 16 * o_chunk_2 + 13;
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[(0) + 3]))
                                : "r"(kbase_2 + vchunk_14 / 8 * 16384 + (row_2 * 128 + (vchunk_14 % 8 * 16 ^ row_2 % 8 * 16))));
                        }
                        unsigned int scale_26 = sfw32_2 >> 16 & 255;
                        {
                            v8_7_2[0] = cake_dsv4_qmul4<5>(kraw4_2[0], scale_26);
                        }
                        {
                            v8_7_2[1] = cake_dsv4_qmul4<6>(kraw4_2[0], scale_26);
                        }
                        {
                            v8_7_2[2] = cake_dsv4_qmul4<5>(kraw4_2[1], scale_26);
                        }
                        {
                            v8_7_2[3] = cake_dsv4_qmul4<6>(kraw4_2[1], scale_26);
                        }
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (32 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_7_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_7_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_7_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_7_2[(0) + 3])));
                    int vblock_8_2 = 32 * o_chunk_2 + 27;
                    unsigned int v8_9_2[4];
                    {
                        unsigned int scale_27 = sfw32_2 >> 24 & 255;
                        {
                            v8_9_2[0] = cake_dsv4_qmul4<5>(kraw4_2[2], scale_27);
                        }
                        {
                            v8_9_2[1] = cake_dsv4_qmul4<6>(kraw4_2[2], scale_27);
                        }
                        {
                            v8_9_2[2] = cake_dsv4_qmul4<5>(kraw4_2[3], scale_27);
                        }
                        {
                            v8_9_2[3] = cake_dsv4_qmul4<6>(kraw4_2[3], scale_27);
                        }
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (48 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_9_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_9_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_9_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_9_2[(0) + 3])));
                    int vblock_10_2 = 32 * o_chunk_2 + 28;
                    unsigned int v8_11_2[4];
                    {
                        int rblock = vblock_10_2 - 28;
                        unsigned int rope[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&rope[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope[(0) + 3]))
                            : "r"(kbase_2 + 32768 + (row_2 * 128 + (2 * rblock * 16 ^ row_2 % 8 * 16))));
                        float lo = __uint_as_float(rope[0] << 16);
                        float hi = __uint_as_float(rope[0] & 4294901760u);
                        uint16_t _e4m3x2_f32_496;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_496) : "f"(hi), "f"(lo));
                        uint16_t pair = _e4m3x2_f32_496;
                        {
                            v8_11_2[0] = (unsigned int)pair;
                        }
                        float lo_0 = __uint_as_float(rope[1] << 16);
                        float hi_1 = __uint_as_float(rope[1] & 4294901760u);
                        uint16_t _e4m3x2_f32_497;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_497) : "f"(hi_1), "f"(lo_0));
                        uint16_t pair_2 = _e4m3x2_f32_497;
                        {
                            v8_11_2[0] = v8_11_2[0] | (unsigned int)pair_2 << 16;
                        }
                        float lo_3 = __uint_as_float(rope[2] << 16);
                        float hi_4 = __uint_as_float(rope[2] & 4294901760u);
                        uint16_t _e4m3x2_f32_498;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_498) : "f"(hi_4), "f"(lo_3));
                        uint16_t pair_5 = _e4m3x2_f32_498;
                        {
                            v8_11_2[1] = (unsigned int)pair_5;
                        }
                        float lo_6 = __uint_as_float(rope[3] << 16);
                        float hi_7 = __uint_as_float(rope[3] & 4294901760u);
                        uint16_t _e4m3x2_f32_499;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_499) : "f"(hi_7), "f"(lo_6));
                        uint16_t pair_8 = _e4m3x2_f32_499;
                        {
                            v8_11_2[1] = v8_11_2[1] | (unsigned int)pair_8 << 16;
                        }
                        unsigned int rope_9[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&rope_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9[(0) + 3]))
                            : "r"(kbase_2 + 32768 + (row_2 * 128 + ((2 * rblock + 1) * 16 ^ row_2 % 8 * 16))));
                        float lo_10 = __uint_as_float(rope_9[0] << 16);
                        float hi_11 = __uint_as_float(rope_9[0] & 4294901760u);
                        uint16_t _e4m3x2_f32_500;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_500) : "f"(hi_11), "f"(lo_10));
                        uint16_t pair_12 = _e4m3x2_f32_500;
                        {
                            v8_11_2[2] = (unsigned int)pair_12;
                        }
                        float lo_13 = __uint_as_float(rope_9[1] << 16);
                        float hi_14 = __uint_as_float(rope_9[1] & 4294901760u);
                        uint16_t _e4m3x2_f32_501;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_501) : "f"(hi_14), "f"(lo_13));
                        uint16_t pair_15 = _e4m3x2_f32_501;
                        {
                            v8_11_2[2] = v8_11_2[2] | (unsigned int)pair_15 << 16;
                        }
                        float lo_16 = __uint_as_float(rope_9[2] << 16);
                        float hi_17 = __uint_as_float(rope_9[2] & 4294901760u);
                        uint16_t _e4m3x2_f32_502;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_502) : "f"(hi_17), "f"(lo_16));
                        uint16_t pair_18 = _e4m3x2_f32_502;
                        {
                            v8_11_2[3] = (unsigned int)pair_18;
                        }
                        float lo_19 = __uint_as_float(rope_9[3] << 16);
                        float hi_20 = __uint_as_float(rope_9[3] & 4294901760u);
                        uint16_t _e4m3x2_f32_503;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_503) : "f"(hi_20), "f"(lo_19));
                        uint16_t pair_21 = _e4m3x2_f32_503;
                        {
                            v8_11_2[3] = v8_11_2[3] | (unsigned int)pair_21 << 16;
                        }
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (64 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_11_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_11_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_11_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_11_2[(0) + 3])));
                    int vblock_12_2 = 32 * o_chunk_2 + 29;
                    unsigned int v8_13_2[4];
                    {
                        int rblock_1 = vblock_12_2 - 28;
                        unsigned int rope_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&rope_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_1[(0) + 3]))
                            : "r"(kbase_2 + 32768 + (row_2 * 128 + (2 * rblock_1 * 16 ^ row_2 % 8 * 16))));
                        float lo_1 = __uint_as_float(rope_1[0] << 16);
                        float hi_2 = __uint_as_float(rope_1[0] & 4294901760u);
                        uint16_t _e4m3x2_f32_512;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_512) : "f"(hi_2), "f"(lo_1));
                        uint16_t pair_1 = _e4m3x2_f32_512;
                        {
                            v8_13_2[0] = (unsigned int)pair_1;
                        }
                        float lo_0_1 = __uint_as_float(rope_1[1] << 16);
                        float hi_1_1 = __uint_as_float(rope_1[1] & 4294901760u);
                        uint16_t _e4m3x2_f32_513;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_513) : "f"(hi_1_1), "f"(lo_0_1));
                        uint16_t pair_2_1 = _e4m3x2_f32_513;
                        {
                            v8_13_2[0] = v8_13_2[0] | (unsigned int)pair_2_1 << 16;
                        }
                        float lo_3_1 = __uint_as_float(rope_1[2] << 16);
                        float hi_4_1 = __uint_as_float(rope_1[2] & 4294901760u);
                        uint16_t _e4m3x2_f32_514;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_514) : "f"(hi_4_1), "f"(lo_3_1));
                        uint16_t pair_5_1 = _e4m3x2_f32_514;
                        {
                            v8_13_2[1] = (unsigned int)pair_5_1;
                        }
                        float lo_6_1 = __uint_as_float(rope_1[3] << 16);
                        float hi_7_1 = __uint_as_float(rope_1[3] & 4294901760u);
                        uint16_t _e4m3x2_f32_515;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_515) : "f"(hi_7_1), "f"(lo_6_1));
                        uint16_t pair_8_1 = _e4m3x2_f32_515;
                        {
                            v8_13_2[1] = v8_13_2[1] | (unsigned int)pair_8_1 << 16;
                        }
                        unsigned int rope_9_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&rope_9_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_1[(0) + 3]))
                            : "r"(kbase_2 + 32768 + (row_2 * 128 + ((2 * rblock_1 + 1) * 16 ^ row_2 % 8 * 16))));
                        float lo_10_1 = __uint_as_float(rope_9_1[0] << 16);
                        float hi_11_1 = __uint_as_float(rope_9_1[0] & 4294901760u);
                        uint16_t _e4m3x2_f32_516;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_516) : "f"(hi_11_1), "f"(lo_10_1));
                        uint16_t pair_12_1 = _e4m3x2_f32_516;
                        {
                            v8_13_2[2] = (unsigned int)pair_12_1;
                        }
                        float lo_13_1 = __uint_as_float(rope_9_1[1] << 16);
                        float hi_14_1 = __uint_as_float(rope_9_1[1] & 4294901760u);
                        uint16_t _e4m3x2_f32_517;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_517) : "f"(hi_14_1), "f"(lo_13_1));
                        uint16_t pair_15_1 = _e4m3x2_f32_517;
                        {
                            v8_13_2[2] = v8_13_2[2] | (unsigned int)pair_15_1 << 16;
                        }
                        float lo_16_1 = __uint_as_float(rope_9_1[2] << 16);
                        float hi_17_1 = __uint_as_float(rope_9_1[2] & 4294901760u);
                        uint16_t _e4m3x2_f32_518;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_518) : "f"(hi_17_1), "f"(lo_16_1));
                        uint16_t pair_18_1 = _e4m3x2_f32_518;
                        {
                            v8_13_2[3] = (unsigned int)pair_18_1;
                        }
                        float lo_19_1 = __uint_as_float(rope_9_1[3] << 16);
                        float hi_20_1 = __uint_as_float(rope_9_1[3] & 4294901760u);
                        uint16_t _e4m3x2_f32_519;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_519) : "f"(hi_20_1), "f"(lo_19_1));
                        uint16_t pair_21_1 = _e4m3x2_f32_519;
                        {
                            v8_13_2[3] = v8_13_2[3] | (unsigned int)pair_21_1 << 16;
                        }
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (80 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_13_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_13_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_13_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_13_2[(0) + 3])));
                    int vblock_14_2 = 32 * o_chunk_2 + 30;
                    unsigned int v8_15_2[4];
                    {
                        int rblock_2 = vblock_14_2 - 28;
                        unsigned int rope_2[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&rope_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_2[(0) + 3]))
                            : "r"(kbase_2 + 32768 + (row_2 * 128 + (2 * rblock_2 * 16 ^ row_2 % 8 * 16))));
                        float lo_2 = __uint_as_float(rope_2[0] << 16);
                        float hi_3 = __uint_as_float(rope_2[0] & 4294901760u);
                        uint16_t _e4m3x2_f32_528;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_528) : "f"(hi_3), "f"(lo_2));
                        uint16_t pair_3 = _e4m3x2_f32_528;
                        {
                            v8_15_2[0] = (unsigned int)pair_3;
                        }
                        float lo_0_2 = __uint_as_float(rope_2[1] << 16);
                        float hi_1_2 = __uint_as_float(rope_2[1] & 4294901760u);
                        uint16_t _e4m3x2_f32_529;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_529) : "f"(hi_1_2), "f"(lo_0_2));
                        uint16_t pair_2_2 = _e4m3x2_f32_529;
                        {
                            v8_15_2[0] = v8_15_2[0] | (unsigned int)pair_2_2 << 16;
                        }
                        float lo_3_2 = __uint_as_float(rope_2[2] << 16);
                        float hi_4_2 = __uint_as_float(rope_2[2] & 4294901760u);
                        uint16_t _e4m3x2_f32_530;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_530) : "f"(hi_4_2), "f"(lo_3_2));
                        uint16_t pair_5_2 = _e4m3x2_f32_530;
                        {
                            v8_15_2[1] = (unsigned int)pair_5_2;
                        }
                        float lo_6_2 = __uint_as_float(rope_2[3] << 16);
                        float hi_7_2 = __uint_as_float(rope_2[3] & 4294901760u);
                        uint16_t _e4m3x2_f32_531;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_531) : "f"(hi_7_2), "f"(lo_6_2));
                        uint16_t pair_8_2 = _e4m3x2_f32_531;
                        {
                            v8_15_2[1] = v8_15_2[1] | (unsigned int)pair_8_2 << 16;
                        }
                        unsigned int rope_9_2[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&rope_9_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_2[(0) + 3]))
                            : "r"(kbase_2 + 32768 + (row_2 * 128 + ((2 * rblock_2 + 1) * 16 ^ row_2 % 8 * 16))));
                        float lo_10_2 = __uint_as_float(rope_9_2[0] << 16);
                        float hi_11_2 = __uint_as_float(rope_9_2[0] & 4294901760u);
                        uint16_t _e4m3x2_f32_532;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_532) : "f"(hi_11_2), "f"(lo_10_2));
                        uint16_t pair_12_2 = _e4m3x2_f32_532;
                        {
                            v8_15_2[2] = (unsigned int)pair_12_2;
                        }
                        float lo_13_2 = __uint_as_float(rope_9_2[1] << 16);
                        float hi_14_2 = __uint_as_float(rope_9_2[1] & 4294901760u);
                        uint16_t _e4m3x2_f32_533;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_533) : "f"(hi_14_2), "f"(lo_13_2));
                        uint16_t pair_15_2 = _e4m3x2_f32_533;
                        {
                            v8_15_2[2] = v8_15_2[2] | (unsigned int)pair_15_2 << 16;
                        }
                        float lo_16_2 = __uint_as_float(rope_9_2[2] << 16);
                        float hi_17_2 = __uint_as_float(rope_9_2[2] & 4294901760u);
                        uint16_t _e4m3x2_f32_534;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_534) : "f"(hi_17_2), "f"(lo_16_2));
                        uint16_t pair_18_2 = _e4m3x2_f32_534;
                        {
                            v8_15_2[3] = (unsigned int)pair_18_2;
                        }
                        float lo_19_2 = __uint_as_float(rope_9_2[3] << 16);
                        float hi_20_2 = __uint_as_float(rope_9_2[3] & 4294901760u);
                        uint16_t _e4m3x2_f32_535;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_535) : "f"(hi_20_2), "f"(lo_19_2));
                        uint16_t pair_21_2 = _e4m3x2_f32_535;
                        {
                            v8_15_2[3] = v8_15_2[3] | (unsigned int)pair_21_2 << 16;
                        }
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (96 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_15_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_15_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_15_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_15_2[(0) + 3])));
                    int vblock_16_2 = 32 * o_chunk_2 + 31;
                    unsigned int v8_17_2[4];
                    {
                        int rblock_3 = vblock_16_2 - 28;
                        unsigned int rope_3[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&rope_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_3[(0) + 3]))
                            : "r"(kbase_2 + 32768 + (row_2 * 128 + (2 * rblock_3 * 16 ^ row_2 % 8 * 16))));
                        float lo_4 = __uint_as_float(rope_3[0] << 16);
                        float hi_5 = __uint_as_float(rope_3[0] & 4294901760u);
                        uint16_t _e4m3x2_f32_544;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_544) : "f"(hi_5), "f"(lo_4));
                        uint16_t pair_4 = _e4m3x2_f32_544;
                        {
                            v8_17_2[0] = (unsigned int)pair_4;
                        }
                        float lo_0_3 = __uint_as_float(rope_3[1] << 16);
                        float hi_1_3 = __uint_as_float(rope_3[1] & 4294901760u);
                        uint16_t _e4m3x2_f32_545;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_545) : "f"(hi_1_3), "f"(lo_0_3));
                        uint16_t pair_2_3 = _e4m3x2_f32_545;
                        {
                            v8_17_2[0] = v8_17_2[0] | (unsigned int)pair_2_3 << 16;
                        }
                        float lo_3_3 = __uint_as_float(rope_3[2] << 16);
                        float hi_4_3 = __uint_as_float(rope_3[2] & 4294901760u);
                        uint16_t _e4m3x2_f32_546;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_546) : "f"(hi_4_3), "f"(lo_3_3));
                        uint16_t pair_5_3 = _e4m3x2_f32_546;
                        {
                            v8_17_2[1] = (unsigned int)pair_5_3;
                        }
                        float lo_6_3 = __uint_as_float(rope_3[3] << 16);
                        float hi_7_3 = __uint_as_float(rope_3[3] & 4294901760u);
                        uint16_t _e4m3x2_f32_547;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_547) : "f"(hi_7_3), "f"(lo_6_3));
                        uint16_t pair_8_3 = _e4m3x2_f32_547;
                        {
                            v8_17_2[1] = v8_17_2[1] | (unsigned int)pair_8_3 << 16;
                        }
                        unsigned int rope_9_3[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&rope_9_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_3[(0) + 3]))
                            : "r"(kbase_2 + 32768 + (row_2 * 128 + ((2 * rblock_3 + 1) * 16 ^ row_2 % 8 * 16))));
                        float lo_10_3 = __uint_as_float(rope_9_3[0] << 16);
                        float hi_11_3 = __uint_as_float(rope_9_3[0] & 4294901760u);
                        uint16_t _e4m3x2_f32_548;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_548) : "f"(hi_11_3), "f"(lo_10_3));
                        uint16_t pair_12_3 = _e4m3x2_f32_548;
                        {
                            v8_17_2[2] = (unsigned int)pair_12_3;
                        }
                        float lo_13_3 = __uint_as_float(rope_9_3[1] << 16);
                        float hi_14_3 = __uint_as_float(rope_9_3[1] & 4294901760u);
                        uint16_t _e4m3x2_f32_549;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_549) : "f"(hi_14_3), "f"(lo_13_3));
                        uint16_t pair_15_3 = _e4m3x2_f32_549;
                        {
                            v8_17_2[2] = v8_17_2[2] | (unsigned int)pair_15_3 << 16;
                        }
                        float lo_16_3 = __uint_as_float(rope_9_3[2] << 16);
                        float hi_17_3 = __uint_as_float(rope_9_3[2] & 4294901760u);
                        uint16_t _e4m3x2_f32_550;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_550) : "f"(hi_17_3), "f"(lo_16_3));
                        uint16_t pair_18_3 = _e4m3x2_f32_550;
                        {
                            v8_17_2[3] = (unsigned int)pair_18_3;
                        }
                        float lo_19_3 = __uint_as_float(rope_9_3[3] << 16);
                        float hi_20_3 = __uint_as_float(rope_9_3[3] & 4294901760u);
                        uint16_t _e4m3x2_f32_551;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_551) : "f"(hi_20_3), "f"(lo_19_3));
                        uint16_t pair_21_3 = _e4m3x2_f32_551;
                        {
                            v8_17_2[3] = v8_17_2[3] | (unsigned int)pair_21_3 << 16;
                        }
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (112 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_17_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_17_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_17_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_17_2[(0) + 3])));
                } else {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(32768 + (row_2 * 128 + (96 ^ row_2 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(32768 + (row_2 * 128 + (112 ^ row_2 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (0 ^ row_2 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (16 ^ row_2 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (32 ^ row_2 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (48 ^ row_2 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (64 ^ row_2 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (80 ^ row_2 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (96 ^ row_2 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (112 ^ row_2 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(v_full_addr);
                mbarrier_wait_hint(s_full_addr, it_2 & 1, 10000000);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int col_lo_2 = 48 + 64 * half_2;
                float sv_2[16];
                tmem_ld_x16(&sv_2[0], taddr + (unsigned int)col_lo_2 + (unsigned int)(tmem_row_origin_2 << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int mword_2 = mask_word_2;
                sv_2[0] = (((mword_2 >> 16 & 1) != 0) ? sv_2[0] : -CAKE_INF);
                sv_2[1] = (((mword_2 >> 17 & 1) != 0) ? sv_2[1] : -CAKE_INF);
                sv_2[2] = (((mword_2 >> 18 & 1) != 0) ? sv_2[2] : -CAKE_INF);
                sv_2[3] = (((mword_2 >> 19 & 1) != 0) ? sv_2[3] : -CAKE_INF);
                sv_2[4] = (((mword_2 >> 20 & 1) != 0) ? sv_2[4] : -CAKE_INF);
                sv_2[5] = (((mword_2 >> 21 & 1) != 0) ? sv_2[5] : -CAKE_INF);
                sv_2[6] = (((mword_2 >> 22 & 1) != 0) ? sv_2[6] : -CAKE_INF);
                sv_2[7] = (((mword_2 >> 23 & 1) != 0) ? sv_2[7] : -CAKE_INF);
                sv_2[8] = (((mword_2 >> 24 & 1) != 0) ? sv_2[8] : -CAKE_INF);
                sv_2[9] = (((mword_2 >> 25 & 1) != 0) ? sv_2[9] : -CAKE_INF);
                sv_2[10] = (((mword_2 >> 26 & 1) != 0) ? sv_2[10] : -CAKE_INF);
                sv_2[11] = (((mword_2 >> 27 & 1) != 0) ? sv_2[11] : -CAKE_INF);
                sv_2[12] = (((mword_2 >> 28 & 1) != 0) ? sv_2[12] : -CAKE_INF);
                sv_2[13] = (((mword_2 >> 29 & 1) != 0) ? sv_2[13] : -CAKE_INF);
                sv_2[14] = (((mword_2 >> 30 & 1) != 0) ? sv_2[14] : -CAKE_INF);
                sv_2[15] = (((mword_2 >> 31 & 1) != 0) ? sv_2[15] : -CAKE_INF);
                float mx_2[16];
                mx_2[0] = sv_2[0];
                mx_2[1] = sv_2[1];
                mx_2[2] = sv_2[2];
                mx_2[3] = sv_2[3];
                mx_2[4] = sv_2[4];
                mx_2[5] = sv_2[5];
                mx_2[6] = sv_2[6];
                mx_2[7] = sv_2[7];
                mx_2[8] = sv_2[8];
                mx_2[9] = sv_2[9];
                mx_2[10] = sv_2[10];
                mx_2[11] = sv_2[11];
                mx_2[12] = sv_2[12];
                mx_2[13] = sv_2[13];
                mx_2[14] = sv_2[14];
                mx_2[15] = sv_2[15];
                float _max_420 = max_noftz(mx_2[0], mx_2[8]);
                mx_2[0] = _max_420;
                float _max_421 = max_noftz(mx_2[1], mx_2[9]);
                mx_2[1] = _max_421;
                float _max_422 = max_noftz(mx_2[2], mx_2[10]);
                mx_2[2] = _max_422;
                float _max_423 = max_noftz(mx_2[3], mx_2[11]);
                mx_2[3] = _max_423;
                float _max_424 = max_noftz(mx_2[4], mx_2[12]);
                mx_2[4] = _max_424;
                float _max_425 = max_noftz(mx_2[5], mx_2[13]);
                mx_2[5] = _max_425;
                float _max_426 = max_noftz(mx_2[6], mx_2[14]);
                mx_2[6] = _max_426;
                float _max_427 = max_noftz(mx_2[7], mx_2[15]);
                mx_2[7] = _max_427;
                float _max_428 = max_noftz(mx_2[0], mx_2[4]);
                mx_2[0] = _max_428;
                float _max_429 = max_noftz(mx_2[1], mx_2[5]);
                mx_2[1] = _max_429;
                float _max_430 = max_noftz(mx_2[2], mx_2[6]);
                mx_2[2] = _max_430;
                float _max_431 = max_noftz(mx_2[3], mx_2[7]);
                mx_2[3] = _max_431;
                float _max_432 = max_noftz(mx_2[0], mx_2[2]);
                mx_2[0] = _max_432;
                float _max_433 = max_noftz(mx_2[1], mx_2[3]);
                mx_2[1] = _max_433;
                float _max_434 = max_noftz(mx_2[0], mx_2[1]);
                mx_2[0] = _max_434;
                int slice_id_4 = 2 + 3 * half_2;
                smem_pmax[slice_id_4 * 64 + head_2] = mx_2[0];
                asm volatile("barrier.sync 10, 384;" ::: "memory");
                float m_tile_2 = smem_pmax[head_2];
                float _max_435 = max_noftz(m_tile_2, smem_pmax[64 + head_2]);
                m_tile_2 = _max_435;
                float _max_436 = max_noftz(m_tile_2, smem_pmax[128 + head_2]);
                m_tile_2 = _max_436;
                float _max_437 = max_noftz(m_tile_2, smem_pmax[192 + head_2]);
                m_tile_2 = _max_437;
                float _max_438 = max_noftz(m_tile_2, smem_pmax[256 + head_2]);
                m_tile_2 = _max_438;
                float _max_439 = max_noftz(m_tile_2, smem_pmax[320 + head_2]);
                m_tile_2 = _max_439;
                float cand_2 = m_tile_2 * softmax_scale_log2_2;
                if (it_2 == 0) {
                    float _max_440 = max_noftz(cand_2, sink_lane_2);
                    cand_2 = _max_440;
                }
                float _max_441 = max_noftz(cand_2, sm_2[0]);
                cand_2 = _max_441;
                int grow_2 = 0;
                float alpha_2 = 1.0f;
                if (it_2 == 0) {
                    grow_2 = 1;
                }
                if (cand_2 - sm_2[0] > 8.0f) {
                    grow_2 = 1;
                }
                if (grow_2 != 0) {
                    float _exp2_52 = approx_exp2(sm_2[0] - cand_2);
                    alpha_2 = ((sm_2[0] > -CAKE_INF) ? _exp2_52 : 0.0f);
                    sm_2[1] = sm_2[1] * alpha_2;
                    sm_2[2] = sm_2[2] * alpha_2;
                    sm_2[0] = cand_2;
                }
                float m_scaled_2 = ((sm_2[0] > -CAKE_INF) ? sm_2[0] : 0.0f);
                unsigned int _vote_6 = __ballot_sync(0xFFFFFFFF, grow_2 != 0);
                unsigned int grow_bits_2 = _vote_6;
                asm volatile("barrier.sync 10, 384;" ::: "memory");
                if (it_2 > 0) {
                    unsigned int any_grow_2 = smem_flag[0] | smem_flag[1];
                    if (any_grow_2 != 0) {
                        float alpha_c_2[16];
                        alpha_c_2[0] = smem_alpha[48];
                        alpha_c_2[1] = smem_alpha[49];
                        alpha_c_2[2] = smem_alpha[50];
                        alpha_c_2[3] = smem_alpha[51];
                        alpha_c_2[4] = smem_alpha[52];
                        alpha_c_2[5] = smem_alpha[53];
                        alpha_c_2[6] = smem_alpha[54];
                        alpha_c_2[7] = smem_alpha[55];
                        alpha_c_2[8] = smem_alpha[56];
                        alpha_c_2[9] = smem_alpha[57];
                        alpha_c_2[10] = smem_alpha[58];
                        alpha_c_2[11] = smem_alpha[59];
                        alpha_c_2[12] = smem_alpha[60];
                        alpha_c_2[13] = smem_alpha[61];
                        alpha_c_2[14] = smem_alpha[62];
                        alpha_c_2[15] = smem_alpha[63];
                        float ov_2[16];
                        tmem_ld_x16(&ov_2[0], taddr + 128 + 48 + (unsigned int)(tmem_row_origin_2 << 16));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        ov_2[0] = ov_2[0] * alpha_c_2[0];
                        ov_2[1] = ov_2[1] * alpha_c_2[1];
                        ov_2[2] = ov_2[2] * alpha_c_2[2];
                        ov_2[3] = ov_2[3] * alpha_c_2[3];
                        ov_2[4] = ov_2[4] * alpha_c_2[4];
                        ov_2[5] = ov_2[5] * alpha_c_2[5];
                        ov_2[6] = ov_2[6] * alpha_c_2[6];
                        ov_2[7] = ov_2[7] * alpha_c_2[7];
                        ov_2[8] = ov_2[8] * alpha_c_2[8];
                        ov_2[9] = ov_2[9] * alpha_c_2[9];
                        ov_2[10] = ov_2[10] * alpha_c_2[10];
                        ov_2[11] = ov_2[11] * alpha_c_2[11];
                        ov_2[12] = ov_2[12] * alpha_c_2[12];
                        ov_2[13] = ov_2[13] * alpha_c_2[13];
                        ov_2[14] = ov_2[14] * alpha_c_2[14];
                        ov_2[15] = ov_2[15] * alpha_c_2[15];
                        tmem_st_x16_f32(taddr + 128 + 48 + (unsigned int)(tmem_row_origin_2 << 16), ov_2);
                        tmem_ld_x16(&ov_2[0], taddr + 192 + 48 + (unsigned int)(tmem_row_origin_2 << 16));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        ov_2[0] = ov_2[0] * alpha_c_2[0];
                        ov_2[1] = ov_2[1] * alpha_c_2[1];
                        ov_2[2] = ov_2[2] * alpha_c_2[2];
                        ov_2[3] = ov_2[3] * alpha_c_2[3];
                        ov_2[4] = ov_2[4] * alpha_c_2[4];
                        ov_2[5] = ov_2[5] * alpha_c_2[5];
                        ov_2[6] = ov_2[6] * alpha_c_2[6];
                        ov_2[7] = ov_2[7] * alpha_c_2[7];
                        ov_2[8] = ov_2[8] * alpha_c_2[8];
                        ov_2[9] = ov_2[9] * alpha_c_2[9];
                        ov_2[10] = ov_2[10] * alpha_c_2[10];
                        ov_2[11] = ov_2[11] * alpha_c_2[11];
                        ov_2[12] = ov_2[12] * alpha_c_2[12];
                        ov_2[13] = ov_2[13] * alpha_c_2[13];
                        ov_2[14] = ov_2[14] * alpha_c_2[14];
                        ov_2[15] = ov_2[15] * alpha_c_2[15];
                        tmem_st_x16_f32(taddr + 192 + 48 + (unsigned int)(tmem_row_origin_2 << 16), ov_2);
                        tmem_ld_x16(&ov_2[0], taddr + 256 + 48 + (unsigned int)(tmem_row_origin_2 << 16));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        ov_2[0] = ov_2[0] * alpha_c_2[0];
                        ov_2[1] = ov_2[1] * alpha_c_2[1];
                        ov_2[2] = ov_2[2] * alpha_c_2[2];
                        ov_2[3] = ov_2[3] * alpha_c_2[3];
                        ov_2[4] = ov_2[4] * alpha_c_2[4];
                        ov_2[5] = ov_2[5] * alpha_c_2[5];
                        ov_2[6] = ov_2[6] * alpha_c_2[6];
                        ov_2[7] = ov_2[7] * alpha_c_2[7];
                        ov_2[8] = ov_2[8] * alpha_c_2[8];
                        ov_2[9] = ov_2[9] * alpha_c_2[9];
                        ov_2[10] = ov_2[10] * alpha_c_2[10];
                        ov_2[11] = ov_2[11] * alpha_c_2[11];
                        ov_2[12] = ov_2[12] * alpha_c_2[12];
                        ov_2[13] = ov_2[13] * alpha_c_2[13];
                        ov_2[14] = ov_2[14] * alpha_c_2[14];
                        ov_2[15] = ov_2[15] * alpha_c_2[15];
                        tmem_st_x16_f32(taddr + 256 + 48 + (unsigned int)(tmem_row_origin_2 << 16), ov_2);
                        tmem_ld_x16(&ov_2[0], taddr + 320 + 48 + (unsigned int)(tmem_row_origin_2 << 16));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        ov_2[0] = ov_2[0] * alpha_c_2[0];
                        ov_2[1] = ov_2[1] * alpha_c_2[1];
                        ov_2[2] = ov_2[2] * alpha_c_2[2];
                        ov_2[3] = ov_2[3] * alpha_c_2[3];
                        ov_2[4] = ov_2[4] * alpha_c_2[4];
                        ov_2[5] = ov_2[5] * alpha_c_2[5];
                        ov_2[6] = ov_2[6] * alpha_c_2[6];
                        ov_2[7] = ov_2[7] * alpha_c_2[7];
                        ov_2[8] = ov_2[8] * alpha_c_2[8];
                        ov_2[9] = ov_2[9] * alpha_c_2[9];
                        ov_2[10] = ov_2[10] * alpha_c_2[10];
                        ov_2[11] = ov_2[11] * alpha_c_2[11];
                        ov_2[12] = ov_2[12] * alpha_c_2[12];
                        ov_2[13] = ov_2[13] * alpha_c_2[13];
                        ov_2[14] = ov_2[14] * alpha_c_2[14];
                        ov_2[15] = ov_2[15] * alpha_c_2[15];
                        tmem_st_x16_f32(taddr + 320 + 48 + (unsigned int)(tmem_row_origin_2 << 16), ov_2);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                }
                float psum_2 = 0.0f;
                float rsum_2 = 0.0f;
                float _exp2_53 = approx_exp2(sv_2[0] * softmax_scale_log2_2 - m_scaled_2);
                sv_2[0] = _exp2_53;
                float _exp2_54 = approx_exp2(sv_2[1] * softmax_scale_log2_2 - m_scaled_2);
                sv_2[1] = _exp2_54;
                float _exp2_55 = approx_exp2(sv_2[2] * softmax_scale_log2_2 - m_scaled_2);
                sv_2[2] = _exp2_55;
                float _exp2_56 = approx_exp2(sv_2[3] * softmax_scale_log2_2 - m_scaled_2);
                sv_2[3] = _exp2_56;
                float _exp2_57 = approx_exp2(sv_2[4] * softmax_scale_log2_2 - m_scaled_2);
                sv_2[4] = _exp2_57;
                float _exp2_58 = approx_exp2(sv_2[5] * softmax_scale_log2_2 - m_scaled_2);
                sv_2[5] = _exp2_58;
                float _exp2_59 = approx_exp2(sv_2[6] * softmax_scale_log2_2 - m_scaled_2);
                sv_2[6] = _exp2_59;
                float _exp2_60 = approx_exp2(sv_2[7] * softmax_scale_log2_2 - m_scaled_2);
                sv_2[7] = _exp2_60;
                float _exp2_61 = approx_exp2(sv_2[8] * softmax_scale_log2_2 - m_scaled_2);
                sv_2[8] = _exp2_61;
                float _exp2_62 = approx_exp2(sv_2[9] * softmax_scale_log2_2 - m_scaled_2);
                sv_2[9] = _exp2_62;
                float _exp2_63 = approx_exp2(sv_2[10] * softmax_scale_log2_2 - m_scaled_2);
                sv_2[10] = _exp2_63;
                float _exp2_64 = approx_exp2(sv_2[11] * softmax_scale_log2_2 - m_scaled_2);
                sv_2[11] = _exp2_64;
                float _exp2_65 = approx_exp2(sv_2[12] * softmax_scale_log2_2 - m_scaled_2);
                sv_2[12] = _exp2_65;
                float _exp2_66 = approx_exp2(sv_2[13] * softmax_scale_log2_2 - m_scaled_2);
                sv_2[13] = _exp2_66;
                float _exp2_67 = approx_exp2(sv_2[14] * softmax_scale_log2_2 - m_scaled_2);
                sv_2[14] = _exp2_67;
                float _exp2_68 = approx_exp2(sv_2[15] * softmax_scale_log2_2 - m_scaled_2);
                sv_2[15] = _exp2_68;
                unsigned int pw_2[4];
                uint16_t _e4m3x2_f32_560;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_560) : "f"(sv_2[1]), "f"(sv_2[0]));
                uint16_t pair0_3 = _e4m3x2_f32_560;
                uint16_t _e4m3x2_f32_561;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_561) : "f"(sv_2[3]), "f"(sv_2[2]));
                uint16_t pair1_4 = _e4m3x2_f32_561;
                pw_2[0] = (unsigned int)pair0_3 | (unsigned int)pair1_4 << 16;
                uint16_t _e4m3x2_f32_562;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_562) : "f"(sv_2[5]), "f"(sv_2[4]));
                uint16_t pair0_0_2 = _e4m3x2_f32_562;
                uint16_t _e4m3x2_f32_563;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_563) : "f"(sv_2[7]), "f"(sv_2[6]));
                uint16_t pair1_1_2 = _e4m3x2_f32_563;
                pw_2[1] = (unsigned int)pair0_0_2 | (unsigned int)pair1_1_2 << 16;
                uint16_t _e4m3x2_f32_564;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_564) : "f"(sv_2[9]), "f"(sv_2[8]));
                uint16_t pair0_2_2 = _e4m3x2_f32_564;
                uint16_t _e4m3x2_f32_565;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_565) : "f"(sv_2[11]), "f"(sv_2[10]));
                uint16_t pair1_3_2 = _e4m3x2_f32_565;
                pw_2[2] = (unsigned int)pair0_2_2 | (unsigned int)pair1_3_2 << 16;
                uint16_t _e4m3x2_f32_566;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_566) : "f"(sv_2[13]), "f"(sv_2[12]));
                uint16_t pair0_4_2 = _e4m3x2_f32_566;
                uint16_t _e4m3x2_f32_567;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_567) : "f"(sv_2[15]), "f"(sv_2[14]));
                uint16_t pair1_5_2 = _e4m3x2_f32_567;
                pw_2[3] = (unsigned int)pair0_4_2 | (unsigned int)pair1_5_2 << 16;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(pbase_2 + (head_2 * 128 + ((col_lo_2 >> 4) * 16 ^ head_2 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&pw_2[0])), "r"(*reinterpret_cast<uint32_t*>(&pw_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&pw_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&pw_2[(0) + 3])));
                psum_2 = psum_2 + sv_2[0];
                float _fp8_rt_48;
                uint16_t _e4m3x2_32;
                uint32_t _f16x2_32;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_32) : "f"(0.0f), "f"(sv_2[0]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_32) : "h"(_e4m3x2_32));
                uint16_t _fp8_h0_32 = (uint16_t)(_f16x2_32 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_48) : "h"(_fp8_h0_32));
                rsum_2 = rsum_2 + _fp8_rt_48;
                psum_2 = psum_2 + sv_2[1];
                float _fp8_rt_49;
                uint16_t _e4m3x2_33;
                uint32_t _f16x2_33;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_33) : "f"(0.0f), "f"(sv_2[1]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_33) : "h"(_e4m3x2_33));
                uint16_t _fp8_h0_33 = (uint16_t)(_f16x2_33 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_49) : "h"(_fp8_h0_33));
                rsum_2 = rsum_2 + _fp8_rt_49;
                psum_2 = psum_2 + sv_2[2];
                float _fp8_rt_50;
                uint16_t _e4m3x2_34;
                uint32_t _f16x2_34;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_34) : "f"(0.0f), "f"(sv_2[2]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_34) : "h"(_e4m3x2_34));
                uint16_t _fp8_h0_34 = (uint16_t)(_f16x2_34 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_50) : "h"(_fp8_h0_34));
                rsum_2 = rsum_2 + _fp8_rt_50;
                psum_2 = psum_2 + sv_2[3];
                float _fp8_rt_51;
                uint16_t _e4m3x2_35;
                uint32_t _f16x2_35;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_35) : "f"(0.0f), "f"(sv_2[3]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_35) : "h"(_e4m3x2_35));
                uint16_t _fp8_h0_35 = (uint16_t)(_f16x2_35 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_51) : "h"(_fp8_h0_35));
                rsum_2 = rsum_2 + _fp8_rt_51;
                psum_2 = psum_2 + sv_2[4];
                float _fp8_rt_52;
                uint16_t _e4m3x2_36;
                uint32_t _f16x2_36;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_36) : "f"(0.0f), "f"(sv_2[4]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_36) : "h"(_e4m3x2_36));
                uint16_t _fp8_h0_36 = (uint16_t)(_f16x2_36 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_52) : "h"(_fp8_h0_36));
                rsum_2 = rsum_2 + _fp8_rt_52;
                psum_2 = psum_2 + sv_2[5];
                float _fp8_rt_53;
                uint16_t _e4m3x2_37;
                uint32_t _f16x2_37;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_37) : "f"(0.0f), "f"(sv_2[5]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_37) : "h"(_e4m3x2_37));
                uint16_t _fp8_h0_37 = (uint16_t)(_f16x2_37 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_53) : "h"(_fp8_h0_37));
                rsum_2 = rsum_2 + _fp8_rt_53;
                psum_2 = psum_2 + sv_2[6];
                float _fp8_rt_54;
                uint16_t _e4m3x2_38;
                uint32_t _f16x2_38;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_38) : "f"(0.0f), "f"(sv_2[6]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_38) : "h"(_e4m3x2_38));
                uint16_t _fp8_h0_38 = (uint16_t)(_f16x2_38 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_54) : "h"(_fp8_h0_38));
                rsum_2 = rsum_2 + _fp8_rt_54;
                psum_2 = psum_2 + sv_2[7];
                float _fp8_rt_55;
                uint16_t _e4m3x2_39;
                uint32_t _f16x2_39;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_39) : "f"(0.0f), "f"(sv_2[7]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_39) : "h"(_e4m3x2_39));
                uint16_t _fp8_h0_39 = (uint16_t)(_f16x2_39 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_55) : "h"(_fp8_h0_39));
                rsum_2 = rsum_2 + _fp8_rt_55;
                psum_2 = psum_2 + sv_2[8];
                float _fp8_rt_56;
                uint16_t _e4m3x2_40;
                uint32_t _f16x2_40;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_40) : "f"(0.0f), "f"(sv_2[8]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_40) : "h"(_e4m3x2_40));
                uint16_t _fp8_h0_40 = (uint16_t)(_f16x2_40 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_56) : "h"(_fp8_h0_40));
                rsum_2 = rsum_2 + _fp8_rt_56;
                psum_2 = psum_2 + sv_2[9];
                float _fp8_rt_57;
                uint16_t _e4m3x2_41;
                uint32_t _f16x2_41;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_41) : "f"(0.0f), "f"(sv_2[9]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_41) : "h"(_e4m3x2_41));
                uint16_t _fp8_h0_41 = (uint16_t)(_f16x2_41 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_57) : "h"(_fp8_h0_41));
                rsum_2 = rsum_2 + _fp8_rt_57;
                psum_2 = psum_2 + sv_2[10];
                float _fp8_rt_58;
                uint16_t _e4m3x2_42;
                uint32_t _f16x2_42;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_42) : "f"(0.0f), "f"(sv_2[10]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_42) : "h"(_e4m3x2_42));
                uint16_t _fp8_h0_42 = (uint16_t)(_f16x2_42 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_58) : "h"(_fp8_h0_42));
                rsum_2 = rsum_2 + _fp8_rt_58;
                psum_2 = psum_2 + sv_2[11];
                float _fp8_rt_59;
                uint16_t _e4m3x2_43;
                uint32_t _f16x2_43;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_43) : "f"(0.0f), "f"(sv_2[11]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_43) : "h"(_e4m3x2_43));
                uint16_t _fp8_h0_43 = (uint16_t)(_f16x2_43 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_59) : "h"(_fp8_h0_43));
                rsum_2 = rsum_2 + _fp8_rt_59;
                psum_2 = psum_2 + sv_2[12];
                float _fp8_rt_60;
                uint16_t _e4m3x2_44;
                uint32_t _f16x2_44;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_44) : "f"(0.0f), "f"(sv_2[12]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_44) : "h"(_e4m3x2_44));
                uint16_t _fp8_h0_44 = (uint16_t)(_f16x2_44 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_60) : "h"(_fp8_h0_44));
                rsum_2 = rsum_2 + _fp8_rt_60;
                psum_2 = psum_2 + sv_2[13];
                float _fp8_rt_61;
                uint16_t _e4m3x2_45;
                uint32_t _f16x2_45;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_45) : "f"(0.0f), "f"(sv_2[13]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_45) : "h"(_e4m3x2_45));
                uint16_t _fp8_h0_45 = (uint16_t)(_f16x2_45 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_61) : "h"(_fp8_h0_45));
                rsum_2 = rsum_2 + _fp8_rt_61;
                psum_2 = psum_2 + sv_2[14];
                float _fp8_rt_62;
                uint16_t _e4m3x2_46;
                uint32_t _f16x2_46;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_46) : "f"(0.0f), "f"(sv_2[14]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_46) : "h"(_e4m3x2_46));
                uint16_t _fp8_h0_46 = (uint16_t)(_f16x2_46 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_62) : "h"(_fp8_h0_46));
                rsum_2 = rsum_2 + _fp8_rt_62;
                psum_2 = psum_2 + sv_2[15];
                float _fp8_rt_63;
                uint16_t _e4m3x2_47;
                uint32_t _f16x2_47;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_47) : "f"(0.0f), "f"(sv_2[15]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_47) : "h"(_e4m3x2_47));
                uint16_t _fp8_h0_47 = (uint16_t)(_f16x2_47 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_63) : "h"(_fp8_h0_47));
                rsum_2 = rsum_2 + _fp8_rt_63;
                if (it_2 == 0) {
                    float sink_term_2 = 0.0f;
                    sm_2[1] = sink_term_2;
                    sm_2[2] = sink_term_2;
                }
                sm_2[1] = sm_2[1] + psum_2;
                sm_2[2] = sm_2[2] + rsum_2;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(p_full_addr);
            }
            int last_par_2 = tiles_per_split - 1 & 1;
            mbarrier_wait_hint(o_full_addr, last_par_2, 10000000);
            mbarrier_wait_hint(o_full_addr + 8, last_par_2, 10000000);
            mbarrier_wait_hint(o_full_addr + 16, last_par_2, 10000000);
            mbarrier_wait_hint(o_full_addr + 24, last_par_2, 10000000);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int slice_id_5 = 2 + 3 * half_2;
            smem_xsum[slice_id_5 * 64 + head_2] = sm_2[1];
            smem_xsum[384 + slice_id_5 * 64 + head_2] = sm_2[2];
            asm volatile("barrier.sync 10, 384;" ::: "memory");
            asm volatile("barrier.sync 10, 384;" ::: "memory");
            float norm_c_2[16];
            norm_c_2[0] = smem_norm[48];
            norm_c_2[1] = smem_norm[49];
            norm_c_2[2] = smem_norm[50];
            norm_c_2[3] = smem_norm[51];
            norm_c_2[4] = smem_norm[52];
            norm_c_2[5] = smem_norm[53];
            norm_c_2[6] = smem_norm[54];
            norm_c_2[7] = smem_norm[55];
            norm_c_2[8] = smem_norm[56];
            norm_c_2[9] = smem_norm[57];
            norm_c_2[10] = smem_norm[58];
            norm_c_2[11] = smem_norm[59];
            norm_c_2[12] = smem_norm[60];
            norm_c_2[13] = smem_norm[61];
            norm_c_2[14] = smem_norm[62];
            norm_c_2[15] = smem_norm[63];
            float o_values_2[16];
            float o_scaled_2[16];
            int is_odd_2 = lane & 1;
            tmem_ld_x16(&o_values_2[0], taddr + 128 + 48 + (unsigned int)(tmem_row_origin_2 << 16));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            o_scaled_2[0] = o_values_2[0] * norm_c_2[0];
            o_scaled_2[1] = o_values_2[1] * norm_c_2[1];
            o_scaled_2[2] = o_values_2[2] * norm_c_2[2];
            o_scaled_2[3] = o_values_2[3] * norm_c_2[3];
            o_scaled_2[4] = o_values_2[4] * norm_c_2[4];
            o_scaled_2[5] = o_values_2[5] * norm_c_2[5];
            o_scaled_2[6] = o_values_2[6] * norm_c_2[6];
            o_scaled_2[7] = o_values_2[7] * norm_c_2[7];
            o_scaled_2[8] = o_values_2[8] * norm_c_2[8];
            o_scaled_2[9] = o_values_2[9] * norm_c_2[9];
            o_scaled_2[10] = o_values_2[10] * norm_c_2[10];
            o_scaled_2[11] = o_values_2[11] * norm_c_2[11];
            o_scaled_2[12] = o_values_2[12] * norm_c_2[12];
            o_scaled_2[13] = o_values_2[13] * norm_c_2[13];
            o_scaled_2[14] = o_values_2[14] * norm_c_2[14];
            o_scaled_2[15] = o_values_2[15] * norm_c_2[15];
            uint32_t o_scaled_bf16_2[8];
            #pragma unroll
            for (int _lp = 0; _lp < 8; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_scaled_2[_lp*2 + 0], o_scaled_2[_lp*2+1 + 0]));
                o_scaled_bf16_2[_lp] = *(uint32_t*)&_bf2;
            }
            int dim_own_2 = row_2;
            smem_o16[24576 + dim_own_2] = (uint16_t)(o_scaled_bf16_2[0] & 65535);
            smem_o16[25088 + dim_own_2] = (uint16_t)(o_scaled_bf16_2[0] >> 16);
            smem_o16[25600 + dim_own_2] = (uint16_t)(o_scaled_bf16_2[1] & 65535);
            smem_o16[26112 + dim_own_2] = (uint16_t)(o_scaled_bf16_2[1] >> 16);
            smem_o16[26624 + dim_own_2] = (uint16_t)(o_scaled_bf16_2[2] & 65535);
            smem_o16[27136 + dim_own_2] = (uint16_t)(o_scaled_bf16_2[2] >> 16);
            smem_o16[27648 + dim_own_2] = (uint16_t)(o_scaled_bf16_2[3] & 65535);
            smem_o16[28160 + dim_own_2] = (uint16_t)(o_scaled_bf16_2[3] >> 16);
            smem_o16[28672 + dim_own_2] = (uint16_t)(o_scaled_bf16_2[4] & 65535);
            smem_o16[29184 + dim_own_2] = (uint16_t)(o_scaled_bf16_2[4] >> 16);
            smem_o16[29696 + dim_own_2] = (uint16_t)(o_scaled_bf16_2[5] & 65535);
            smem_o16[30208 + dim_own_2] = (uint16_t)(o_scaled_bf16_2[5] >> 16);
            smem_o16[30720 + dim_own_2] = (uint16_t)(o_scaled_bf16_2[6] & 65535);
            smem_o16[31232 + dim_own_2] = (uint16_t)(o_scaled_bf16_2[6] >> 16);
            smem_o16[31744 + dim_own_2] = (uint16_t)(o_scaled_bf16_2[7] & 65535);
            smem_o16[32256 + dim_own_2] = (uint16_t)(o_scaled_bf16_2[7] >> 16);
            tmem_ld_x16(&o_values_2[0], taddr + 192 + 48 + (unsigned int)(tmem_row_origin_2 << 16));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            o_scaled_2[0] = o_values_2[0] * norm_c_2[0];
            o_scaled_2[1] = o_values_2[1] * norm_c_2[1];
            o_scaled_2[2] = o_values_2[2] * norm_c_2[2];
            o_scaled_2[3] = o_values_2[3] * norm_c_2[3];
            o_scaled_2[4] = o_values_2[4] * norm_c_2[4];
            o_scaled_2[5] = o_values_2[5] * norm_c_2[5];
            o_scaled_2[6] = o_values_2[6] * norm_c_2[6];
            o_scaled_2[7] = o_values_2[7] * norm_c_2[7];
            o_scaled_2[8] = o_values_2[8] * norm_c_2[8];
            o_scaled_2[9] = o_values_2[9] * norm_c_2[9];
            o_scaled_2[10] = o_values_2[10] * norm_c_2[10];
            o_scaled_2[11] = o_values_2[11] * norm_c_2[11];
            o_scaled_2[12] = o_values_2[12] * norm_c_2[12];
            o_scaled_2[13] = o_values_2[13] * norm_c_2[13];
            o_scaled_2[14] = o_values_2[14] * norm_c_2[14];
            o_scaled_2[15] = o_values_2[15] * norm_c_2[15];
            uint32_t o_scaled_bf16_8_2[8];
            #pragma unroll
            for (int _lp = 0; _lp < 8; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_scaled_2[_lp*2 + 0], o_scaled_2[_lp*2+1 + 0]));
                o_scaled_bf16_8_2[_lp] = *(uint32_t*)&_bf2;
            }
            int dim_own_9_2 = 128 + row_2;
            smem_o16[24576 + dim_own_9_2] = (uint16_t)(o_scaled_bf16_8_2[0] & 65535);
            smem_o16[25088 + dim_own_9_2] = (uint16_t)(o_scaled_bf16_8_2[0] >> 16);
            smem_o16[25600 + dim_own_9_2] = (uint16_t)(o_scaled_bf16_8_2[1] & 65535);
            smem_o16[26112 + dim_own_9_2] = (uint16_t)(o_scaled_bf16_8_2[1] >> 16);
            smem_o16[26624 + dim_own_9_2] = (uint16_t)(o_scaled_bf16_8_2[2] & 65535);
            smem_o16[27136 + dim_own_9_2] = (uint16_t)(o_scaled_bf16_8_2[2] >> 16);
            smem_o16[27648 + dim_own_9_2] = (uint16_t)(o_scaled_bf16_8_2[3] & 65535);
            smem_o16[28160 + dim_own_9_2] = (uint16_t)(o_scaled_bf16_8_2[3] >> 16);
            smem_o16[28672 + dim_own_9_2] = (uint16_t)(o_scaled_bf16_8_2[4] & 65535);
            smem_o16[29184 + dim_own_9_2] = (uint16_t)(o_scaled_bf16_8_2[4] >> 16);
            smem_o16[29696 + dim_own_9_2] = (uint16_t)(o_scaled_bf16_8_2[5] & 65535);
            smem_o16[30208 + dim_own_9_2] = (uint16_t)(o_scaled_bf16_8_2[5] >> 16);
            smem_o16[30720 + dim_own_9_2] = (uint16_t)(o_scaled_bf16_8_2[6] & 65535);
            smem_o16[31232 + dim_own_9_2] = (uint16_t)(o_scaled_bf16_8_2[6] >> 16);
            smem_o16[31744 + dim_own_9_2] = (uint16_t)(o_scaled_bf16_8_2[7] & 65535);
            smem_o16[32256 + dim_own_9_2] = (uint16_t)(o_scaled_bf16_8_2[7] >> 16);
            tmem_ld_x16(&o_values_2[0], taddr + 256 + 48 + (unsigned int)(tmem_row_origin_2 << 16));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            o_scaled_2[0] = o_values_2[0] * norm_c_2[0];
            o_scaled_2[1] = o_values_2[1] * norm_c_2[1];
            o_scaled_2[2] = o_values_2[2] * norm_c_2[2];
            o_scaled_2[3] = o_values_2[3] * norm_c_2[3];
            o_scaled_2[4] = o_values_2[4] * norm_c_2[4];
            o_scaled_2[5] = o_values_2[5] * norm_c_2[5];
            o_scaled_2[6] = o_values_2[6] * norm_c_2[6];
            o_scaled_2[7] = o_values_2[7] * norm_c_2[7];
            o_scaled_2[8] = o_values_2[8] * norm_c_2[8];
            o_scaled_2[9] = o_values_2[9] * norm_c_2[9];
            o_scaled_2[10] = o_values_2[10] * norm_c_2[10];
            o_scaled_2[11] = o_values_2[11] * norm_c_2[11];
            o_scaled_2[12] = o_values_2[12] * norm_c_2[12];
            o_scaled_2[13] = o_values_2[13] * norm_c_2[13];
            o_scaled_2[14] = o_values_2[14] * norm_c_2[14];
            o_scaled_2[15] = o_values_2[15] * norm_c_2[15];
            uint32_t o_scaled_bf16_10_2[8];
            #pragma unroll
            for (int _lp = 0; _lp < 8; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_scaled_2[_lp*2 + 0], o_scaled_2[_lp*2+1 + 0]));
                o_scaled_bf16_10_2[_lp] = *(uint32_t*)&_bf2;
            }
            int dim_own_11_2 = 256 + row_2;
            smem_o16[24576 + dim_own_11_2] = (uint16_t)(o_scaled_bf16_10_2[0] & 65535);
            smem_o16[25088 + dim_own_11_2] = (uint16_t)(o_scaled_bf16_10_2[0] >> 16);
            smem_o16[25600 + dim_own_11_2] = (uint16_t)(o_scaled_bf16_10_2[1] & 65535);
            smem_o16[26112 + dim_own_11_2] = (uint16_t)(o_scaled_bf16_10_2[1] >> 16);
            smem_o16[26624 + dim_own_11_2] = (uint16_t)(o_scaled_bf16_10_2[2] & 65535);
            smem_o16[27136 + dim_own_11_2] = (uint16_t)(o_scaled_bf16_10_2[2] >> 16);
            smem_o16[27648 + dim_own_11_2] = (uint16_t)(o_scaled_bf16_10_2[3] & 65535);
            smem_o16[28160 + dim_own_11_2] = (uint16_t)(o_scaled_bf16_10_2[3] >> 16);
            smem_o16[28672 + dim_own_11_2] = (uint16_t)(o_scaled_bf16_10_2[4] & 65535);
            smem_o16[29184 + dim_own_11_2] = (uint16_t)(o_scaled_bf16_10_2[4] >> 16);
            smem_o16[29696 + dim_own_11_2] = (uint16_t)(o_scaled_bf16_10_2[5] & 65535);
            smem_o16[30208 + dim_own_11_2] = (uint16_t)(o_scaled_bf16_10_2[5] >> 16);
            smem_o16[30720 + dim_own_11_2] = (uint16_t)(o_scaled_bf16_10_2[6] & 65535);
            smem_o16[31232 + dim_own_11_2] = (uint16_t)(o_scaled_bf16_10_2[6] >> 16);
            smem_o16[31744 + dim_own_11_2] = (uint16_t)(o_scaled_bf16_10_2[7] & 65535);
            smem_o16[32256 + dim_own_11_2] = (uint16_t)(o_scaled_bf16_10_2[7] >> 16);
            tmem_ld_x16(&o_values_2[0], taddr + 320 + 48 + (unsigned int)(tmem_row_origin_2 << 16));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            o_scaled_2[0] = o_values_2[0] * norm_c_2[0];
            o_scaled_2[1] = o_values_2[1] * norm_c_2[1];
            o_scaled_2[2] = o_values_2[2] * norm_c_2[2];
            o_scaled_2[3] = o_values_2[3] * norm_c_2[3];
            o_scaled_2[4] = o_values_2[4] * norm_c_2[4];
            o_scaled_2[5] = o_values_2[5] * norm_c_2[5];
            o_scaled_2[6] = o_values_2[6] * norm_c_2[6];
            o_scaled_2[7] = o_values_2[7] * norm_c_2[7];
            o_scaled_2[8] = o_values_2[8] * norm_c_2[8];
            o_scaled_2[9] = o_values_2[9] * norm_c_2[9];
            o_scaled_2[10] = o_values_2[10] * norm_c_2[10];
            o_scaled_2[11] = o_values_2[11] * norm_c_2[11];
            o_scaled_2[12] = o_values_2[12] * norm_c_2[12];
            o_scaled_2[13] = o_values_2[13] * norm_c_2[13];
            o_scaled_2[14] = o_values_2[14] * norm_c_2[14];
            o_scaled_2[15] = o_values_2[15] * norm_c_2[15];
            uint32_t o_scaled_bf16_12_2[8];
            #pragma unroll
            for (int _lp = 0; _lp < 8; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_scaled_2[_lp*2 + 0], o_scaled_2[_lp*2+1 + 0]));
                o_scaled_bf16_12_2[_lp] = *(uint32_t*)&_bf2;
            }
            int dim_own_13_2 = 384 + row_2;
            smem_o16[24576 + dim_own_13_2] = (uint16_t)(o_scaled_bf16_12_2[0] & 65535);
            smem_o16[25088 + dim_own_13_2] = (uint16_t)(o_scaled_bf16_12_2[0] >> 16);
            smem_o16[25600 + dim_own_13_2] = (uint16_t)(o_scaled_bf16_12_2[1] & 65535);
            smem_o16[26112 + dim_own_13_2] = (uint16_t)(o_scaled_bf16_12_2[1] >> 16);
            smem_o16[26624 + dim_own_13_2] = (uint16_t)(o_scaled_bf16_12_2[2] & 65535);
            smem_o16[27136 + dim_own_13_2] = (uint16_t)(o_scaled_bf16_12_2[2] >> 16);
            smem_o16[27648 + dim_own_13_2] = (uint16_t)(o_scaled_bf16_12_2[3] & 65535);
            smem_o16[28160 + dim_own_13_2] = (uint16_t)(o_scaled_bf16_12_2[3] >> 16);
            smem_o16[28672 + dim_own_13_2] = (uint16_t)(o_scaled_bf16_12_2[4] & 65535);
            smem_o16[29184 + dim_own_13_2] = (uint16_t)(o_scaled_bf16_12_2[4] >> 16);
            smem_o16[29696 + dim_own_13_2] = (uint16_t)(o_scaled_bf16_12_2[5] & 65535);
            smem_o16[30208 + dim_own_13_2] = (uint16_t)(o_scaled_bf16_12_2[5] >> 16);
            smem_o16[30720 + dim_own_13_2] = (uint16_t)(o_scaled_bf16_12_2[6] & 65535);
            smem_o16[31232 + dim_own_13_2] = (uint16_t)(o_scaled_bf16_12_2[6] >> 16);
            smem_o16[31744 + dim_own_13_2] = (uint16_t)(o_scaled_bf16_12_2[7] & 65535);
            smem_o16[32256 + dim_own_13_2] = (uint16_t)(o_scaled_bf16_12_2[7] >> 16);
            asm volatile("barrier.sync 10, 384;" ::: "memory");
            int tid384_2 = 256 + row_2;
            int ci_2 = tid384_2;
            if (ci_2 < 4096) {
                int h_22 = ci_2 / 64;
                int ch_22 = ci_2 - h_22 * 64;
                if (head_base_2 + h_22 < num_heads) {
                    unsigned int w4_22[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_22[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_22[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_22[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_22[(0) + 3]))
                        : "r"(smem_o32_addr + (unsigned int)(h_22 * 1024) + (unsigned int)(ch_22 * 16)));
                    long long out_off_22 = ((long long)(query_idx_2 * num_heads + head_base_2 + h_22) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)(o_chunk_2 * 512) + (long long)(ch_22 * 8);
                    reinterpret_cast<int4*>(partial_O + out_off_22)[0] = reinterpret_cast<int4*>(w4_22)[0];
                }
            }
            int ci_14_2 = tid384_2 + 384;
            if (ci_14_2 < 4096) {
                int h_23 = ci_14_2 / 64;
                int ch_23 = ci_14_2 - h_23 * 64;
                if (head_base_2 + h_23 < num_heads) {
                    unsigned int w4_23[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_23[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_23[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_23[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_23[(0) + 3]))
                        : "r"(smem_o32_addr + (unsigned int)(h_23 * 1024) + (unsigned int)(ch_23 * 16)));
                    long long out_off_23 = ((long long)(query_idx_2 * num_heads + head_base_2 + h_23) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)(o_chunk_2 * 512) + (long long)(ch_23 * 8);
                    reinterpret_cast<int4*>(partial_O + out_off_23)[0] = reinterpret_cast<int4*>(w4_23)[0];
                }
            }
            int ci_15_2 = tid384_2 + 768;
            if (ci_15_2 < 4096) {
                int h_24 = ci_15_2 / 64;
                int ch_24 = ci_15_2 - h_24 * 64;
                if (head_base_2 + h_24 < num_heads) {
                    unsigned int w4_24[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_24[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_24[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_24[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_24[(0) + 3]))
                        : "r"(smem_o32_addr + (unsigned int)(h_24 * 1024) + (unsigned int)(ch_24 * 16)));
                    long long out_off_24 = ((long long)(query_idx_2 * num_heads + head_base_2 + h_24) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)(o_chunk_2 * 512) + (long long)(ch_24 * 8);
                    reinterpret_cast<int4*>(partial_O + out_off_24)[0] = reinterpret_cast<int4*>(w4_24)[0];
                }
            }
            int ci_16_2 = tid384_2 + 1152;
            if (ci_16_2 < 4096) {
                int h_25 = ci_16_2 / 64;
                int ch_25 = ci_16_2 - h_25 * 64;
                if (head_base_2 + h_25 < num_heads) {
                    unsigned int w4_25[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_25[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_25[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_25[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_25[(0) + 3]))
                        : "r"(smem_o32_addr + (unsigned int)(h_25 * 1024) + (unsigned int)(ch_25 * 16)));
                    long long out_off_25 = ((long long)(query_idx_2 * num_heads + head_base_2 + h_25) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)(o_chunk_2 * 512) + (long long)(ch_25 * 8);
                    reinterpret_cast<int4*>(partial_O + out_off_25)[0] = reinterpret_cast<int4*>(w4_25)[0];
                }
            }
            int ci_17_2 = tid384_2 + 1536;
            if (ci_17_2 < 4096) {
                int h_26 = ci_17_2 / 64;
                int ch_26 = ci_17_2 - h_26 * 64;
                if (head_base_2 + h_26 < num_heads) {
                    unsigned int w4_26[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_26[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_26[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_26[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_26[(0) + 3]))
                        : "r"(smem_o32_addr + (unsigned int)(h_26 * 1024) + (unsigned int)(ch_26 * 16)));
                    long long out_off_26 = ((long long)(query_idx_2 * num_heads + head_base_2 + h_26) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)(o_chunk_2 * 512) + (long long)(ch_26 * 8);
                    reinterpret_cast<int4*>(partial_O + out_off_26)[0] = reinterpret_cast<int4*>(w4_26)[0];
                }
            }
            int ci_18_2 = tid384_2 + 1920;
            if (ci_18_2 < 4096) {
                int h_27 = ci_18_2 / 64;
                int ch_27 = ci_18_2 - h_27 * 64;
                if (head_base_2 + h_27 < num_heads) {
                    unsigned int w4_27[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_27[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_27[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_27[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_27[(0) + 3]))
                        : "r"(smem_o32_addr + (unsigned int)(h_27 * 1024) + (unsigned int)(ch_27 * 16)));
                    long long out_off_27 = ((long long)(query_idx_2 * num_heads + head_base_2 + h_27) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)(o_chunk_2 * 512) + (long long)(ch_27 * 8);
                    reinterpret_cast<int4*>(partial_O + out_off_27)[0] = reinterpret_cast<int4*>(w4_27)[0];
                }
            }
            int ci_19_2 = tid384_2 + 2304;
            if (ci_19_2 < 4096) {
                int h_28 = ci_19_2 / 64;
                int ch_28 = ci_19_2 - h_28 * 64;
                if (head_base_2 + h_28 < num_heads) {
                    unsigned int w4_28[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_28[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_28[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_28[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_28[(0) + 3]))
                        : "r"(smem_o32_addr + (unsigned int)(h_28 * 1024) + (unsigned int)(ch_28 * 16)));
                    long long out_off_28 = ((long long)(query_idx_2 * num_heads + head_base_2 + h_28) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)(o_chunk_2 * 512) + (long long)(ch_28 * 8);
                    reinterpret_cast<int4*>(partial_O + out_off_28)[0] = reinterpret_cast<int4*>(w4_28)[0];
                }
            }
            int ci_20_2 = tid384_2 + 2688;
            if (ci_20_2 < 4096) {
                int h_29 = ci_20_2 / 64;
                int ch_29 = ci_20_2 - h_29 * 64;
                if (head_base_2 + h_29 < num_heads) {
                    unsigned int w4_29[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_29[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_29[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_29[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_29[(0) + 3]))
                        : "r"(smem_o32_addr + (unsigned int)(h_29 * 1024) + (unsigned int)(ch_29 * 16)));
                    long long out_off_29 = ((long long)(query_idx_2 * num_heads + head_base_2 + h_29) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)(o_chunk_2 * 512) + (long long)(ch_29 * 8);
                    reinterpret_cast<int4*>(partial_O + out_off_29)[0] = reinterpret_cast<int4*>(w4_29)[0];
                }
            }
            int ci_21_2 = tid384_2 + 3072;
            if (ci_21_2 < 4096) {
                int h_30 = ci_21_2 / 64;
                int ch_30 = ci_21_2 - h_30 * 64;
                if (head_base_2 + h_30 < num_heads) {
                    unsigned int w4_30[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_30[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_30[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_30[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_30[(0) + 3]))
                        : "r"(smem_o32_addr + (unsigned int)(h_30 * 1024) + (unsigned int)(ch_30 * 16)));
                    long long out_off_30 = ((long long)(query_idx_2 * num_heads + head_base_2 + h_30) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)(o_chunk_2 * 512) + (long long)(ch_30 * 8);
                    reinterpret_cast<int4*>(partial_O + out_off_30)[0] = reinterpret_cast<int4*>(w4_30)[0];
                }
            }
            int ci_22_2 = tid384_2 + 3456;
            if (ci_22_2 < 4096) {
                int h_31 = ci_22_2 / 64;
                int ch_31 = ci_22_2 - h_31 * 64;
                if (head_base_2 + h_31 < num_heads) {
                    unsigned int w4_31[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_31[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_31[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_31[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_31[(0) + 3]))
                        : "r"(smem_o32_addr + (unsigned int)(h_31 * 1024) + (unsigned int)(ch_31 * 16)));
                    long long out_off_31 = ((long long)(query_idx_2 * num_heads + head_base_2 + h_31) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)(o_chunk_2 * 512) + (long long)(ch_31 * 8);
                    reinterpret_cast<int4*>(partial_O + out_off_31)[0] = reinterpret_cast<int4*>(w4_31)[0];
                }
            }
            int ci_23_2 = tid384_2 + 3840;
            if (ci_23_2 < 4096) {
                int h_32 = ci_23_2 / 64;
                int ch_32 = ci_23_2 - h_32 * 64;
                if (head_base_2 + h_32 < num_heads) {
                    unsigned int w4_32[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_32[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_32[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_32[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_32[(0) + 3]))
                        : "r"(smem_o32_addr + (unsigned int)(h_32 * 1024) + (unsigned int)(ch_32 * 16)));
                    long long out_off_32 = ((long long)(query_idx_2 * num_heads + head_base_2 + h_32) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)(o_chunk_2 * 512) + (long long)(ch_32 * 8);
                    reinterpret_cast<int4*>(partial_O + out_off_32)[0] = reinterpret_cast<int4*>(w4_32)[0];
                }
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 12) {
        { // mma_warp_main
            if (tiles_per_split > 1) {
                if (elect_sync()) {
                    mbarrier_arrive(tok_free_addr + 8);
                }
            }
            unsigned int _phase_q_ready_0 = 0;
            mbarrier_wait(q_ready_addr, _phase_q_ready_0);
            _phase_q_ready_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (elect_sync()) {
                tcgen05_cp_32x128b_warpx4(tmem_tmem_sfq0, make_sf_cp_desc_lo_sbo512((((smem_qsf_addr) >> 4))));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfq0 + 4), make_sf_cp_desc_lo_sbo512((((smem_qsf_addr) >> 4) + 8)));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfq0 + 8), make_sf_cp_desc_lo_sbo512((((smem_qsf_addr) >> 4) + 16)));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfq0 + 12), make_sf_cp_desc_lo_sbo512((((smem_qsf_addr) >> 4) + 24)));
                tcgen05_cp_32x128b_warpx4(tmem_tmem_sfq1, make_sf_cp_desc_lo_sbo512((((smem_qsf_addr) >> 4) + 128)));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfq1 + 4), make_sf_cp_desc_lo_sbo512((((smem_qsf_addr) >> 4) + 128 + 8)));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfq1 + 8), make_sf_cp_desc_lo_sbo512((((smem_qsf_addr) >> 4) + 128 + 16)));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfq1 + 12), make_sf_cp_desc_lo_sbo512((((smem_qsf_addr) >> 4) + 128 + 24)));
            }
            unsigned int _phase_q_rope_full_0 = 0;
            mbarrier_wait(q_rope_full_addr, _phase_q_rope_full_0);
            _phase_q_rope_full_0 ^= 1;
            for (int it2 = 0; it2 < (tiles_per_split + 1) / 2; it2++) {
                int it_3 = 2 * it2;
                if (it_3 < tiles_per_split) {
                    int first = ((it_3 == 0) ? 1 : 0);
                    mbarrier_wait(kv_full_addr, 0);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfk0, make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4))));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfk0 + 4), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 8)));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfk0 + 8), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 16)));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfk0 + 12), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 24)));
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfk1, make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 128)));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfk1 + 4), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 128 + 8)));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfk1 + 8), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 128 + 16)));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfk1 + 12), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 128 + 24)));
                        int _mma_a_lo_0 = ((smem_qrope_addr) >> 4) & 0x3FFF;
                        int _mma_b_lo_0 = ((smem_krope_0_addr) >> 4) & 0x3FFF;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
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
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_tmem_s), "r"(0));
                        int _mma_b_lo_1 = (((smem_kf4_0_addr) >> 4) & 0x3FFF) + (0) * 1024;
                        {
                            uint64_t b_desc = ((uint64_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);
                            asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_s), "r"(tmem_tmem_q0), "l"(b_desc), "r"(0x8200480U), "r"(tmem_tmem_sfq0), "r"(tmem_tmem_sfk0), "r"(1) : "memory");
                            asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_s), "r"((tmem_tmem_q0 + 8)), "l"((b_desc + 2)), "r"(0x8200480U), "r"((tmem_tmem_sfq0 + 4)), "r"((tmem_tmem_sfk0 + 4)), "r"(1) : "memory");
                            asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_s), "r"((tmem_tmem_q0 + 16)), "l"((b_desc + 4)), "r"(0x8200480U), "r"((tmem_tmem_sfq0 + 8)), "r"((tmem_tmem_sfk0 + 8)), "r"(1) : "memory");
                            asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_s), "r"((tmem_tmem_q0 + 24)), "l"((b_desc + 6)), "r"(0x8200480U), "r"((tmem_tmem_sfq0 + 12)), "r"((tmem_tmem_sfk0 + 12)), "r"(1) : "memory");
                        }
                        int _mma_b_lo_2 = (((smem_kf4_0_addr) >> 4) & 0x3FFF) + (1) * 1024;
                        {
                            uint64_t b_desc = ((uint64_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);
                            asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_s), "r"(tmem_tmem_q1), "l"(b_desc), "r"(0x8200480U), "r"(tmem_tmem_sfq1), "r"(tmem_tmem_sfk1), "r"(1) : "memory");
                            asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_s), "r"((tmem_tmem_q1 + 8)), "l"((b_desc + 2)), "r"(0x8200480U), "r"((tmem_tmem_sfq1 + 4)), "r"((tmem_tmem_sfk1 + 4)), "r"(1) : "memory");
                            asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_s), "r"((tmem_tmem_q1 + 16)), "l"((b_desc + 4)), "r"(0x8200480U), "r"((tmem_tmem_sfq1 + 8)), "r"((tmem_tmem_sfk1 + 8)), "r"(1) : "memory");
                            asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_s), "r"((tmem_tmem_q1 + 24)), "l"((b_desc + 6)), "r"(0x8200480U), "r"((tmem_tmem_sfq1 + 12)), "r"((tmem_tmem_sfk1 + 12)), "r"(1) : "memory");
                        }
                        tcgen05_commit(s_full_addr);
                    }
                    mbarrier_wait(v_full_addr, 0);
                    if (it_3 + 2 < tiles_per_split) {
                        if (elect_sync()) {
                            tcgen05_commit(tok_free_addr);
                        }
                    }
                    mbarrier_wait(p_full_addr, 0);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        int _mma_a_lo_3 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (0) * 1024;
                        int _mma_b_lo_3 = (((smem_p_0_addr) >> 4) & 0x3FFF) | 0x2000000;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 135299088;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"(tmem_tmem_o0), "r"(((first) ? 0 : 1)));
                        tcgen05_commit(o_full_addr);
                        int _mma_a_lo_4 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (1) * 1024;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 135299088;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_3), "r"(tmem_tmem_o1), "r"(((first) ? 0 : 1)));
                        tcgen05_commit(o_full_addr + 8);
                        int _mma_a_lo_5 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (2) * 1024;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 135299088;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_3), "r"(tmem_tmem_o2), "r"(((first) ? 0 : 1)));
                        tcgen05_commit(o_full_addr + 16);
                        int _mma_a_lo_6 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (3) * 1024;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 135299088;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_6), "r"(_mma_b_lo_3), "r"(tmem_tmem_o3), "r"(((first) ? 0 : 1)));
                        tcgen05_commit(o_full_addr + 24);
                    }
                }
                int it_0 = 2 * it2 + 1;
                if (it_0 < tiles_per_split) {
                    int first_1 = ((it_0 == 0) ? 1 : 0);
                    mbarrier_wait(kv_full_addr, 1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfk0, make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4))));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfk0 + 4), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 8)));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfk0 + 8), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 16)));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfk0 + 12), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 24)));
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfk1, make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 128)));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfk1 + 4), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 128 + 8)));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfk1 + 8), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 128 + 16)));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfk1 + 12), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 128 + 24)));
                        int _mma_a_lo_7 = ((smem_qrope_addr) >> 4) & 0x3FFF;
                        int _mma_b_lo_7 = ((smem_krope_1_addr) >> 4) & 0x3FFF;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
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
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_7), "r"(_mma_b_lo_7), "r"(tmem_tmem_s), "r"(0));
                        int _mma_b_lo_8 = (((smem_kf4_1_addr) >> 4) & 0x3FFF) + (0) * 1024;
                        {
                            uint64_t b_desc = ((uint64_t)_mma_b_lo_8) | ((uint64_t)0x40004040 << 32);
                            asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_s), "r"(tmem_tmem_q0), "l"(b_desc), "r"(0x8200480U), "r"(tmem_tmem_sfq0), "r"(tmem_tmem_sfk0), "r"(1) : "memory");
                            asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_s), "r"((tmem_tmem_q0 + 8)), "l"((b_desc + 2)), "r"(0x8200480U), "r"((tmem_tmem_sfq0 + 4)), "r"((tmem_tmem_sfk0 + 4)), "r"(1) : "memory");
                            asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_s), "r"((tmem_tmem_q0 + 16)), "l"((b_desc + 4)), "r"(0x8200480U), "r"((tmem_tmem_sfq0 + 8)), "r"((tmem_tmem_sfk0 + 8)), "r"(1) : "memory");
                            asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_s), "r"((tmem_tmem_q0 + 24)), "l"((b_desc + 6)), "r"(0x8200480U), "r"((tmem_tmem_sfq0 + 12)), "r"((tmem_tmem_sfk0 + 12)), "r"(1) : "memory");
                        }
                        int _mma_b_lo_9 = (((smem_kf4_1_addr) >> 4) & 0x3FFF) + (1) * 1024;
                        {
                            uint64_t b_desc = ((uint64_t)_mma_b_lo_9) | ((uint64_t)0x40004040 << 32);
                            asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_s), "r"(tmem_tmem_q1), "l"(b_desc), "r"(0x8200480U), "r"(tmem_tmem_sfq1), "r"(tmem_tmem_sfk1), "r"(1) : "memory");
                            asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_s), "r"((tmem_tmem_q1 + 8)), "l"((b_desc + 2)), "r"(0x8200480U), "r"((tmem_tmem_sfq1 + 4)), "r"((tmem_tmem_sfk1 + 4)), "r"(1) : "memory");
                            asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_s), "r"((tmem_tmem_q1 + 16)), "l"((b_desc + 4)), "r"(0x8200480U), "r"((tmem_tmem_sfq1 + 8)), "r"((tmem_tmem_sfk1 + 8)), "r"(1) : "memory");
                            asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_s), "r"((tmem_tmem_q1 + 24)), "l"((b_desc + 6)), "r"(0x8200480U), "r"((tmem_tmem_sfq1 + 12)), "r"((tmem_tmem_sfk1 + 12)), "r"(1) : "memory");
                        }
                        tcgen05_commit(s_full_addr);
                    }
                    mbarrier_wait(v_full_addr, 1);
                    if (it_0 + 2 < tiles_per_split) {
                        if (elect_sync()) {
                            tcgen05_commit(tok_free_addr + 8);
                        }
                    }
                    mbarrier_wait(p_full_addr, 1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        int _mma_a_lo_10 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (0) * 1024;
                        int _mma_b_lo_10 = (((smem_p_1_addr) >> 4) & 0x3FFF) | 0x2000000;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 135299088;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_10), "r"(_mma_b_lo_10), "r"(tmem_tmem_o0), "r"(((first_1) ? 0 : 1)));
                        tcgen05_commit(o_full_addr);
                        int _mma_a_lo_11 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (1) * 1024;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 135299088;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_11), "r"(_mma_b_lo_10), "r"(tmem_tmem_o1), "r"(((first_1) ? 0 : 1)));
                        tcgen05_commit(o_full_addr + 8);
                        int _mma_a_lo_12 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (2) * 1024;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 135299088;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_12), "r"(_mma_b_lo_10), "r"(tmem_tmem_o2), "r"(((first_1) ? 0 : 1)));
                        tcgen05_commit(o_full_addr + 16);
                        int _mma_a_lo_13 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (3) * 1024;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 135299088;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_13), "r"(_mma_b_lo_10), "r"(tmem_tmem_o3), "r"(((first_1) ? 0 : 1)));
                        tcgen05_commit(o_full_addr + 24);
                    }
                }
            }
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait(tmem_dealloc_addr, _phase_tmem_dealloc_0);
            _phase_tmem_dealloc_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
        }
    }
    // ---- Role: load_warp ----
    if (warp >= 13 && warp <= 15) {
        { // load_warp_main
            const int load_tid = (warp - 13) * 32 + lane;
            int work_idx_3 = blockIdx.x;
            int head_tile_3 = work_idx_3 % num_head_tiles;
            int split_work_3 = work_idx_3 / num_head_tiles;
            int query_idx_3 = split_work_3 / num_splits;
            if (load_tid == 0) {
                mbarrier_arrive_expect_tx(q_nope_full0_addr, 24576);
                tma_4d_gmem2smem(smem_qstage_addr, (&tmap_q), 0, head_tile_3 * 64, 0, query_idx_3, q_nope_full0_addr);
                tma_4d_gmem2smem(smem_qstage_addr + 8192, (&tmap_q), 0, head_tile_3 * 64, 1, query_idx_3, q_nope_full0_addr);
                tma_4d_gmem2smem(smem_qstage_addr + 16384, (&tmap_q), 0, head_tile_3 * 64, 2, query_idx_3, q_nope_full0_addr);
                mbarrier_arrive_expect_tx(q_nope_full1_addr, 16384);
                tma_4d_gmem2smem(smem_qstage_addr + 24576, (&tmap_q), 0, head_tile_3 * 64, 3, query_idx_3, q_nope_full1_addr);
                tma_4d_gmem2smem(smem_qstage_addr + 32768, (&tmap_q), 0, head_tile_3 * 64, 4, query_idx_3, q_nope_full1_addr);
                mbarrier_arrive_expect_tx(q_nope_full2_addr, 16384);
                tma_4d_gmem2smem(smem_qstage_addr + 40960, (&tmap_q), 0, head_tile_3 * 64, 5, query_idx_3, q_nope_full2_addr);
                tma_4d_gmem2smem(smem_qstage_addr + 49152, (&tmap_q), 0, head_tile_3 * 64, 6, query_idx_3, q_nope_full2_addr);
                mbarrier_arrive_expect_tx(q_rope_full_addr, 16384);
                tma_4d_gmem2smem(smem_qrope_addr, (&tmap_q), 0, head_tile_3 * 64, 7, query_idx_3, q_rope_full_addr);
                tma_4d_gmem2smem(smem_qrope_addr + 8192, (&tmap_q), 0, head_tile_3 * 64, 7, query_idx_3, q_rope_full_addr);
            }
            const int load_tid_0 = (warp - 13) * 32 + lane;
            const int l_chunk = load_tid_0 % 24;
            const int l_row0 = load_tid_0 / 24;
            const int l_kind = ((l_chunk < 14) ? 0 : ((l_chunk < 22) ? 1 : 2));
            const int l_sw_chunk = ((l_kind == 0) ? l_chunk : l_chunk - 14);
            long long l_src_off = (long long)(((l_kind < 2) ? 16 * l_chunk : 16 * (l_chunk - 14 - 8)));
            const int l_row_bytes = ((l_kind < 2) ? 128 : 32);
            const int l_rowoff = l_row0 * 16;
            int l_dst_0 = ((l_kind == 0) ? smem_kf4_0_addr + (unsigned int)(l_chunk / 8 * 16384) : ((l_kind == 1) ? smem_krope_0_addr : smem_sfs_0_addr + (unsigned int)(16 * (l_chunk - 14 - 8))));
            int l_dst_1 = ((l_kind == 0) ? smem_kf4_1_addr + (unsigned int)(l_chunk / 8 * 16384) : ((l_kind == 1) ? smem_krope_1_addr : smem_sfs_1_addr + (unsigned int)(16 * (l_chunk - 14 - 8))));
            const int l_sw_r0 = ((l_kind < 2) ? l_row0 * 128 + (((l_sw_chunk ^ l_row0) & 7) << 4) : l_row0 * 32);
            const int l_sw_r4 = ((l_kind < 2) ? (l_row0 + 4) * 128 + (((l_sw_chunk ^ l_row0 + 4) & 7) << 4) : (l_row0 + 4) * 32);
            int l_dst0_0 = l_dst_0 + l_sw_r0;
            int l_dst1_0 = l_dst_0 + l_sw_r4;
            int l_dst0_1 = l_dst_1 + l_sw_r0;
            int l_dst1_1 = l_dst_1 + l_sw_r4;
            int l_work_idx = blockIdx.x;
            int l_head_tile = l_work_idx % num_head_tiles;
            int l_split_work = l_work_idx / num_head_tiles;
            int l_split_idx = l_split_work % num_splits;
            int l_query_idx = l_split_work / num_splits;
            int tile_lo = l_split_idx * tiles_per_split;
            int idx[2];
            unsigned int offw[8];
            int is_main = 1;
            if (tile_lo >= num_main_tiles) {
                is_main = 0;
            }
            int tile_in_table = ((is_main != 0) ? tile_lo : tile_lo - num_main_tiles);
            int table_width = ((is_main != 0) ? main_width : extra_width);
            int* row_ptr = ((is_main != 0) ? (main_indices + (l_query_idx * main_index_stride)) : (extra_indices + (l_query_idx * extra_index_stride)));
            int active_len = table_width;
            if (is_main != 0) {
                if (has_main_lengths != 0) {
                    active_len = main_lengths[l_query_idx];
                }
            } else if (has_extra_lengths != 0) {
                active_len = extra_lengths[l_query_idx];
            }
            if (active_len < 0) {
                active_len = 0;
            }
            if (active_len > table_width) {
                active_len = table_width;
            }
            int r = load_tid_0;
            idx[0] = -1;
            if (r < 128) {
                int col = tile_in_table * 128 + r;
                if (col < active_len) {
                    idx[0] = row_ptr[col];
                }
            }
            int r_1 = load_tid_0 + 96;
            idx[1] = -1;
            if (r_1 < 128) {
                int col_1 = tile_in_table * 128 + r_1;
                if (col_1 < active_len) {
                    idx[1] = row_ptr[col_1];
                }
            }
            int page_shift = ((is_main != 0) ? main_page_shift : extra_page_shift);
            long long page_stride = ((is_main != 0) ? main_page_stride : extra_page_stride);
            offw[0] = 4294967295;
            offw[1] = 4294967295;
            offw[2] = 4294967295;
            offw[3] = 4294967295;
            if (idx[0] >= 0) {
                int cpage = idx[0] >> page_shift;
                int cslot = idx[0] - (cpage << page_shift);
                long long cbase = (long long)cpage * page_stride;
                long long offd = cbase + (long long)(cslot * 352);
                long long offs = cbase + (long long)((1 << page_shift) * 352) + (long long)(cslot * 32);
                offw[0] = (unsigned int)offd;
                offw[1] = (unsigned int)(offd >> 32);
                offw[2] = (unsigned int)offs;
                offw[3] = (unsigned int)(offs >> 32);
            }
            offw[4] = 4294967295;
            offw[5] = 4294967295;
            offw[6] = 4294967295;
            offw[7] = 4294967295;
            if (idx[1] >= 0) {
                int cpage_1 = idx[1] >> page_shift;
                int cslot_1 = idx[1] - (cpage_1 << page_shift);
                long long cbase_1 = (long long)cpage_1 * page_stride;
                long long offd_1 = cbase_1 + (long long)(cslot_1 * 352);
                long long offs_1 = cbase_1 + (long long)((1 << page_shift) * 352) + (long long)(cslot_1 * 32);
                offw[4] = (unsigned int)offd_1;
                offw[5] = (unsigned int)(offd_1 >> 32);
                offw[6] = (unsigned int)offs_1;
                offw[7] = (unsigned int)(offs_1 >> 32);
            }
            for (int it2_1 = 0; it2_1 < (tiles_per_split + 1) / 2; it2_1++) {
                int it_4 = 2 * it2_1;
                if (it_4 < tiles_per_split) {
                    if (it_4 > 1) {
                        mbarrier_wait_hint(tok_free_addr, it2_1 - 1 & 1, 10000000);
                    }
                    int r_0 = load_tid_0;
                    if (r_0 < 128) {
                        smem_kw32[(it_4 & 1) * 256 + r_0] = idx[0];
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_rowoff_0_addr + (unsigned int)(r_0 * 16)), "r"(*reinterpret_cast<uint32_t*>(&offw[0])), "r"(*reinterpret_cast<uint32_t*>(&offw[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&offw[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&offw[(0) + 3])));
                    }
                    int r_2 = load_tid_0 + 96;
                    if (r_2 < 128) {
                        smem_kw32[(it_4 & 1) * 256 + r_2] = idx[1];
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_rowoff_0_addr + (unsigned int)(r_2 * 16)), "r"(*reinterpret_cast<uint32_t*>(&offw[4])), "r"(*reinterpret_cast<uint32_t*>(&offw[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&offw[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&offw[(4) + 3])));
                    }
                    unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, idx[0] >= 0);
                    unsigned int vb0 = _vote_0;
                    unsigned int _vote_1 = __ballot_sync(0xFFFFFFFF, idx[1] >= 0);
                    unsigned int vb1 = _vote_1;
                    if (lane == 0) {
                        int lw = warp - 13;
                        smem_kw32[(it_4 & 1) * 256 + ((lw == 0) ? 128 : ((lw == 1) ? 129 : 130))] = (int)vb0;
                        if (lw == 0) {
                            smem_kw32[(it_4 & 1) * 256 + 131] = (int)vb1;
                        }
                    }
                    asm volatile("barrier.sync 8, 96;" ::: "memory");
                    int is_main_3 = 1;
                    if (tile_lo + it_4 >= num_main_tiles) {
                        is_main_3 = 0;
                    }
                    uint8_t* cache = ((is_main_3 != 0) ? (main_cache) : (extra_cache));
                    for (int i8 = 0; i8 < 4; i8++) {
                        long long offs8[8];
                        unsigned int ok8[8];
                        unsigned int w4_33[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_33[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_33[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_33[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_33[(0) + 3]))
                            : "r"(smem_rowoff_0_addr + (unsigned int)l_rowoff + (unsigned int)(i8 * 512)));
                        unsigned int lo_5 = ((l_kind < 2) ? w4_33[0] : w4_33[2]);
                        unsigned int hi_6 = ((l_kind < 2) ? w4_33[1] : w4_33[3]);
                        offs8[0] = (long long)hi_6 << 32 | (long long)lo_5;
                        ok8[0] = hi_6 & 2147483648u;
                        unsigned int w4_0[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_0[(0) + 3]))
                            : "r"(smem_rowoff_0_addr + (unsigned int)l_rowoff + (unsigned int)(i8 * 512) + 64));
                        unsigned int lo_1_1 = ((l_kind < 2) ? w4_0[0] : w4_0[2]);
                        unsigned int hi_2_1 = ((l_kind < 2) ? w4_0[1] : w4_0[3]);
                        offs8[1] = (long long)hi_2_1 << 32 | (long long)lo_1_1;
                        ok8[1] = hi_2_1 & 2147483648u;
                        unsigned int w4_3_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_3_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_3_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_3_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_3_1[(0) + 3]))
                            : "r"(smem_rowoff_0_addr + (unsigned int)l_rowoff + (unsigned int)(i8 * 512) + 128));
                        unsigned int lo_4_1 = ((l_kind < 2) ? w4_3_1[0] : w4_3_1[2]);
                        unsigned int hi_5_1 = ((l_kind < 2) ? w4_3_1[1] : w4_3_1[3]);
                        offs8[2] = (long long)hi_5_1 << 32 | (long long)lo_4_1;
                        ok8[2] = hi_5_1 & 2147483648u;
                        unsigned int w4_6_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_6_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_6_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_6_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_6_1[(0) + 3]))
                            : "r"(smem_rowoff_0_addr + (unsigned int)l_rowoff + (unsigned int)(i8 * 512) + 192));
                        unsigned int lo_7 = ((l_kind < 2) ? w4_6_1[0] : w4_6_1[2]);
                        unsigned int hi_8 = ((l_kind < 2) ? w4_6_1[1] : w4_6_1[3]);
                        offs8[3] = (long long)hi_8 << 32 | (long long)lo_7;
                        ok8[3] = hi_8 & 2147483648u;
                        unsigned int w4_9_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_9_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_9_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_9_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_9_1[(0) + 3]))
                            : "r"(smem_rowoff_0_addr + (unsigned int)l_rowoff + (unsigned int)(i8 * 512) + 256));
                        unsigned int lo_10_4 = ((l_kind < 2) ? w4_9_1[0] : w4_9_1[2]);
                        unsigned int hi_11_4 = ((l_kind < 2) ? w4_9_1[1] : w4_9_1[3]);
                        offs8[4] = (long long)hi_11_4 << 32 | (long long)lo_10_4;
                        ok8[4] = hi_11_4 & 2147483648u;
                        unsigned int w4_12_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_12_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12_1[(0) + 3]))
                            : "r"(smem_rowoff_0_addr + (unsigned int)l_rowoff + (unsigned int)(i8 * 512) + 320));
                        unsigned int lo_13_4 = ((l_kind < 2) ? w4_12_1[0] : w4_12_1[2]);
                        unsigned int hi_14_4 = ((l_kind < 2) ? w4_12_1[1] : w4_12_1[3]);
                        offs8[5] = (long long)hi_14_4 << 32 | (long long)lo_13_4;
                        ok8[5] = hi_14_4 & 2147483648u;
                        unsigned int w4_15_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_15_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_15_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_15_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_15_1[(0) + 3]))
                            : "r"(smem_rowoff_0_addr + (unsigned int)l_rowoff + (unsigned int)(i8 * 512) + 384));
                        unsigned int lo_16_4 = ((l_kind < 2) ? w4_15_1[0] : w4_15_1[2]);
                        unsigned int hi_17_4 = ((l_kind < 2) ? w4_15_1[1] : w4_15_1[3]);
                        offs8[6] = (long long)hi_17_4 << 32 | (long long)lo_16_4;
                        ok8[6] = hi_17_4 & 2147483648u;
                        unsigned int w4_18_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_18_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_18_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_18_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_18_1[(0) + 3]))
                            : "r"(smem_rowoff_0_addr + (unsigned int)l_rowoff + (unsigned int)(i8 * 512) + 448));
                        unsigned int lo_19_4 = ((l_kind < 2) ? w4_18_1[0] : w4_18_1[2]);
                        unsigned int hi_20_4 = ((l_kind < 2) ? w4_18_1[1] : w4_18_1[3]);
                        offs8[7] = (long long)hi_20_4 << 32 | (long long)lo_19_4;
                        ok8[7] = hi_20_4 & 2147483648u;
                        if (ok8[0] == 0) {
                            int dst = l_dst0_0 + 32 * i8 * l_row_bytes;
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(dst), "l"(cache + (offs8[0] + l_src_off)));
                        }
                        if (ok8[1] == 0) {
                            int dst_1 = l_dst1_0 + 32 * i8 * l_row_bytes;
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(dst_1), "l"(cache + (offs8[1] + l_src_off)));
                        }
                        if (ok8[2] == 0) {
                            int dst_2 = l_dst0_0 + (32 * i8 + 8) * l_row_bytes;
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(dst_2), "l"(cache + (offs8[2] + l_src_off)));
                        }
                        if (ok8[3] == 0) {
                            int dst_3 = l_dst1_0 + (32 * i8 + 8) * l_row_bytes;
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(dst_3), "l"(cache + (offs8[3] + l_src_off)));
                        }
                        if (ok8[4] == 0) {
                            int dst_4 = l_dst0_0 + (32 * i8 + 16) * l_row_bytes;
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(dst_4), "l"(cache + (offs8[4] + l_src_off)));
                        }
                        if (ok8[5] == 0) {
                            int dst_5 = l_dst1_0 + (32 * i8 + 16) * l_row_bytes;
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(dst_5), "l"(cache + (offs8[5] + l_src_off)));
                        }
                        if (ok8[6] == 0) {
                            int dst_6 = l_dst0_0 + (32 * i8 + 24) * l_row_bytes;
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(dst_6), "l"(cache + (offs8[6] + l_src_off)));
                        }
                        if (ok8[7] == 0) {
                            int dst_7 = l_dst1_0 + (32 * i8 + 24) * l_row_bytes;
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(dst_7), "l"(cache + (offs8[7] + l_src_off)));
                        }
                    }
                    asm volatile(
                        "{\n\t"
                        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                        "}"
                        :: "r"(tok_full_addr) : "memory");
                    if (it_4 + 1 < tiles_per_split) {
                        int is_main_0 = 1;
                        if (tile_lo + it_4 + 1 >= num_main_tiles) {
                            is_main_0 = 0;
                        }
                        int tile_in_table_1 = ((is_main_0 != 0) ? tile_lo + it_4 + 1 : tile_lo + it_4 + 1 - num_main_tiles);
                        int table_width_2 = ((is_main_0 != 0) ? main_width : extra_width);
                        int* row_ptr_3 = ((is_main_0 != 0) ? (main_indices + (l_query_idx * main_index_stride)) : (extra_indices + (l_query_idx * extra_index_stride)));
                        int active_len_4 = table_width_2;
                        if (is_main_0 != 0) {
                            if (has_main_lengths != 0) {
                                active_len_4 = main_lengths[l_query_idx];
                            }
                        } else if (has_extra_lengths != 0) {
                            active_len_4 = extra_lengths[l_query_idx];
                        }
                        if (active_len_4 < 0) {
                            active_len_4 = 0;
                        }
                        if (active_len_4 > table_width_2) {
                            active_len_4 = table_width_2;
                        }
                        int r_5 = load_tid_0;
                        idx[0] = -1;
                        if (r_5 < 128) {
                            int col_2 = tile_in_table_1 * 128 + r_5;
                            if (col_2 < active_len_4) {
                                idx[0] = row_ptr_3[col_2];
                            }
                        }
                        int r_6 = load_tid_0 + 96;
                        idx[1] = -1;
                        if (r_6 < 128) {
                            int col_3 = tile_in_table_1 * 128 + r_6;
                            if (col_3 < active_len_4) {
                                idx[1] = row_ptr_3[col_3];
                            }
                        }
                        int page_shift_7 = ((is_main_0 != 0) ? main_page_shift : extra_page_shift);
                        long long page_stride_8 = ((is_main_0 != 0) ? main_page_stride : extra_page_stride);
                        offw[0] = 4294967295;
                        offw[1] = 4294967295;
                        offw[2] = 4294967295;
                        offw[3] = 4294967295;
                        if (idx[0] >= 0) {
                            int cpage_2 = idx[0] >> page_shift_7;
                            int cslot_2 = idx[0] - (cpage_2 << page_shift_7);
                            long long cbase_2 = (long long)cpage_2 * page_stride_8;
                            long long offd_2 = cbase_2 + (long long)(cslot_2 * 352);
                            long long offs_2 = cbase_2 + (long long)((1 << page_shift_7) * 352) + (long long)(cslot_2 * 32);
                            offw[0] = (unsigned int)offd_2;
                            offw[1] = (unsigned int)(offd_2 >> 32);
                            offw[2] = (unsigned int)offs_2;
                            offw[3] = (unsigned int)(offs_2 >> 32);
                        }
                        offw[4] = 4294967295;
                        offw[5] = 4294967295;
                        offw[6] = 4294967295;
                        offw[7] = 4294967295;
                        if (idx[1] >= 0) {
                            int cpage_3 = idx[1] >> page_shift_7;
                            int cslot_3 = idx[1] - (cpage_3 << page_shift_7);
                            long long cbase_3 = (long long)cpage_3 * page_stride_8;
                            long long offd_3 = cbase_3 + (long long)(cslot_3 * 352);
                            long long offs_3 = cbase_3 + (long long)((1 << page_shift_7) * 352) + (long long)(cslot_3 * 32);
                            offw[4] = (unsigned int)offd_3;
                            offw[5] = (unsigned int)(offd_3 >> 32);
                            offw[6] = (unsigned int)offs_3;
                            offw[7] = (unsigned int)(offs_3 >> 32);
                        }
                    }
                }
                int it_0_1 = 2 * it2_1 + 1;
                if (it_0_1 < tiles_per_split) {
                    mbarrier_wait_hint(tok_free_addr + 8, it2_1 & 1, 10000000);
                    int r_0_1 = load_tid_0;
                    if (r_0_1 < 128) {
                        smem_kw32[(it_0_1 & 1) * 256 + r_0_1] = idx[0];
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_rowoff_1_addr + (unsigned int)(r_0_1 * 16)), "r"(*reinterpret_cast<uint32_t*>(&offw[0])), "r"(*reinterpret_cast<uint32_t*>(&offw[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&offw[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&offw[(0) + 3])));
                    }
                    int r_2_1 = load_tid_0 + 96;
                    if (r_2_1 < 128) {
                        smem_kw32[(it_0_1 & 1) * 256 + r_2_1] = idx[1];
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_rowoff_1_addr + (unsigned int)(r_2_1 * 16)), "r"(*reinterpret_cast<uint32_t*>(&offw[4])), "r"(*reinterpret_cast<uint32_t*>(&offw[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&offw[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&offw[(4) + 3])));
                    }
                    unsigned int _vote_2 = __ballot_sync(0xFFFFFFFF, idx[0] >= 0);
                    unsigned int vb0_1 = _vote_2;
                    unsigned int _vote_3 = __ballot_sync(0xFFFFFFFF, idx[1] >= 0);
                    unsigned int vb1_1 = _vote_3;
                    if (lane == 0) {
                        int lw_1 = warp - 13;
                        smem_kw32[(it_0_1 & 1) * 256 + ((lw_1 == 0) ? 128 : ((lw_1 == 1) ? 129 : 130))] = (int)vb0_1;
                        if (lw_1 == 0) {
                            smem_kw32[(it_0_1 & 1) * 256 + 131] = (int)vb1_1;
                        }
                    }
                    asm volatile("barrier.sync 8, 96;" ::: "memory");
                    int is_main_3_1 = 1;
                    if (tile_lo + it_0_1 >= num_main_tiles) {
                        is_main_3_1 = 0;
                    }
                    uint8_t* cache_1 = ((is_main_3_1 != 0) ? (main_cache) : (extra_cache));
                    for (int i8_1 = 0; i8_1 < 4; i8_1++) {
                        long long offs8_1[8];
                        unsigned int ok8_1[8];
                        unsigned int w4_34[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_34[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_34[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_34[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_34[(0) + 3]))
                            : "r"(smem_rowoff_1_addr + (unsigned int)l_rowoff + (unsigned int)(i8_1 * 512)));
                        unsigned int lo_8 = ((l_kind < 2) ? w4_34[0] : w4_34[2]);
                        unsigned int hi_9 = ((l_kind < 2) ? w4_34[1] : w4_34[3]);
                        offs8_1[0] = (long long)hi_9 << 32 | (long long)lo_8;
                        ok8_1[0] = hi_9 & 2147483648u;
                        unsigned int w4_0_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_0_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_0_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_0_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_0_1[(0) + 3]))
                            : "r"(smem_rowoff_1_addr + (unsigned int)l_rowoff + (unsigned int)(i8_1 * 512) + 64));
                        unsigned int lo_1_2 = ((l_kind < 2) ? w4_0_1[0] : w4_0_1[2]);
                        unsigned int hi_2_2 = ((l_kind < 2) ? w4_0_1[1] : w4_0_1[3]);
                        offs8_1[1] = (long long)hi_2_2 << 32 | (long long)lo_1_2;
                        ok8_1[1] = hi_2_2 & 2147483648u;
                        unsigned int w4_3_2[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_3_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_3_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_3_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_3_2[(0) + 3]))
                            : "r"(smem_rowoff_1_addr + (unsigned int)l_rowoff + (unsigned int)(i8_1 * 512) + 128));
                        unsigned int lo_4_2 = ((l_kind < 2) ? w4_3_2[0] : w4_3_2[2]);
                        unsigned int hi_5_2 = ((l_kind < 2) ? w4_3_2[1] : w4_3_2[3]);
                        offs8_1[2] = (long long)hi_5_2 << 32 | (long long)lo_4_2;
                        ok8_1[2] = hi_5_2 & 2147483648u;
                        unsigned int w4_6_2[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_6_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_6_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_6_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_6_2[(0) + 3]))
                            : "r"(smem_rowoff_1_addr + (unsigned int)l_rowoff + (unsigned int)(i8_1 * 512) + 192));
                        unsigned int lo_7_1 = ((l_kind < 2) ? w4_6_2[0] : w4_6_2[2]);
                        unsigned int hi_8_1 = ((l_kind < 2) ? w4_6_2[1] : w4_6_2[3]);
                        offs8_1[3] = (long long)hi_8_1 << 32 | (long long)lo_7_1;
                        ok8_1[3] = hi_8_1 & 2147483648u;
                        unsigned int w4_9_2[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_9_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_9_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_9_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_9_2[(0) + 3]))
                            : "r"(smem_rowoff_1_addr + (unsigned int)l_rowoff + (unsigned int)(i8_1 * 512) + 256));
                        unsigned int lo_10_5 = ((l_kind < 2) ? w4_9_2[0] : w4_9_2[2]);
                        unsigned int hi_11_5 = ((l_kind < 2) ? w4_9_2[1] : w4_9_2[3]);
                        offs8_1[4] = (long long)hi_11_5 << 32 | (long long)lo_10_5;
                        ok8_1[4] = hi_11_5 & 2147483648u;
                        unsigned int w4_12_2[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_12_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12_2[(0) + 3]))
                            : "r"(smem_rowoff_1_addr + (unsigned int)l_rowoff + (unsigned int)(i8_1 * 512) + 320));
                        unsigned int lo_13_5 = ((l_kind < 2) ? w4_12_2[0] : w4_12_2[2]);
                        unsigned int hi_14_5 = ((l_kind < 2) ? w4_12_2[1] : w4_12_2[3]);
                        offs8_1[5] = (long long)hi_14_5 << 32 | (long long)lo_13_5;
                        ok8_1[5] = hi_14_5 & 2147483648u;
                        unsigned int w4_15_2[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_15_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_15_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_15_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_15_2[(0) + 3]))
                            : "r"(smem_rowoff_1_addr + (unsigned int)l_rowoff + (unsigned int)(i8_1 * 512) + 384));
                        unsigned int lo_16_5 = ((l_kind < 2) ? w4_15_2[0] : w4_15_2[2]);
                        unsigned int hi_17_5 = ((l_kind < 2) ? w4_15_2[1] : w4_15_2[3]);
                        offs8_1[6] = (long long)hi_17_5 << 32 | (long long)lo_16_5;
                        ok8_1[6] = hi_17_5 & 2147483648u;
                        unsigned int w4_18_2[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_18_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_18_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_18_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_18_2[(0) + 3]))
                            : "r"(smem_rowoff_1_addr + (unsigned int)l_rowoff + (unsigned int)(i8_1 * 512) + 448));
                        unsigned int lo_19_5 = ((l_kind < 2) ? w4_18_2[0] : w4_18_2[2]);
                        unsigned int hi_20_5 = ((l_kind < 2) ? w4_18_2[1] : w4_18_2[3]);
                        offs8_1[7] = (long long)hi_20_5 << 32 | (long long)lo_19_5;
                        ok8_1[7] = hi_20_5 & 2147483648u;
                        if (ok8_1[0] == 0) {
                            int dst_8 = l_dst0_1 + 32 * i8_1 * l_row_bytes;
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(dst_8), "l"(cache_1 + (offs8_1[0] + l_src_off)));
                        }
                        if (ok8_1[1] == 0) {
                            int dst_9 = l_dst1_1 + 32 * i8_1 * l_row_bytes;
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(dst_9), "l"(cache_1 + (offs8_1[1] + l_src_off)));
                        }
                        if (ok8_1[2] == 0) {
                            int dst_10 = l_dst0_1 + (32 * i8_1 + 8) * l_row_bytes;
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(dst_10), "l"(cache_1 + (offs8_1[2] + l_src_off)));
                        }
                        if (ok8_1[3] == 0) {
                            int dst_11 = l_dst1_1 + (32 * i8_1 + 8) * l_row_bytes;
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(dst_11), "l"(cache_1 + (offs8_1[3] + l_src_off)));
                        }
                        if (ok8_1[4] == 0) {
                            int dst_12 = l_dst0_1 + (32 * i8_1 + 16) * l_row_bytes;
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(dst_12), "l"(cache_1 + (offs8_1[4] + l_src_off)));
                        }
                        if (ok8_1[5] == 0) {
                            int dst_13 = l_dst1_1 + (32 * i8_1 + 16) * l_row_bytes;
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(dst_13), "l"(cache_1 + (offs8_1[5] + l_src_off)));
                        }
                        if (ok8_1[6] == 0) {
                            int dst_14 = l_dst0_1 + (32 * i8_1 + 24) * l_row_bytes;
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(dst_14), "l"(cache_1 + (offs8_1[6] + l_src_off)));
                        }
                        if (ok8_1[7] == 0) {
                            int dst_15 = l_dst1_1 + (32 * i8_1 + 24) * l_row_bytes;
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(dst_15), "l"(cache_1 + (offs8_1[7] + l_src_off)));
                        }
                    }
                    asm volatile(
                        "{\n\t"
                        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                        "}"
                        :: "r"(tok_full_addr + 8) : "memory");
                    if (it_0_1 + 1 < tiles_per_split) {
                        int is_main_0_1 = 1;
                        if (tile_lo + it_0_1 + 1 >= num_main_tiles) {
                            is_main_0_1 = 0;
                        }
                        int tile_in_table_1_1 = ((is_main_0_1 != 0) ? tile_lo + it_0_1 + 1 : tile_lo + it_0_1 + 1 - num_main_tiles);
                        int table_width_2_1 = ((is_main_0_1 != 0) ? main_width : extra_width);
                        int* row_ptr_3_1 = ((is_main_0_1 != 0) ? (main_indices + (l_query_idx * main_index_stride)) : (extra_indices + (l_query_idx * extra_index_stride)));
                        int active_len_4_1 = table_width_2_1;
                        if (is_main_0_1 != 0) {
                            if (has_main_lengths != 0) {
                                active_len_4_1 = main_lengths[l_query_idx];
                            }
                        } else if (has_extra_lengths != 0) {
                            active_len_4_1 = extra_lengths[l_query_idx];
                        }
                        if (active_len_4_1 < 0) {
                            active_len_4_1 = 0;
                        }
                        if (active_len_4_1 > table_width_2_1) {
                            active_len_4_1 = table_width_2_1;
                        }
                        int r_5_1 = load_tid_0;
                        idx[0] = -1;
                        if (r_5_1 < 128) {
                            int col_4 = tile_in_table_1_1 * 128 + r_5_1;
                            if (col_4 < active_len_4_1) {
                                idx[0] = row_ptr_3_1[col_4];
                            }
                        }
                        int r_6_1 = load_tid_0 + 96;
                        idx[1] = -1;
                        if (r_6_1 < 128) {
                            int col_5 = tile_in_table_1_1 * 128 + r_6_1;
                            if (col_5 < active_len_4_1) {
                                idx[1] = row_ptr_3_1[col_5];
                            }
                        }
                        int page_shift_7_1 = ((is_main_0_1 != 0) ? main_page_shift : extra_page_shift);
                        long long page_stride_8_1 = ((is_main_0_1 != 0) ? main_page_stride : extra_page_stride);
                        offw[0] = 4294967295;
                        offw[1] = 4294967295;
                        offw[2] = 4294967295;
                        offw[3] = 4294967295;
                        if (idx[0] >= 0) {
                            int cpage_4 = idx[0] >> page_shift_7_1;
                            int cslot_4 = idx[0] - (cpage_4 << page_shift_7_1);
                            long long cbase_4 = (long long)cpage_4 * page_stride_8_1;
                            long long offd_4 = cbase_4 + (long long)(cslot_4 * 352);
                            long long offs_4 = cbase_4 + (long long)((1 << page_shift_7_1) * 352) + (long long)(cslot_4 * 32);
                            offw[0] = (unsigned int)offd_4;
                            offw[1] = (unsigned int)(offd_4 >> 32);
                            offw[2] = (unsigned int)offs_4;
                            offw[3] = (unsigned int)(offs_4 >> 32);
                        }
                        offw[4] = 4294967295;
                        offw[5] = 4294967295;
                        offw[6] = 4294967295;
                        offw[7] = 4294967295;
                        if (idx[1] >= 0) {
                            int cpage_5 = idx[1] >> page_shift_7_1;
                            int cslot_5 = idx[1] - (cpage_5 << page_shift_7_1);
                            long long cbase_5 = (long long)cpage_5 * page_stride_8_1;
                            long long offd_5 = cbase_5 + (long long)(cslot_5 * 352);
                            long long offs_5 = cbase_5 + (long long)((1 << page_shift_7_1) * 352) + (long long)(cslot_5 * 32);
                            offw[4] = (unsigned int)offd_5;
                            offw[5] = (unsigned int)(offd_5 >> 32);
                            offw[6] = (unsigned int)offs_5;
                            offw[7] = (unsigned int)(offs_5 >> 32);
                        }
                    }
                }
            }
            asm volatile("cp.async.wait_group 0;");
        }
    }

    // Cleanup
}

} // extern "C"
