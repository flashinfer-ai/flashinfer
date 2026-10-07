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
#define TMEM_TMEM_S_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QF4_OFF 1024
#define SMEM_SMEM_QF4_STAGE_BYTES 4096
#define SMEM_SMEM_QF4_STRIDE 4096
#define SMEM_SMEM_QSF_OFF 9216
#define SMEM_SMEM_QSF_STAGE_BYTES 2048
#define SMEM_SMEM_QSF_STRIDE 2048
#define SMEM_SMEM_QSF32_OFF 9216
#define SMEM_SMEM_QSF32_STAGE_BYTES 4096
#define SMEM_SMEM_QSF32_STRIDE 4096
#define SMEM_SMEM_QROPE_OFF 13312
#define SMEM_SMEM_QSTAGE_OFF 74752
#define SMEM_SMEM_OSTAGE_OFF 21504
#define SMEM_SMEM_OSTAGE_STAGE_BYTES 16384
#define SMEM_SMEM_OSTAGE_STRIDE 16384
#define SMEM_SMEM_KSF_OFF 17408
#define SMEM_SMEM_KSF_STAGE_BYTES 2048
#define SMEM_SMEM_KSF_STRIDE 2048
#define SMEM_SMEM_KSF32_OFF 17408
#define SMEM_SMEM_KSF32_STAGE_BYTES 4096
#define SMEM_SMEM_KSF32_STRIDE 4096
#define SMEM_SMEM_KF4_0_OFF 21504
#define SMEM_SMEM_KF4_0_STAGE_BYTES 16384
#define SMEM_SMEM_KF4_0_STRIDE 16384
#define SMEM_SMEM_KROPE_0_OFF 54272
#define SMEM_SMEM_KROPE_0_STAGE_BYTES 16384
#define SMEM_SMEM_KROPE_0_STRIDE 16384
#define SMEM_SMEM_SFS_0_OFF 70656
#define SMEM_SMEM_SFS_0_STAGE_BYTES 4096
#define SMEM_SMEM_SFS_0_STRIDE 4096
#define SMEM_SMEM_SFS32_0_OFF 70656
#define SMEM_SMEM_SFS32_0_STAGE_BYTES 4096
#define SMEM_SMEM_SFS32_0_STRIDE 4096
#define SMEM_SMEM_TOK_0_OFF 199712
#define SMEM_SMEM_TOK_0_STAGE_BYTES 512
#define SMEM_SMEM_TOK_0_STRIDE 512
#define SMEM_SMEM_KF4_1_OFF 74752
#define SMEM_SMEM_KF4_1_STAGE_BYTES 16384
#define SMEM_SMEM_KF4_1_STRIDE 16384
#define SMEM_SMEM_KROPE_1_OFF 107520
#define SMEM_SMEM_KROPE_1_STAGE_BYTES 16384
#define SMEM_SMEM_KROPE_1_STRIDE 16384
#define SMEM_SMEM_SFS_1_OFF 123904
#define SMEM_SMEM_SFS_1_STAGE_BYTES 4096
#define SMEM_SMEM_SFS_1_STRIDE 4096
#define SMEM_SMEM_SFS32_1_OFF 123904
#define SMEM_SMEM_SFS32_1_STAGE_BYTES 4096
#define SMEM_SMEM_SFS32_1_STRIDE 4096
#define SMEM_SMEM_TOK_1_OFF 200224
#define SMEM_SMEM_TOK_1_STAGE_BYTES 512
#define SMEM_SMEM_TOK_1_STRIDE 512
#define SMEM_SMEM_TOKR_0_OFF 199712
#define SMEM_SMEM_TOKR_0_STAGE_BYTES 512
#define SMEM_SMEM_TOKR_0_STRIDE 512
#define SMEM_SMEM_TOKR_1_OFF 200224
#define SMEM_SMEM_TOKR_1_STAGE_BYTES 512
#define SMEM_SMEM_TOKR_1_STRIDE 512
#define SMEM_SMEM_V_OFF 128000
#define SMEM_SMEM_V_STAGE_BYTES 16384
#define SMEM_SMEM_V_STRIDE 16384
#define SMEM_SMEM_P_OFF 193536
#define SMEM_SMEM_P_STAGE_BYTES 4096
#define SMEM_SMEM_P_STRIDE 4096
#define SMEM_SMEM_RCPTAB_OFF 197632
#define SMEM_SMEM_RCPTAB_STAGE_BYTES 64
#define SMEM_SMEM_RCPTAB_STRIDE 64
#define SMEM_SMEM_MASK_OFF 199680
#define SMEM_SMEM_MASK_STAGE_BYTES 16
#define SMEM_SMEM_MASK_STRIDE 16
#define SMEM_SMEM_PMAX_OFF 201248
#define SMEM_SMEM_QF4B_OFF 1024
#define SMEM_SMEM_QF4B_STRIDE 4096
#define SMEM_SMEM_QROPE_B_OFF 13312
#define SMEM_SMEM_PB_OFF 193536
#define THREADS 512
#define LAUNCH_MIN_BLOCKS 1

namespace cake::dsv4_nvfp4 {

template <int TILE_N>
__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
decode_pv(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_out, const __grid_constant__ CUtensorMap tmap_g4d, const __grid_constant__ CUtensorMap tmap_g4f, const __grid_constant__ CUtensorMap tmap_g4dx, const __grid_constant__ CUtensorMap tmap_g4fx, __nv_bfloat16* __restrict__ q_rows, uint8_t* __restrict__ main_cache, uint8_t* __restrict__ extra_cache, int* __restrict__ main_indices, int* __restrict__ extra_indices, int* __restrict__ main_lengths, int* __restrict__ extra_lengths, float* __restrict__ sinks, float* __restrict__ bmm1_scale, float* __restrict__ bmm2_scale, __nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_lse, __nv_bfloat16* __restrict__ O, float* __restrict__ lse_out, int num_heads, int num_head_tiles, int num_splits, int num_main_tiles, int tiles_per_split, int main_width, int extra_width, int main_index_stride, int extra_index_stride, int has_main_lengths, int has_extra_lengths, int main_page_shift, int extra_page_shift, long long main_page_stride, long long extra_page_stride, int has_sinks, float lse_partial_scale, float lse_scale)
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
    uint8_t* smem_qf4 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_qf4_addr = smem + 1024;
    uint8_t* smem_qsf = reinterpret_cast<uint8_t*>(smem_raw + 9216);
    const int smem_qsf_addr = smem + 9216;
    unsigned int* smem_qsf32 = reinterpret_cast<unsigned int*>(smem_raw + 9216);
    const int smem_qsf32_addr = smem + 9216;
    __nv_bfloat16* smem_qrope = reinterpret_cast<__nv_bfloat16*>(smem_raw + 13312);
    const int smem_qrope_addr = smem + 13312;
    __nv_bfloat16* smem_qstage = reinterpret_cast<__nv_bfloat16*>(smem_raw + 74752);
    const int smem_qstage_addr = smem + 74752;
    __nv_bfloat16* smem_ostage = reinterpret_cast<__nv_bfloat16*>(smem_raw + 21504);
    const int smem_ostage_addr = smem + 21504;
    uint8_t* smem_ksf = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_ksf_addr = smem + 17408;
    unsigned int* smem_ksf32 = reinterpret_cast<unsigned int*>(smem_raw + 17408);
    const int smem_ksf32_addr = smem + 17408;
    uint8_t* smem_kf4_0 = reinterpret_cast<uint8_t*>(smem_raw + 21504);
    const int smem_kf4_0_addr = smem + 21504;
    __nv_bfloat16* smem_krope_0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 54272);
    const int smem_krope_0_addr = smem + 54272;
    uint8_t* smem_sfs_0 = reinterpret_cast<uint8_t*>(smem_raw + 70656);
    const int smem_sfs_0_addr = smem + 70656;
    unsigned int* smem_sfs32_0 = reinterpret_cast<unsigned int*>(smem_raw + 70656);
    const int smem_sfs32_0_addr = smem + 70656;
    int* smem_tok_0 = reinterpret_cast<int*>(smem_raw + 199712);
    const int smem_tok_0_addr = smem + 199712;
    uint8_t* smem_kf4_1 = reinterpret_cast<uint8_t*>(smem_raw + 74752);
    const int smem_kf4_1_addr = smem + 74752;
    __nv_bfloat16* smem_krope_1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 107520);
    const int smem_krope_1_addr = smem + 107520;
    uint8_t* smem_sfs_1 = reinterpret_cast<uint8_t*>(smem_raw + 123904);
    const int smem_sfs_1_addr = smem + 123904;
    unsigned int* smem_sfs32_1 = reinterpret_cast<unsigned int*>(smem_raw + 123904);
    const int smem_sfs32_1_addr = smem + 123904;
    int* smem_tok_1 = reinterpret_cast<int*>(smem_raw + 200224);
    const int smem_tok_1_addr = smem + 200224;
    int* smem_tokr_0 = reinterpret_cast<int*>(smem_raw + 199712);
    const int smem_tokr_0_addr = smem + 199712;
    int* smem_tokr_1 = reinterpret_cast<int*>(smem_raw + 200224);
    const int smem_tokr_1_addr = smem + 200224;
    uint8_t* smem_v = reinterpret_cast<uint8_t*>(smem_raw + 128000);
    const int smem_v_addr = smem + 128000;
    uint8_t* smem_p = reinterpret_cast<uint8_t*>(smem_raw + 193536);
    const int smem_p_addr = smem + 193536;
    unsigned int* smem_rcptab = reinterpret_cast<unsigned int*>(smem_raw + 197632);
    const int smem_rcptab_addr = smem + 197632;
    unsigned int* smem_mask = reinterpret_cast<unsigned int*>(smem_raw + 199680);
    const int smem_mask_addr = smem + 199680;
    float* smem_pmax = reinterpret_cast<float*>(smem_raw + 201248);
    const int smem_pmax_addr = smem + 201248;
    float* smem_psum = reinterpret_cast<float*>(smem_raw + (24 * TILE_N + 201248));
    const int smem_psum_addr = smem + (24 * TILE_N + 201248);
    float* smem_rsum = reinterpret_cast<float*>(smem_raw + (48 * TILE_N + 201248));
    const int smem_rsum_addr = smem + (48 * TILE_N + 201248);
    uint8_t* smem_qf4b = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_qf4b_addr = smem + 1024;
    __nv_bfloat16* smem_qrope_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 13312);
    const int smem_qrope_b_addr = smem + 13312;
    uint8_t* smem_pb = reinterpret_cast<uint8_t*>(smem_raw + 193536);
    const int smem_pb_addr = smem + 193536;
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
            // tok_free: 2 barriers, init_count=384
            mbarrier_init(smem + 128, 384);
            mbarrier_init(smem + 136, 384);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    __syncwarp();
    // TMEM alloc (256 columns, (5 * TILE_N + 64) used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 144);
    if (warp == 0) {
        int _tmem_hold = smem + 144;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    const int taddr = tmem_addr_storage[0];
    // Kernel post-init ops
    const int tmem_tmem_s = taddr;
    const int tmem_tmem_o0 = taddr + TILE_N;
    const int tmem_tmem_o1 = taddr + 2 * TILE_N;
    const int tmem_tmem_o2 = taddr + 3 * TILE_N;
    const int tmem_tmem_o3 = taddr + 4 * TILE_N;
    const int tmem_tmem_sfa0 = taddr + 5 * TILE_N;
    const int tmem_tmem_sfa1 = taddr + (5 * TILE_N + 16);
    const int tmem_tmem_sfb0 = taddr + (5 * TILE_N + 32);
    const int tmem_tmem_sfb1 = taddr + (5 * TILE_N + 48);
    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
    }
    // ---- Role: compute0 ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 144;");
        // compute0_main
        {
            const int local_warp = warp;
            int o_chunk = 0;
            int work_idx = blockIdx.x;
            int head_tile = work_idx % num_head_tiles;
            int split_work = work_idx / num_head_tiles;
            int split_idx = split_work % num_splits;
            int query_idx = split_work / num_splits;
            int head_base = head_tile * 128;
            const int row = local_warp * 32 + lane;
            const int tmem_row_origin = local_warp * 32;
            if (local_warp == 0) {
                if (lane == 0) {
                    smem_rcptab[0] = 1065353216;
                    smem_rcptab[1] = 1063489081;
                    smem_rcptab[2] = 1061997773;
                    smem_rcptab[3] = 1060777612;
                    smem_rcptab[4] = 1059760811;
                    smem_rcptab[5] = 1058900441;
                    smem_rcptab[6] = 1058162981;
                    smem_rcptab[7] = 1057523849;
                    smem_rcptab[8] = 0;
                    smem_rcptab[9] = 1065353216;
                    smem_rcptab[10] = 1056964608;
                    smem_rcptab[11] = 1051372203;
                    smem_rcptab[12] = 1048576000;
                    smem_rcptab[13] = 1045220557;
                    smem_rcptab[14] = 1042983595;
                    smem_rcptab[15] = 1041385765;
                }
            }
            asm volatile("barrier.sync 9, 384;" ::: "memory");
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
            const int q_warp = warp;
            for (int i = 0; i < 1; i++) {
                int unit = q_warp + 12 * i;
                if (unit < 7) {
                    int q_block = unit / 7;
                    int kset = unit - q_block * 7;
                    int q_row = q_block * 32 + lane;
                    if (head_base + q_block * 32 < num_heads && q_row < TILE_N) {
                        int q_row_addr = smem_qstage_addr + (unsigned int)(kset * 128 * TILE_N) + (unsigned int)(q_row * 128);
                        unsigned int sf_word = 0;
                        for (int bp = 0; bp < 2; bp++) {
                            unsigned int words[4];
                            unsigned int qa[4];
                            unsigned int qb[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&qa[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa[(0) + 3]))
                                : "r"(q_row_addr + (4 * bp ^ q_row % 8) * 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&qb[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb[(0) + 3]))
                                : "r"(q_row_addr + (4 * bp + 1 ^ q_row % 8) * 16));
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
                            float inv = 0.0f;
                            if (sc_exp == 0) {
                                inv = __uint_as_float(smem_rcptab[8 + sc_man]) * 512.0f;
                            } else {
                                inv = __uint_as_float(smem_rcptab[sc_man]) * __uint_as_float(134 - sc_exp << 23);
                            }
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
                            sf_word = sf_word | sc_byte << (unsigned int)(8 * (2 * bp));
                            unsigned int qa_0[4];
                            unsigned int qb_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&qa_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0[(0) + 3]))
                                : "r"(q_row_addr + (4 * bp + 2 ^ q_row % 8) * 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&qb_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1[(0) + 3]))
                                : "r"(q_row_addr + (4 * bp + 2 + 1 ^ q_row % 8) * 16));
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
                            float inv_11 = 0.0f;
                            if (sc_exp_9 == 0) {
                                inv_11 = __uint_as_float(smem_rcptab[8 + sc_man_10]) * 512.0f;
                            } else {
                                inv_11 = __uint_as_float(smem_rcptab[sc_man_10]) * __uint_as_float(134 - sc_exp_9 << 23);
                            }
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_1 = {inv_11, inv_11};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_2)[_ls], _scale2_1);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++) {
                                qv_2[_ls] = qv_2[_ls] * inv_11;
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
                            sf_word = sf_word | sc_byte_8 << (unsigned int)(8 * (2 * bp + 1));
                            int chunk = 2 * kset + bp;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(smem_qf4_addr + (unsigned int)(chunk / 8 * 4096 + (q_row * 128 + (chunk % 8 * 16 ^ q_row % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&words[0])), "r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3])));
                        }
                        smem_qsf32[kset / 4 * 2048 + q_row % 32 / 8 * 512 + kset % 4 * 128 + q_row % 8 * 16 + q_row / 32 % 4 * 4 >> 2] = sf_word;
                    } else {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_qf4_addr + (unsigned int)(2 * kset / 8 * 4096 + (q_row * 128 + (2 * kset % 8 * 16 ^ q_row % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_qf4_addr + (unsigned int)((2 * kset + 1) / 8 * 4096 + (q_row * 128 + ((2 * kset + 1) % 8 * 16 ^ q_row % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                        smem_qsf32[kset / 4 * 2048 + q_row % 32 / 8 * 512 + kset % 4 * 128 + q_row % 8 * 16 + q_row / 32 % 4 * 4 >> 2] = 0;
                    }
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(q_ready_addr);
            asm volatile("barrier.sync 9, 384;" ::: "memory");
            mbarrier_arrive(tok_free_addr + 8);
            float softmax_scale_log2 = bmm1_scale[0] * 1.4426950408889634f;
            float output_scale = bmm2_scale[0];
            int strip_0 = smem_sfs_0_addr + (unsigned int)(row * 32);
            int strip_1 = smem_sfs_1_addr + (unsigned int)(row * 32);
            float m_run = -CAKE_INF;
            float l_run = 0.0f;
            float r_run = 0.0f;
            float sink_lane = -CAKE_INF;
            if (lane < (TILE_N / 2)) {
                if (has_sinks != 0 && split_idx == 0 && head_base + lane < num_heads) {
                    sink_lane = sinks[head_base + lane] * 1.4426950408889634f;
                }
            }
            for (int it2 = 0; it2 < (tiles_per_split + 1) / 2; it2++) {
                int par2 = it2 & 1;
                int it = 2 * it2;
                if (it < tiles_per_split) {
                    mbarrier_wait_hint(tok_full_addr, it2 & 1, 10000000);
                    int raw_index = smem_tok_0[row];
                    int valid = 1;
                    if (raw_index < 0) {
                        valid = 0;
                    }
                    if (valid != 0) {
                        {
                            unsigned int sfw[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&sfw[0])), "=r"(*reinterpret_cast<uint32_t*>(&sfw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sfw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sfw[(0) + 3]))
                                : "r"(strip_0));
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
                        mbarrier_wait_hint(o_full_addr, 1, 10000000);
                        mbarrier_wait_hint(o_full_addr + 8, 1, 10000000);
                        mbarrier_wait_hint(o_full_addr + 16, 1, 10000000);
                        mbarrier_wait_hint(o_full_addr + 24, 1, 10000000);
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
                                    : "r"(smem_kf4_0_addr + (unsigned int)(vchunk / 8 * 16384) + (unsigned int)(row * 128 + (vchunk % 8 * 16 ^ row % 8 * 16))));
                            }
                            {
                                sfw32 = smem_sfs32_0[row * 8 + 8 * o_chunk];
                            }
                            unsigned int scale = sfw32 & 255;
                            {
                                v8[0] = cake_dsv4_qmul4_portable<5>(kraw4[0], scale);
                            }
                            {
                                v8[1] = cake_dsv4_qmul4_portable<6>(kraw4[0], scale);
                            }
                            {
                                v8[2] = cake_dsv4_qmul4_portable<5>(kraw4[1], scale);
                            }
                            {
                                v8[3] = cake_dsv4_qmul4_portable<6>(kraw4[1], scale);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(row * 128 + (0 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8[0])), "r"(*reinterpret_cast<uint32_t*>(&v8[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8[(0) + 3])));
                        int vblock_0 = 32 * o_chunk + 1;
                        unsigned int v8_1[4];
                        {
                            unsigned int scale_1 = sfw32 >> 8 & 255;
                            {
                                v8_1[0] = cake_dsv4_qmul4_portable<5>(kraw4[2], scale_1);
                            }
                            {
                                v8_1[1] = cake_dsv4_qmul4_portable<6>(kraw4[2], scale_1);
                            }
                            {
                                v8_1[2] = cake_dsv4_qmul4_portable<5>(kraw4[3], scale_1);
                            }
                            {
                                v8_1[3] = cake_dsv4_qmul4_portable<6>(kraw4[3], scale_1);
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
                                    : "r"(smem_kf4_0_addr + (unsigned int)(vchunk_1 / 8 * 16384) + (unsigned int)(row * 128 + (vchunk_1 % 8 * 16 ^ row % 8 * 16))));
                            }
                            unsigned int scale_2 = sfw32 >> 16 & 255;
                            {
                                v8_3[0] = cake_dsv4_qmul4_portable<5>(kraw4[0], scale_2);
                            }
                            {
                                v8_3[1] = cake_dsv4_qmul4_portable<6>(kraw4[0], scale_2);
                            }
                            {
                                v8_3[2] = cake_dsv4_qmul4_portable<5>(kraw4[1], scale_2);
                            }
                            {
                                v8_3[3] = cake_dsv4_qmul4_portable<6>(kraw4[1], scale_2);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(row * 128 + (32 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_3[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_3[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_3[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_3[(0) + 3])));
                        int vblock_4 = 32 * o_chunk + 3;
                        unsigned int v8_5[4];
                        {
                            unsigned int scale_3 = sfw32 >> 24 & 255;
                            {
                                v8_5[0] = cake_dsv4_qmul4_portable<5>(kraw4[2], scale_3);
                            }
                            {
                                v8_5[1] = cake_dsv4_qmul4_portable<6>(kraw4[2], scale_3);
                            }
                            {
                                v8_5[2] = cake_dsv4_qmul4_portable<5>(kraw4[3], scale_3);
                            }
                            {
                                v8_5[3] = cake_dsv4_qmul4_portable<6>(kraw4[3], scale_3);
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
                                    : "r"(smem_kf4_0_addr + (unsigned int)(vchunk_2 / 8 * 16384) + (unsigned int)(row * 128 + (vchunk_2 % 8 * 16 ^ row % 8 * 16))));
                            }
                            {
                                sfw32 = smem_sfs32_0[row * 8 + 8 * o_chunk + 1];
                            }
                            unsigned int scale_4 = sfw32 & 255;
                            {
                                v8_7[0] = cake_dsv4_qmul4_portable<5>(kraw4[0], scale_4);
                            }
                            {
                                v8_7[1] = cake_dsv4_qmul4_portable<6>(kraw4[0], scale_4);
                            }
                            {
                                v8_7[2] = cake_dsv4_qmul4_portable<5>(kraw4[1], scale_4);
                            }
                            {
                                v8_7[3] = cake_dsv4_qmul4_portable<6>(kraw4[1], scale_4);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(row * 128 + (64 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_7[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_7[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_7[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_7[(0) + 3])));
                        int vblock_8 = 32 * o_chunk + 5;
                        unsigned int v8_9[4];
                        {
                            unsigned int scale_5 = sfw32 >> 8 & 255;
                            {
                                v8_9[0] = cake_dsv4_qmul4_portable<5>(kraw4[2], scale_5);
                            }
                            {
                                v8_9[1] = cake_dsv4_qmul4_portable<6>(kraw4[2], scale_5);
                            }
                            {
                                v8_9[2] = cake_dsv4_qmul4_portable<5>(kraw4[3], scale_5);
                            }
                            {
                                v8_9[3] = cake_dsv4_qmul4_portable<6>(kraw4[3], scale_5);
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
                                    : "r"(smem_kf4_0_addr + (unsigned int)(vchunk_3 / 8 * 16384) + (unsigned int)(row * 128 + (vchunk_3 % 8 * 16 ^ row % 8 * 16))));
                            }
                            unsigned int scale_6 = sfw32 >> 16 & 255;
                            {
                                v8_11[0] = cake_dsv4_qmul4_portable<5>(kraw4[0], scale_6);
                            }
                            {
                                v8_11[1] = cake_dsv4_qmul4_portable<6>(kraw4[0], scale_6);
                            }
                            {
                                v8_11[2] = cake_dsv4_qmul4_portable<5>(kraw4[1], scale_6);
                            }
                            {
                                v8_11[3] = cake_dsv4_qmul4_portable<6>(kraw4[1], scale_6);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(row * 128 + (96 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_11[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_11[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_11[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_11[(0) + 3])));
                        int vblock_12 = 32 * o_chunk + 7;
                        unsigned int v8_13[4];
                        {
                            unsigned int scale_7 = sfw32 >> 24 & 255;
                            {
                                v8_13[0] = cake_dsv4_qmul4_portable<5>(kraw4[2], scale_7);
                            }
                            {
                                v8_13[1] = cake_dsv4_qmul4_portable<6>(kraw4[2], scale_7);
                            }
                            {
                                v8_13[2] = cake_dsv4_qmul4_portable<5>(kraw4[3], scale_7);
                            }
                            {
                                v8_13[3] = cake_dsv4_qmul4_portable<6>(kraw4[3], scale_7);
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
                                    : "r"(smem_kf4_0_addr + (unsigned int)(vchunk_4 / 8 * 16384) + (unsigned int)(row * 128 + (vchunk_4 % 8 * 16 ^ row % 8 * 16))));
                            }
                            {
                                sfw32 = smem_sfs32_0[row * 8 + 8 * o_chunk + 2];
                            }
                            unsigned int scale_8 = sfw32 & 255;
                            {
                                v8_15[0] = cake_dsv4_qmul4_portable<5>(kraw4[0], scale_8);
                            }
                            {
                                v8_15[1] = cake_dsv4_qmul4_portable<6>(kraw4[0], scale_8);
                            }
                            {
                                v8_15[2] = cake_dsv4_qmul4_portable<5>(kraw4[1], scale_8);
                            }
                            {
                                v8_15[3] = cake_dsv4_qmul4_portable<6>(kraw4[1], scale_8);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(16384 + (row * 128 + (0 ^ row % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_15[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_15[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_15[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_15[(0) + 3])));
                        int vblock_16 = 32 * o_chunk + 9;
                        unsigned int v8_17[4];
                        {
                            unsigned int scale_9 = sfw32 >> 8 & 255;
                            {
                                v8_17[0] = cake_dsv4_qmul4_portable<5>(kraw4[2], scale_9);
                            }
                            {
                                v8_17[1] = cake_dsv4_qmul4_portable<6>(kraw4[2], scale_9);
                            }
                            {
                                v8_17[2] = cake_dsv4_qmul4_portable<5>(kraw4[3], scale_9);
                            }
                            {
                                v8_17[3] = cake_dsv4_qmul4_portable<6>(kraw4[3], scale_9);
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
                                    : "r"(smem_kf4_0_addr + (unsigned int)(vchunk_5 / 8 * 16384) + (unsigned int)(row * 128 + (vchunk_5 % 8 * 16 ^ row % 8 * 16))));
                            }
                            unsigned int scale_10 = sfw32 >> 16 & 255;
                            {
                                v8_19[0] = cake_dsv4_qmul4_portable<5>(kraw4[0], scale_10);
                            }
                            {
                                v8_19[1] = cake_dsv4_qmul4_portable<6>(kraw4[0], scale_10);
                            }
                            {
                                v8_19[2] = cake_dsv4_qmul4_portable<5>(kraw4[1], scale_10);
                            }
                            {
                                v8_19[3] = cake_dsv4_qmul4_portable<6>(kraw4[1], scale_10);
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
                    mbarrier_wait_hint(s_full_addr, 0, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (it + 2 < tiles_per_split) {
                        mbarrier_arrive(tok_free_addr);
                    }
                    {
                        float score_values[(TILE_N / 2)];
                        if constexpr (TILE_N == 16) {
                            tmem_ld_x8(&score_values[0], taddr + (unsigned int)(tmem_row_origin << 16));
                        } else {
                            tmem_ld_x16(&score_values[0], taddr + (unsigned int)(tmem_row_origin << 16));
                        }
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        if (valid == 0) {
                            score_values[0] = -CAKE_INF;
                            score_values[1] = -CAKE_INF;
                            score_values[2] = -CAKE_INF;
                            score_values[3] = -CAKE_INF;
                            score_values[4] = -CAKE_INF;
                            score_values[5] = -CAKE_INF;
                            score_values[6] = -CAKE_INF;
                            score_values[7] = -CAKE_INF;
                            if constexpr (TILE_N == 32) {
                                score_values[8] = -CAKE_INF;
                                score_values[9] = -CAKE_INF;
                                score_values[10] = -CAKE_INF;
                                score_values[11] = -CAKE_INF;
                                score_values[12] = -CAKE_INF;
                                score_values[13] = -CAKE_INF;
                                score_values[14] = -CAKE_INF;
                                score_values[15] = -CAKE_INF;
                            }
                        }
                        float tr_vals[(TILE_N / 2)];
                        if constexpr (TILE_N == 32) {
                            tr_vals[0] = score_values[0];
                            tr_vals[1] = score_values[1];
                            tr_vals[2] = score_values[2];
                            tr_vals[3] = score_values[3];
                            tr_vals[4] = score_values[4];
                            tr_vals[5] = score_values[5];
                            tr_vals[6] = score_values[6];
                            tr_vals[7] = score_values[7];
                        }
                        tr_vals[(TILE_N / 2 + -8)] = score_values[(TILE_N / 2 + -8)];
                        tr_vals[(TILE_N / 2 + -7)] = score_values[(TILE_N / 2 + -7)];
                        tr_vals[(TILE_N / 2 + -6)] = score_values[(TILE_N / 2 + -6)];
                        tr_vals[(TILE_N / 2 + -5)] = score_values[(TILE_N / 2 + -5)];
                        tr_vals[(TILE_N / 2 + -4)] = score_values[(TILE_N / 2 + -4)];
                        tr_vals[(TILE_N / 2 + -3)] = score_values[(TILE_N / 2 + -3)];
                        tr_vals[(TILE_N / 2 + -2)] = score_values[(TILE_N / 2 + -2)];
                        tr_vals[(TILE_N / 2 + -1)] = score_values[(TILE_N / 2 + -1)];
                        int hi_bit = lane & 16;
                        float send = ((hi_bit != 0) ? tr_vals[0] : tr_vals[(TILE_N / 4)]);
                        float keep = ((hi_bit != 0) ? tr_vals[(TILE_N / 4)] : tr_vals[0]);
                        float _shfl_0 = __shfl_sync(0xFFFFFFFF, send, lane ^ 16);
                        float recv = _shfl_0;
                        float _max_30 = max_noftz(keep, recv);
                        tr_vals[0] = _max_30;
                        float send_0 = ((hi_bit != 0) ? tr_vals[1] : tr_vals[(TILE_N / 4 + 1)]);
                        float keep_1 = ((hi_bit != 0) ? tr_vals[(TILE_N / 4 + 1)] : tr_vals[1]);
                        float _shfl_1 = __shfl_sync(0xFFFFFFFF, send_0, lane ^ 16);
                        float recv_2 = _shfl_1;
                        float _max_31 = max_noftz(keep_1, recv_2);
                        tr_vals[1] = _max_31;
                        float send_3 = ((hi_bit != 0) ? tr_vals[2] : tr_vals[(TILE_N / 4 + 2)]);
                        float keep_4 = ((hi_bit != 0) ? tr_vals[(TILE_N / 4 + 2)] : tr_vals[2]);
                        float _shfl_2 = __shfl_sync(0xFFFFFFFF, send_3, lane ^ 16);
                        float recv_5 = _shfl_2;
                        float _max_32 = max_noftz(keep_4, recv_5);
                        tr_vals[2] = _max_32;
                        float send_6 = ((hi_bit != 0) ? tr_vals[3] : tr_vals[(TILE_N / 4 + 3)]);
                        float keep_7 = ((hi_bit != 0) ? tr_vals[(TILE_N / 4 + 3)] : tr_vals[3]);
                        float _shfl_3 = __shfl_sync(0xFFFFFFFF, send_6, lane ^ 16);
                        float recv_8 = _shfl_3;
                        float _max_33 = max_noftz(keep_7, recv_8);
                        tr_vals[3] = _max_33;
                        int hi_bit_9;
                        float _max_34;
                        if constexpr (TILE_N == 16) {
                            hi_bit_9 = lane & 8;
                            float send_10 = ((hi_bit_9 != 0) ? tr_vals[0] : tr_vals[2]);
                            float keep_11 = ((hi_bit_9 != 0) ? tr_vals[2] : tr_vals[0]);
                            float _shfl_4 = __shfl_sync(0xFFFFFFFF, send_10, lane ^ 8);
                            float recv_12 = _shfl_4;
                            _max_34 = max_noftz(keep_11, recv_12);
                        } else {
                            float send_9 = ((hi_bit != 0) ? tr_vals[4] : tr_vals[12]);
                            float keep_10 = ((hi_bit != 0) ? tr_vals[12] : tr_vals[4]);
                            float _shfl_4 = __shfl_sync(0xFFFFFFFF, send_9, lane ^ 16);
                            float recv_11 = _shfl_4;
                            _max_34 = max_noftz(keep_10, recv_11);
                        }
                        tr_vals[(TILE_N / 4 + -4)] = _max_34;
                        float _max_35;
                        if constexpr (TILE_N == 16) {
                            float send_13 = ((hi_bit_9 != 0) ? tr_vals[1] : tr_vals[3]);
                            float keep_14 = ((hi_bit_9 != 0) ? tr_vals[3] : tr_vals[1]);
                            float _shfl_5 = __shfl_sync(0xFFFFFFFF, send_13, lane ^ 8);
                            float recv_15 = _shfl_5;
                            _max_35 = max_noftz(keep_14, recv_15);
                        } else {
                            float send_12 = ((hi_bit != 0) ? tr_vals[5] : tr_vals[13]);
                            float keep_13 = ((hi_bit != 0) ? tr_vals[13] : tr_vals[5]);
                            float _shfl_5 = __shfl_sync(0xFFFFFFFF, send_12, lane ^ 16);
                            float recv_14 = _shfl_5;
                            _max_35 = max_noftz(keep_13, recv_14);
                        }
                        tr_vals[(TILE_N / 4 + -3)] = _max_35;
                        float _max_36;
                        if constexpr (TILE_N == 16) {
                            int hi_bit_16 = lane & 4;
                            float send_17 = ((hi_bit_16 != 0) ? tr_vals[0] : tr_vals[1]);
                            float keep_18 = ((hi_bit_16 != 0) ? tr_vals[1] : tr_vals[0]);
                            float _shfl_6 = __shfl_sync(0xFFFFFFFF, send_17, lane ^ 4);
                            float recv_19 = _shfl_6;
                            _max_36 = max_noftz(keep_18, recv_19);
                        } else {
                            float send_15 = ((hi_bit != 0) ? tr_vals[6] : tr_vals[14]);
                            float keep_16 = ((hi_bit != 0) ? tr_vals[14] : tr_vals[6]);
                            float _shfl_6 = __shfl_sync(0xFFFFFFFF, send_15, lane ^ 16);
                            float recv_17 = _shfl_6;
                            _max_36 = max_noftz(keep_16, recv_17);
                        }
                        tr_vals[((3 * TILE_N) / 8 + -6)] = _max_36;
                        float _max_37;
                        if constexpr (TILE_N == 16) {
                            float _shfl_7 = __shfl_sync(0xFFFFFFFF, tr_vals[0], lane ^ 2);
                            float other = _shfl_7;
                            _max_37 = max_noftz(tr_vals[0], other);
                        } else {
                            float send_18 = ((hi_bit != 0) ? tr_vals[7] : tr_vals[15]);
                            float keep_19 = ((hi_bit != 0) ? tr_vals[15] : tr_vals[7]);
                            float _shfl_7 = __shfl_sync(0xFFFFFFFF, send_18, lane ^ 16);
                            float recv_20 = _shfl_7;
                            _max_37 = max_noftz(keep_19, recv_20);
                        }
                        tr_vals[((7 * TILE_N) / 16 + -7)] = _max_37;
                        float _max_38;
                        int hi_bit_21;
                        if constexpr (TILE_N == 16) {
                            float _shfl_8 = __shfl_sync(0xFFFFFFFF, tr_vals[0], lane ^ 1);
                            float other_20 = _shfl_8;
                            _max_38 = max_noftz(tr_vals[0], other_20);
                        } else {
                            hi_bit_21 = lane & 8;
                            float send_22 = ((hi_bit_21 != 0) ? tr_vals[0] : tr_vals[4]);
                            float keep_23 = ((hi_bit_21 != 0) ? tr_vals[4] : tr_vals[0]);
                            float _shfl_8 = __shfl_sync(0xFFFFFFFF, send_22, lane ^ 8);
                            float recv_24 = _shfl_8;
                            _max_38 = max_noftz(keep_23, recv_24);
                        }
                        tr_vals[0] = _max_38;
                        if constexpr (TILE_N == 32) {
                            float send_25 = ((hi_bit_21 != 0) ? tr_vals[1] : tr_vals[5]);
                            float keep_26 = ((hi_bit_21 != 0) ? tr_vals[5] : tr_vals[1]);
                            float _shfl_9 = __shfl_sync(0xFFFFFFFF, send_25, lane ^ 8);
                            float recv_27 = _shfl_9;
                            float _max_39 = max_noftz(keep_26, recv_27);
                            tr_vals[1] = _max_39;
                            float send_28 = ((hi_bit_21 != 0) ? tr_vals[2] : tr_vals[6]);
                            float keep_29 = ((hi_bit_21 != 0) ? tr_vals[6] : tr_vals[2]);
                            float _shfl_10 = __shfl_sync(0xFFFFFFFF, send_28, lane ^ 8);
                            float recv_30 = _shfl_10;
                            float _max_40 = max_noftz(keep_29, recv_30);
                            tr_vals[2] = _max_40;
                            float send_31 = ((hi_bit_21 != 0) ? tr_vals[3] : tr_vals[7]);
                            float keep_32 = ((hi_bit_21 != 0) ? tr_vals[7] : tr_vals[3]);
                            float _shfl_11 = __shfl_sync(0xFFFFFFFF, send_31, lane ^ 8);
                            float recv_33 = _shfl_11;
                            float _max_41 = max_noftz(keep_32, recv_33);
                            tr_vals[3] = _max_41;
                            int hi_bit_34 = lane & 4;
                            float send_35 = ((hi_bit_34 != 0) ? tr_vals[0] : tr_vals[2]);
                            float keep_36 = ((hi_bit_34 != 0) ? tr_vals[2] : tr_vals[0]);
                            float _shfl_12 = __shfl_sync(0xFFFFFFFF, send_35, lane ^ 4);
                            float recv_37 = _shfl_12;
                            float _max_42 = max_noftz(keep_36, recv_37);
                            tr_vals[0] = _max_42;
                            float send_38 = ((hi_bit_34 != 0) ? tr_vals[1] : tr_vals[3]);
                            float keep_39 = ((hi_bit_34 != 0) ? tr_vals[3] : tr_vals[1]);
                            float _shfl_13 = __shfl_sync(0xFFFFFFFF, send_38, lane ^ 4);
                            float recv_40 = _shfl_13;
                            float _max_43 = max_noftz(keep_39, recv_40);
                            tr_vals[1] = _max_43;
                            int hi_bit_41 = lane & 2;
                            float send_42 = ((hi_bit_41 != 0) ? tr_vals[0] : tr_vals[1]);
                            float keep_43 = ((hi_bit_41 != 0) ? tr_vals[1] : tr_vals[0]);
                            float _shfl_14 = __shfl_sync(0xFFFFFFFF, send_42, lane ^ 2);
                            float recv_44 = _shfl_14;
                            float _max_44 = max_noftz(keep_43, recv_44);
                            tr_vals[0] = _max_44;
                            float _shfl_15 = __shfl_sync(0xFFFFFFFF, tr_vals[0], lane ^ 1);
                            float other = _shfl_15;
                            float _max_45 = max_noftz(tr_vals[0], other);
                            tr_vals[0] = _max_45;
                        }
                        if ((lane & ((-1 * TILE_N) / 8 + 5)) == 0) {
                            smem_pmax[local_warp * (TILE_N / 2) + (lane >> ((-1 * TILE_N) / 16 + 3))] = tr_vals[0];
                        }
                        asm volatile("barrier.sync 11, 128;" ::: "memory");
                        float m_lane = -CAKE_INF;
                        float alpha = 1.0f;
                        int grow = 0;
                        if (lane < (TILE_N / 2)) {
                            if constexpr (TILE_N == 16) {
                                float _max_39 = max_noftz(smem_pmax[lane], smem_pmax[8 + lane]);
                                float _max_40 = max_noftz(smem_pmax[16 + lane], smem_pmax[24 + lane]);
                                float _max_41 = max_noftz(_max_39, _max_40);
                                m_lane = _max_41;
                            } else {
                                float _max_46 = max_noftz(smem_pmax[lane], smem_pmax[16 + lane]);
                                float _max_47 = max_noftz(smem_pmax[32 + lane], smem_pmax[48 + lane]);
                                float _max_48 = max_noftz(_max_46, _max_47);
                                m_lane = _max_48;
                            }
                            float cand = m_lane * softmax_scale_log2;
                            if (it == 0) {
                                if constexpr (TILE_N == 16) {
                                    float _max_42 = max_noftz(cand, sink_lane);
                                    cand = _max_42;
                                } else {
                                    float _max_49 = max_noftz(cand, sink_lane);
                                    cand = _max_49;
                                }
                            }
                            if constexpr (TILE_N == 16) {
                                float _max_43 = max_noftz(cand, m_run);
                                cand = _max_43;
                            } else {
                                float _max_50 = max_noftz(cand, m_run);
                                cand = _max_50;
                            }
                            if (it == 0) {
                                grow = 1;
                            }
                            if (cand - m_run > 8.0f) {
                                grow = 1;
                            }
                            if (grow != 0) {
                                float _exp2_0 = approx_exp2(m_run - cand);
                                alpha = ((m_run > -CAKE_INF) ? _exp2_0 : 0.0f);
                                l_run = l_run * alpha;
                                r_run = r_run * alpha;
                                m_run = cand;
                            }
                        }
                        float m_scaled = ((m_run > -CAKE_INF) ? m_run : 0.0f);
                        unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, grow != 0);
                        unsigned int grow_bits = _vote_0;
                        if (it > 0) {
                            if (grow_bits != 0) {
                                float alpha_c[(TILE_N / 2)];
                                if constexpr (TILE_N == 16) {
                                    float _shfl_9 = __shfl_sync(0xFFFFFFFF, alpha, 0);
                                    alpha_c[0] = _shfl_9;
                                    float _shfl_10 = __shfl_sync(0xFFFFFFFF, alpha, 1);
                                    alpha_c[1] = _shfl_10;
                                    float _shfl_11 = __shfl_sync(0xFFFFFFFF, alpha, 2);
                                    alpha_c[2] = _shfl_11;
                                    float _shfl_12 = __shfl_sync(0xFFFFFFFF, alpha, 3);
                                    alpha_c[3] = _shfl_12;
                                    float _shfl_13 = __shfl_sync(0xFFFFFFFF, alpha, 4);
                                    alpha_c[4] = _shfl_13;
                                    float _shfl_14 = __shfl_sync(0xFFFFFFFF, alpha, 5);
                                    alpha_c[5] = _shfl_14;
                                    float _shfl_15 = __shfl_sync(0xFFFFFFFF, alpha, 6);
                                    alpha_c[6] = _shfl_15;
                                }
                                float _shfl_16 = __shfl_sync(0xFFFFFFFF, alpha, (7 * (32 / TILE_N) + -7));
                                alpha_c[(7 * (32 / TILE_N) + -7)] = _shfl_16;
                                if constexpr (TILE_N == 32) {
                                    float _shfl_17 = __shfl_sync(0xFFFFFFFF, alpha, 1);
                                    alpha_c[1] = _shfl_17;
                                    float _shfl_18 = __shfl_sync(0xFFFFFFFF, alpha, 2);
                                    alpha_c[2] = _shfl_18;
                                    float _shfl_19 = __shfl_sync(0xFFFFFFFF, alpha, 3);
                                    alpha_c[3] = _shfl_19;
                                    float _shfl_20 = __shfl_sync(0xFFFFFFFF, alpha, 4);
                                    alpha_c[4] = _shfl_20;
                                    float _shfl_21 = __shfl_sync(0xFFFFFFFF, alpha, 5);
                                    alpha_c[5] = _shfl_21;
                                    float _shfl_22 = __shfl_sync(0xFFFFFFFF, alpha, 6);
                                    alpha_c[6] = _shfl_22;
                                    float _shfl_23 = __shfl_sync(0xFFFFFFFF, alpha, 7);
                                    alpha_c[7] = _shfl_23;
                                    float _shfl_24 = __shfl_sync(0xFFFFFFFF, alpha, 8);
                                    alpha_c[8] = _shfl_24;
                                    float _shfl_25 = __shfl_sync(0xFFFFFFFF, alpha, 9);
                                    alpha_c[9] = _shfl_25;
                                    float _shfl_26 = __shfl_sync(0xFFFFFFFF, alpha, 10);
                                    alpha_c[10] = _shfl_26;
                                    float _shfl_27 = __shfl_sync(0xFFFFFFFF, alpha, 11);
                                    alpha_c[11] = _shfl_27;
                                    float _shfl_28 = __shfl_sync(0xFFFFFFFF, alpha, 12);
                                    alpha_c[12] = _shfl_28;
                                    float _shfl_29 = __shfl_sync(0xFFFFFFFF, alpha, 13);
                                    alpha_c[13] = _shfl_29;
                                    float _shfl_30 = __shfl_sync(0xFFFFFFFF, alpha, 14);
                                    alpha_c[14] = _shfl_30;
                                    float _shfl_31 = __shfl_sync(0xFFFFFFFF, alpha, 15);
                                    alpha_c[15] = _shfl_31;
                                }
                                float ov[(TILE_N / 2)];
                                if constexpr (TILE_N == 16) {
                                    tmem_ld_x8(&ov[0], taddr + 16 + (unsigned int)(tmem_row_origin << 16));
                                } else {
                                    tmem_ld_x16(&ov[0], taddr + 32 + (unsigned int)(tmem_row_origin << 16));
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                ov[0] = ov[0] * alpha_c[0];
                                ov[1] = ov[1] * alpha_c[1];
                                ov[2] = ov[2] * alpha_c[2];
                                ov[3] = ov[3] * alpha_c[3];
                                ov[4] = ov[4] * alpha_c[4];
                                ov[5] = ov[5] * alpha_c[5];
                                ov[6] = ov[6] * alpha_c[6];
                                ov[7] = ov[7] * alpha_c[7];
                                if constexpr (TILE_N == 16) {
                                    tmem_st_x8_f32(taddr + 16 + (unsigned int)(tmem_row_origin << 16), ov);
                                    tmem_ld_x8(&ov[0], taddr + 32 + (unsigned int)(tmem_row_origin << 16));
                                } else {
                                    ov[8] = ov[8] * alpha_c[8];
                                    ov[9] = ov[9] * alpha_c[9];
                                    ov[10] = ov[10] * alpha_c[10];
                                    ov[11] = ov[11] * alpha_c[11];
                                    ov[12] = ov[12] * alpha_c[12];
                                    ov[13] = ov[13] * alpha_c[13];
                                    ov[14] = ov[14] * alpha_c[14];
                                    ov[15] = ov[15] * alpha_c[15];
                                    tmem_st_x16_f32(taddr + 32 + (unsigned int)(tmem_row_origin << 16), ov);
                                    tmem_ld_x16(&ov[0], taddr + 64 + (unsigned int)(tmem_row_origin << 16));
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                ov[0] = ov[0] * alpha_c[0];
                                ov[1] = ov[1] * alpha_c[1];
                                ov[2] = ov[2] * alpha_c[2];
                                ov[3] = ov[3] * alpha_c[3];
                                ov[4] = ov[4] * alpha_c[4];
                                ov[5] = ov[5] * alpha_c[5];
                                ov[6] = ov[6] * alpha_c[6];
                                ov[7] = ov[7] * alpha_c[7];
                                if constexpr (TILE_N == 16) {
                                    tmem_st_x8_f32(taddr + 32 + (unsigned int)(tmem_row_origin << 16), ov);
                                    tmem_ld_x8(&ov[0], taddr + 48 + (unsigned int)(tmem_row_origin << 16));
                                } else {
                                    ov[8] = ov[8] * alpha_c[8];
                                    ov[9] = ov[9] * alpha_c[9];
                                    ov[10] = ov[10] * alpha_c[10];
                                    ov[11] = ov[11] * alpha_c[11];
                                    ov[12] = ov[12] * alpha_c[12];
                                    ov[13] = ov[13] * alpha_c[13];
                                    ov[14] = ov[14] * alpha_c[14];
                                    ov[15] = ov[15] * alpha_c[15];
                                    tmem_st_x16_f32(taddr + 64 + (unsigned int)(tmem_row_origin << 16), ov);
                                    tmem_ld_x16(&ov[0], taddr + 96 + (unsigned int)(tmem_row_origin << 16));
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                ov[0] = ov[0] * alpha_c[0];
                                ov[1] = ov[1] * alpha_c[1];
                                ov[2] = ov[2] * alpha_c[2];
                                ov[3] = ov[3] * alpha_c[3];
                                ov[4] = ov[4] * alpha_c[4];
                                ov[5] = ov[5] * alpha_c[5];
                                ov[6] = ov[6] * alpha_c[6];
                                ov[7] = ov[7] * alpha_c[7];
                                if constexpr (TILE_N == 16) {
                                    tmem_st_x8_f32(taddr + 48 + (unsigned int)(tmem_row_origin << 16), ov);
                                    tmem_ld_x8(&ov[0], taddr + 64 + (unsigned int)(tmem_row_origin << 16));
                                } else {
                                    ov[8] = ov[8] * alpha_c[8];
                                    ov[9] = ov[9] * alpha_c[9];
                                    ov[10] = ov[10] * alpha_c[10];
                                    ov[11] = ov[11] * alpha_c[11];
                                    ov[12] = ov[12] * alpha_c[12];
                                    ov[13] = ov[13] * alpha_c[13];
                                    ov[14] = ov[14] * alpha_c[14];
                                    ov[15] = ov[15] * alpha_c[15];
                                    tmem_st_x16_f32(taddr + 96 + (unsigned int)(tmem_row_origin << 16), ov);
                                    tmem_ld_x16(&ov[0], taddr + 128 + (unsigned int)(tmem_row_origin << 16));
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                ov[0] = ov[0] * alpha_c[0];
                                ov[1] = ov[1] * alpha_c[1];
                                ov[2] = ov[2] * alpha_c[2];
                                ov[3] = ov[3] * alpha_c[3];
                                ov[4] = ov[4] * alpha_c[4];
                                ov[5] = ov[5] * alpha_c[5];
                                ov[6] = ov[6] * alpha_c[6];
                                ov[7] = ov[7] * alpha_c[7];
                                if constexpr (TILE_N == 16) {
                                    tmem_st_x8_f32(taddr + 64 + (unsigned int)(tmem_row_origin << 16), ov);
                                } else {
                                    ov[8] = ov[8] * alpha_c[8];
                                    ov[9] = ov[9] * alpha_c[9];
                                    ov[10] = ov[10] * alpha_c[10];
                                    ov[11] = ov[11] * alpha_c[11];
                                    ov[12] = ov[12] * alpha_c[12];
                                    ov[13] = ov[13] * alpha_c[13];
                                    ov[14] = ov[14] * alpha_c[14];
                                    ov[15] = ov[15] * alpha_c[15];
                                    tmem_st_x16_f32(taddr + 128 + (unsigned int)(tmem_row_origin << 16), ov);
                                }
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                        }
                        float col_max[(TILE_N / 2)];
                        if constexpr (TILE_N == 16) {
                            float _shfl_17 = __shfl_sync(0xFFFFFFFF, m_scaled, 0);
                            col_max[0] = _shfl_17;
                            float _shfl_18 = __shfl_sync(0xFFFFFFFF, m_scaled, 1);
                            col_max[1] = _shfl_18;
                            float _shfl_19 = __shfl_sync(0xFFFFFFFF, m_scaled, 2);
                            col_max[2] = _shfl_19;
                            float _shfl_20 = __shfl_sync(0xFFFFFFFF, m_scaled, 3);
                            col_max[3] = _shfl_20;
                            float _shfl_21 = __shfl_sync(0xFFFFFFFF, m_scaled, 4);
                            col_max[4] = _shfl_21;
                            float _shfl_22 = __shfl_sync(0xFFFFFFFF, m_scaled, 5);
                            col_max[5] = _shfl_22;
                            float _shfl_23 = __shfl_sync(0xFFFFFFFF, m_scaled, 6);
                            col_max[6] = _shfl_23;
                            float _shfl_24 = __shfl_sync(0xFFFFFFFF, m_scaled, 7);
                            col_max[7] = _shfl_24;
                        } else {
                            float _shfl_32 = __shfl_sync(0xFFFFFFFF, m_scaled, 0);
                            col_max[0] = _shfl_32;
                            float _shfl_33 = __shfl_sync(0xFFFFFFFF, m_scaled, 1);
                            col_max[1] = _shfl_33;
                            float _shfl_34 = __shfl_sync(0xFFFFFFFF, m_scaled, 2);
                            col_max[2] = _shfl_34;
                            float _shfl_35 = __shfl_sync(0xFFFFFFFF, m_scaled, 3);
                            col_max[3] = _shfl_35;
                            float _shfl_36 = __shfl_sync(0xFFFFFFFF, m_scaled, 4);
                            col_max[4] = _shfl_36;
                            float _shfl_37 = __shfl_sync(0xFFFFFFFF, m_scaled, 5);
                            col_max[5] = _shfl_37;
                            float _shfl_38 = __shfl_sync(0xFFFFFFFF, m_scaled, 6);
                            col_max[6] = _shfl_38;
                            float _shfl_39 = __shfl_sync(0xFFFFFFFF, m_scaled, 7);
                            col_max[7] = _shfl_39;
                            float _shfl_40 = __shfl_sync(0xFFFFFFFF, m_scaled, 8);
                            col_max[8] = _shfl_40;
                            float _shfl_41 = __shfl_sync(0xFFFFFFFF, m_scaled, 9);
                            col_max[9] = _shfl_41;
                            float _shfl_42 = __shfl_sync(0xFFFFFFFF, m_scaled, 10);
                            col_max[10] = _shfl_42;
                            float _shfl_43 = __shfl_sync(0xFFFFFFFF, m_scaled, 11);
                            col_max[11] = _shfl_43;
                            float _shfl_44 = __shfl_sync(0xFFFFFFFF, m_scaled, 12);
                            col_max[12] = _shfl_44;
                            float _shfl_45 = __shfl_sync(0xFFFFFFFF, m_scaled, 13);
                            col_max[13] = _shfl_45;
                            float _shfl_46 = __shfl_sync(0xFFFFFFFF, m_scaled, 14);
                            col_max[14] = _shfl_46;
                            float _shfl_47 = __shfl_sync(0xFFFFFFFF, m_scaled, 15);
                            col_max[15] = _shfl_47;
                        }
                        float _exp2_1 = approx_exp2(score_values[0] * softmax_scale_log2 - col_max[0]);
                        score_values[0] = _exp2_1;
                        float _exp2_2 = approx_exp2(score_values[1] * softmax_scale_log2 - col_max[1]);
                        score_values[1] = _exp2_2;
                        float _exp2_3 = approx_exp2(score_values[2] * softmax_scale_log2 - col_max[2]);
                        score_values[2] = _exp2_3;
                        float _exp2_4 = approx_exp2(score_values[3] * softmax_scale_log2 - col_max[3]);
                        score_values[3] = _exp2_4;
                        float _exp2_5 = approx_exp2(score_values[4] * softmax_scale_log2 - col_max[4]);
                        score_values[4] = _exp2_5;
                        float _exp2_6 = approx_exp2(score_values[5] * softmax_scale_log2 - col_max[5]);
                        score_values[5] = _exp2_6;
                        float _exp2_7 = approx_exp2(score_values[6] * softmax_scale_log2 - col_max[6]);
                        score_values[6] = _exp2_7;
                        float _exp2_8 = approx_exp2(score_values[7] * softmax_scale_log2 - col_max[7]);
                        score_values[7] = _exp2_8;
                        if constexpr (TILE_N == 32) {
                            float _exp2_9 = approx_exp2(score_values[8] * softmax_scale_log2 - col_max[8]);
                            score_values[8] = _exp2_9;
                            float _exp2_10 = approx_exp2(score_values[9] * softmax_scale_log2 - col_max[9]);
                            score_values[9] = _exp2_10;
                            float _exp2_11 = approx_exp2(score_values[10] * softmax_scale_log2 - col_max[10]);
                            score_values[10] = _exp2_11;
                            float _exp2_12 = approx_exp2(score_values[11] * softmax_scale_log2 - col_max[11]);
                            score_values[11] = _exp2_12;
                            float _exp2_13 = approx_exp2(score_values[12] * softmax_scale_log2 - col_max[12]);
                            score_values[12] = _exp2_13;
                            float _exp2_14 = approx_exp2(score_values[13] * softmax_scale_log2 - col_max[13]);
                            score_values[13] = _exp2_14;
                            float _exp2_15 = approx_exp2(score_values[14] * softmax_scale_log2 - col_max[14]);
                            score_values[14] = _exp2_15;
                            float _exp2_16 = approx_exp2(score_values[15] * softmax_scale_log2 - col_max[15]);
                            score_values[15] = _exp2_16;
                        }
                        {
                            uint16_t _fp8_pair_46;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                : "=h"(_fp8_pair_46) : "f"(0.0f), "f"(score_values[0]));
                            uint32_t _byte_46 = (uint32_t)(_fp8_pair_46 & 0xFF);
                            uint32_t _addr_46 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(row ^ (row >> 7 & 7) << 4)));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_46), "r"(_byte_46) : "memory");
                        }
                        float _fp8_rt_0;
                        uint16_t _e4m3x2_47;
                        uint32_t _f16x2_47;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_47) : "f"(0.0f), "f"(score_values[0]));
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_47) : "h"(_e4m3x2_47));
                        uint16_t _fp8_h0_47 = (uint16_t)(_f16x2_47 & 0xFFFFu);
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_47));
                        tr_vals[0] = _fp8_rt_0;
                        {
                            uint16_t _fp8_pair_48;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                : "=h"(_fp8_pair_48) : "f"(0.0f), "f"(score_values[1]));
                            uint32_t _byte_48 = (uint32_t)(_fp8_pair_48 & 0xFF);
                            uint32_t _addr_48 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(128 + row ^ (128 + row >> 7 & 7) << 4)));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_48), "r"(_byte_48) : "memory");
                        }
                        float _fp8_rt_1;
                        uint16_t _e4m3x2_49;
                        uint32_t _f16x2_49;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_49) : "f"(0.0f), "f"(score_values[1]));
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_49) : "h"(_e4m3x2_49));
                        uint16_t _fp8_h0_49 = (uint16_t)(_f16x2_49 & 0xFFFFu);
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_1) : "h"(_fp8_h0_49));
                        tr_vals[1] = _fp8_rt_1;
                        {
                            uint16_t _fp8_pair_50;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                : "=h"(_fp8_pair_50) : "f"(0.0f), "f"(score_values[2]));
                            uint32_t _byte_50 = (uint32_t)(_fp8_pair_50 & 0xFF);
                            uint32_t _addr_50 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(256 + row ^ (256 + row >> 7 & 7) << 4)));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_50), "r"(_byte_50) : "memory");
                        }
                        float _fp8_rt_2;
                        uint16_t _e4m3x2_51;
                        uint32_t _f16x2_51;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_51) : "f"(0.0f), "f"(score_values[2]));
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_51) : "h"(_e4m3x2_51));
                        uint16_t _fp8_h0_51 = (uint16_t)(_f16x2_51 & 0xFFFFu);
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_2) : "h"(_fp8_h0_51));
                        tr_vals[2] = _fp8_rt_2;
                        {
                            uint16_t _fp8_pair_52;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                : "=h"(_fp8_pair_52) : "f"(0.0f), "f"(score_values[3]));
                            uint32_t _byte_52 = (uint32_t)(_fp8_pair_52 & 0xFF);
                            uint32_t _addr_52 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(384 + row ^ (384 + row >> 7 & 7) << 4)));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_52), "r"(_byte_52) : "memory");
                        }
                        float _fp8_rt_3;
                        uint16_t _e4m3x2_53;
                        uint32_t _f16x2_53;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_53) : "f"(0.0f), "f"(score_values[3]));
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_53) : "h"(_e4m3x2_53));
                        uint16_t _fp8_h0_53 = (uint16_t)(_f16x2_53 & 0xFFFFu);
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_3) : "h"(_fp8_h0_53));
                        tr_vals[3] = _fp8_rt_3;
                        {
                            uint16_t _fp8_pair_54;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                : "=h"(_fp8_pair_54) : "f"(0.0f), "f"(score_values[4]));
                            uint32_t _byte_54 = (uint32_t)(_fp8_pair_54 & 0xFF);
                            uint32_t _addr_54 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(512 + row ^ (512 + row >> 7 & 7) << 4)));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_54), "r"(_byte_54) : "memory");
                        }
                        float _fp8_rt_4;
                        uint16_t _e4m3x2_55;
                        uint32_t _f16x2_55;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_55) : "f"(0.0f), "f"(score_values[4]));
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_55) : "h"(_e4m3x2_55));
                        uint16_t _fp8_h0_55 = (uint16_t)(_f16x2_55 & 0xFFFFu);
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_4) : "h"(_fp8_h0_55));
                        tr_vals[4] = _fp8_rt_4;
                        {
                            uint16_t _fp8_pair_56;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                : "=h"(_fp8_pair_56) : "f"(0.0f), "f"(score_values[5]));
                            uint32_t _byte_56 = (uint32_t)(_fp8_pair_56 & 0xFF);
                            uint32_t _addr_56 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(640 + row ^ (640 + row >> 7 & 7) << 4)));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_56), "r"(_byte_56) : "memory");
                        }
                        float _fp8_rt_5;
                        uint16_t _e4m3x2_57;
                        uint32_t _f16x2_57;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_57) : "f"(0.0f), "f"(score_values[5]));
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_57) : "h"(_e4m3x2_57));
                        uint16_t _fp8_h0_57 = (uint16_t)(_f16x2_57 & 0xFFFFu);
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_5) : "h"(_fp8_h0_57));
                        tr_vals[5] = _fp8_rt_5;
                        {
                            uint16_t _fp8_pair_58;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                : "=h"(_fp8_pair_58) : "f"(0.0f), "f"(score_values[6]));
                            uint32_t _byte_58 = (uint32_t)(_fp8_pair_58 & 0xFF);
                            uint32_t _addr_58 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(768 + row ^ (768 + row >> 7 & 7) << 4)));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_58), "r"(_byte_58) : "memory");
                        }
                        float _fp8_rt_6;
                        uint16_t _e4m3x2_59;
                        uint32_t _f16x2_59;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_59) : "f"(0.0f), "f"(score_values[6]));
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_59) : "h"(_e4m3x2_59));
                        uint16_t _fp8_h0_59 = (uint16_t)(_f16x2_59 & 0xFFFFu);
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_6) : "h"(_fp8_h0_59));
                        tr_vals[6] = _fp8_rt_6;
                        {
                            uint16_t _fp8_pair_60;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                : "=h"(_fp8_pair_60) : "f"(0.0f), "f"(score_values[7]));
                            uint32_t _byte_60 = (uint32_t)(_fp8_pair_60 & 0xFF);
                            uint32_t _addr_60 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(896 + row ^ (896 + row >> 7 & 7) << 4)));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_60), "r"(_byte_60) : "memory");
                        }
                        float _fp8_rt_7;
                        uint16_t _e4m3x2_61;
                        uint32_t _f16x2_61;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_61) : "f"(0.0f), "f"(score_values[7]));
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_61) : "h"(_e4m3x2_61));
                        uint16_t _fp8_h0_61 = (uint16_t)(_f16x2_61 & 0xFFFFu);
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_7) : "h"(_fp8_h0_61));
                        tr_vals[7] = _fp8_rt_7;
                        if constexpr (TILE_N == 16) {
                            int hi_bit_21 = lane & 16;
                            float send_22 = ((hi_bit_21 != 0) ? score_values[0] : score_values[4]);
                            float keep_23 = ((hi_bit_21 != 0) ? score_values[4] : score_values[0]);
                            float _shfl_25 = __shfl_sync(0xFFFFFFFF, send_22, lane ^ 16);
                            float recv_24 = _shfl_25;
                            score_values[0] = keep_23 + recv_24;
                            float send_25 = ((hi_bit_21 != 0) ? score_values[1] : score_values[5]);
                            float keep_26 = ((hi_bit_21 != 0) ? score_values[5] : score_values[1]);
                            float _shfl_26 = __shfl_sync(0xFFFFFFFF, send_25, lane ^ 16);
                            float recv_27 = _shfl_26;
                            score_values[1] = keep_26 + recv_27;
                            float send_28 = ((hi_bit_21 != 0) ? score_values[2] : score_values[6]);
                            float keep_29 = ((hi_bit_21 != 0) ? score_values[6] : score_values[2]);
                            float _shfl_27 = __shfl_sync(0xFFFFFFFF, send_28, lane ^ 16);
                            float recv_30 = _shfl_27;
                            score_values[2] = keep_29 + recv_30;
                            float send_31 = ((hi_bit_21 != 0) ? score_values[3] : score_values[7]);
                            float keep_32 = ((hi_bit_21 != 0) ? score_values[7] : score_values[3]);
                            float _shfl_28 = __shfl_sync(0xFFFFFFFF, send_31, lane ^ 16);
                            float recv_33 = _shfl_28;
                            score_values[3] = keep_32 + recv_33;
                            int hi_bit_34 = lane & 8;
                            float send_35 = ((hi_bit_34 != 0) ? score_values[0] : score_values[2]);
                            float keep_36 = ((hi_bit_34 != 0) ? score_values[2] : score_values[0]);
                            float _shfl_29 = __shfl_sync(0xFFFFFFFF, send_35, lane ^ 8);
                            float recv_37 = _shfl_29;
                            score_values[0] = keep_36 + recv_37;
                            float send_38 = ((hi_bit_34 != 0) ? score_values[1] : score_values[3]);
                            float keep_39 = ((hi_bit_34 != 0) ? score_values[3] : score_values[1]);
                            float _shfl_30 = __shfl_sync(0xFFFFFFFF, send_38, lane ^ 8);
                            float recv_40 = _shfl_30;
                            score_values[1] = keep_39 + recv_40;
                            int hi_bit_41 = lane & 4;
                            float send_42 = ((hi_bit_41 != 0) ? score_values[0] : score_values[1]);
                            float keep_43 = ((hi_bit_41 != 0) ? score_values[1] : score_values[0]);
                            float _shfl_31 = __shfl_sync(0xFFFFFFFF, send_42, lane ^ 4);
                            float recv_44 = _shfl_31;
                            score_values[0] = keep_43 + recv_44;
                            float _shfl_32 = __shfl_sync(0xFFFFFFFF, score_values[0], lane ^ 2);
                            float other_45 = _shfl_32;
                            score_values[0] = score_values[0] + other_45;
                            float _shfl_33 = __shfl_sync(0xFFFFFFFF, score_values[0], lane ^ 1);
                            float other_46 = _shfl_33;
                            score_values[0] = score_values[0] + other_46;
                            int hi_bit_47 = lane & 16;
                            float send_48 = ((hi_bit_47 != 0) ? tr_vals[0] : tr_vals[4]);
                            float keep_49 = ((hi_bit_47 != 0) ? tr_vals[4] : tr_vals[0]);
                            float _shfl_34 = __shfl_sync(0xFFFFFFFF, send_48, lane ^ 16);
                            float recv_50 = _shfl_34;
                            tr_vals[0] = keep_49 + recv_50;
                            float send_51 = ((hi_bit_47 != 0) ? tr_vals[1] : tr_vals[5]);
                            float keep_52 = ((hi_bit_47 != 0) ? tr_vals[5] : tr_vals[1]);
                            float _shfl_35 = __shfl_sync(0xFFFFFFFF, send_51, lane ^ 16);
                            float recv_53 = _shfl_35;
                            tr_vals[1] = keep_52 + recv_53;
                            float send_54 = ((hi_bit_47 != 0) ? tr_vals[2] : tr_vals[6]);
                            float keep_55 = ((hi_bit_47 != 0) ? tr_vals[6] : tr_vals[2]);
                            float _shfl_36 = __shfl_sync(0xFFFFFFFF, send_54, lane ^ 16);
                            float recv_56 = _shfl_36;
                            tr_vals[2] = keep_55 + recv_56;
                            float send_57 = ((hi_bit_47 != 0) ? tr_vals[3] : tr_vals[7]);
                            float keep_58 = ((hi_bit_47 != 0) ? tr_vals[7] : tr_vals[3]);
                            float _shfl_37 = __shfl_sync(0xFFFFFFFF, send_57, lane ^ 16);
                            float recv_59 = _shfl_37;
                            tr_vals[3] = keep_58 + recv_59;
                            int hi_bit_60 = lane & 8;
                            float send_61 = ((hi_bit_60 != 0) ? tr_vals[0] : tr_vals[2]);
                            float keep_62 = ((hi_bit_60 != 0) ? tr_vals[2] : tr_vals[0]);
                            float _shfl_38 = __shfl_sync(0xFFFFFFFF, send_61, lane ^ 8);
                            float recv_63 = _shfl_38;
                            tr_vals[0] = keep_62 + recv_63;
                            float send_64 = ((hi_bit_60 != 0) ? tr_vals[1] : tr_vals[3]);
                            float keep_65 = ((hi_bit_60 != 0) ? tr_vals[3] : tr_vals[1]);
                            float _shfl_39 = __shfl_sync(0xFFFFFFFF, send_64, lane ^ 8);
                            float recv_66 = _shfl_39;
                            tr_vals[1] = keep_65 + recv_66;
                            int hi_bit_67 = lane & 4;
                            float send_68 = ((hi_bit_67 != 0) ? tr_vals[0] : tr_vals[1]);
                            float keep_69 = ((hi_bit_67 != 0) ? tr_vals[1] : tr_vals[0]);
                            float _shfl_40 = __shfl_sync(0xFFFFFFFF, send_68, lane ^ 4);
                            float recv_70 = _shfl_40;
                            tr_vals[0] = keep_69 + recv_70;
                            float _shfl_41 = __shfl_sync(0xFFFFFFFF, tr_vals[0], lane ^ 2);
                            float other_71 = _shfl_41;
                            tr_vals[0] = tr_vals[0] + other_71;
                            float _shfl_42 = __shfl_sync(0xFFFFFFFF, tr_vals[0], lane ^ 1);
                            float other_72 = _shfl_42;
                            tr_vals[0] = tr_vals[0] + other_72;
                        } else {
                            {
                                uint16_t _fp8_pair_62;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_62) : "f"(0.0f), "f"(score_values[8]));
                                uint32_t _byte_62 = (uint32_t)(_fp8_pair_62 & 0xFF);
                                uint32_t _addr_62 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1024 + row ^ (1024 + row >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_62), "r"(_byte_62) : "memory");
                            }
                            float _fp8_rt_8;
                            uint16_t _e4m3x2_63;
                            uint32_t _f16x2_63;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_63) : "f"(0.0f), "f"(score_values[8]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_63) : "h"(_e4m3x2_63));
                            uint16_t _fp8_h0_63 = (uint16_t)(_f16x2_63 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_8) : "h"(_fp8_h0_63));
                            tr_vals[8] = _fp8_rt_8;
                            {
                                uint16_t _fp8_pair_64;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_64) : "f"(0.0f), "f"(score_values[9]));
                                uint32_t _byte_64 = (uint32_t)(_fp8_pair_64 & 0xFF);
                                uint32_t _addr_64 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1152 + row ^ (1152 + row >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_64), "r"(_byte_64) : "memory");
                            }
                            float _fp8_rt_9;
                            uint16_t _e4m3x2_65;
                            uint32_t _f16x2_65;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_65) : "f"(0.0f), "f"(score_values[9]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_65) : "h"(_e4m3x2_65));
                            uint16_t _fp8_h0_65 = (uint16_t)(_f16x2_65 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_9) : "h"(_fp8_h0_65));
                            tr_vals[9] = _fp8_rt_9;
                            {
                                uint16_t _fp8_pair_66;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_66) : "f"(0.0f), "f"(score_values[10]));
                                uint32_t _byte_66 = (uint32_t)(_fp8_pair_66 & 0xFF);
                                uint32_t _addr_66 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1280 + row ^ (1280 + row >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_66), "r"(_byte_66) : "memory");
                            }
                            float _fp8_rt_10;
                            uint16_t _e4m3x2_67;
                            uint32_t _f16x2_67;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_67) : "f"(0.0f), "f"(score_values[10]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_67) : "h"(_e4m3x2_67));
                            uint16_t _fp8_h0_67 = (uint16_t)(_f16x2_67 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_10) : "h"(_fp8_h0_67));
                            tr_vals[10] = _fp8_rt_10;
                            {
                                uint16_t _fp8_pair_68;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_68) : "f"(0.0f), "f"(score_values[11]));
                                uint32_t _byte_68 = (uint32_t)(_fp8_pair_68 & 0xFF);
                                uint32_t _addr_68 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1408 + row ^ (1408 + row >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_68), "r"(_byte_68) : "memory");
                            }
                            float _fp8_rt_11;
                            uint16_t _e4m3x2_69;
                            uint32_t _f16x2_69;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_69) : "f"(0.0f), "f"(score_values[11]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_69) : "h"(_e4m3x2_69));
                            uint16_t _fp8_h0_69 = (uint16_t)(_f16x2_69 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_11) : "h"(_fp8_h0_69));
                            tr_vals[11] = _fp8_rt_11;
                            {
                                uint16_t _fp8_pair_70;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_70) : "f"(0.0f), "f"(score_values[12]));
                                uint32_t _byte_70 = (uint32_t)(_fp8_pair_70 & 0xFF);
                                uint32_t _addr_70 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1536 + row ^ (1536 + row >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_70), "r"(_byte_70) : "memory");
                            }
                            float _fp8_rt_12;
                            uint16_t _e4m3x2_71;
                            uint32_t _f16x2_71;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_71) : "f"(0.0f), "f"(score_values[12]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_71) : "h"(_e4m3x2_71));
                            uint16_t _fp8_h0_71 = (uint16_t)(_f16x2_71 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_12) : "h"(_fp8_h0_71));
                            tr_vals[12] = _fp8_rt_12;
                            {
                                uint16_t _fp8_pair_72;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_72) : "f"(0.0f), "f"(score_values[13]));
                                uint32_t _byte_72 = (uint32_t)(_fp8_pair_72 & 0xFF);
                                uint32_t _addr_72 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1664 + row ^ (1664 + row >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_72), "r"(_byte_72) : "memory");
                            }
                            float _fp8_rt_13;
                            uint16_t _e4m3x2_73;
                            uint32_t _f16x2_73;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_73) : "f"(0.0f), "f"(score_values[13]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_73) : "h"(_e4m3x2_73));
                            uint16_t _fp8_h0_73 = (uint16_t)(_f16x2_73 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_13) : "h"(_fp8_h0_73));
                            tr_vals[13] = _fp8_rt_13;
                            {
                                uint16_t _fp8_pair_74;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_74) : "f"(0.0f), "f"(score_values[14]));
                                uint32_t _byte_74 = (uint32_t)(_fp8_pair_74 & 0xFF);
                                uint32_t _addr_74 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1792 + row ^ (1792 + row >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_74), "r"(_byte_74) : "memory");
                            }
                            float _fp8_rt_14;
                            uint16_t _e4m3x2_75;
                            uint32_t _f16x2_75;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_75) : "f"(0.0f), "f"(score_values[14]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_75) : "h"(_e4m3x2_75));
                            uint16_t _fp8_h0_75 = (uint16_t)(_f16x2_75 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_14) : "h"(_fp8_h0_75));
                            tr_vals[14] = _fp8_rt_14;
                            {
                                uint16_t _fp8_pair_76;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_76) : "f"(0.0f), "f"(score_values[15]));
                                uint32_t _byte_76 = (uint32_t)(_fp8_pair_76 & 0xFF);
                                uint32_t _addr_76 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1920 + row ^ (1920 + row >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_76), "r"(_byte_76) : "memory");
                            }
                            float _fp8_rt_15;
                            uint16_t _e4m3x2_77;
                            uint32_t _f16x2_77;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_77) : "f"(0.0f), "f"(score_values[15]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_77) : "h"(_e4m3x2_77));
                            uint16_t _fp8_h0_77 = (uint16_t)(_f16x2_77 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_15) : "h"(_fp8_h0_77));
                            tr_vals[15] = _fp8_rt_15;
                            int hi_bit_45 = lane & 16;
                            float send_46 = ((hi_bit_45 != 0) ? score_values[0] : score_values[8]);
                            float keep_47 = ((hi_bit_45 != 0) ? score_values[8] : score_values[0]);
                            float _shfl_48 = __shfl_sync(0xFFFFFFFF, send_46, lane ^ 16);
                            float recv_48 = _shfl_48;
                            score_values[0] = keep_47 + recv_48;
                            float send_49 = ((hi_bit_45 != 0) ? score_values[1] : score_values[9]);
                            float keep_50 = ((hi_bit_45 != 0) ? score_values[9] : score_values[1]);
                            float _shfl_49 = __shfl_sync(0xFFFFFFFF, send_49, lane ^ 16);
                            float recv_51 = _shfl_49;
                            score_values[1] = keep_50 + recv_51;
                            float send_52 = ((hi_bit_45 != 0) ? score_values[2] : score_values[10]);
                            float keep_53 = ((hi_bit_45 != 0) ? score_values[10] : score_values[2]);
                            float _shfl_50 = __shfl_sync(0xFFFFFFFF, send_52, lane ^ 16);
                            float recv_54 = _shfl_50;
                            score_values[2] = keep_53 + recv_54;
                            float send_55 = ((hi_bit_45 != 0) ? score_values[3] : score_values[11]);
                            float keep_56 = ((hi_bit_45 != 0) ? score_values[11] : score_values[3]);
                            float _shfl_51 = __shfl_sync(0xFFFFFFFF, send_55, lane ^ 16);
                            float recv_57 = _shfl_51;
                            score_values[3] = keep_56 + recv_57;
                            float send_58 = ((hi_bit_45 != 0) ? score_values[4] : score_values[12]);
                            float keep_59 = ((hi_bit_45 != 0) ? score_values[12] : score_values[4]);
                            float _shfl_52 = __shfl_sync(0xFFFFFFFF, send_58, lane ^ 16);
                            float recv_60 = _shfl_52;
                            score_values[4] = keep_59 + recv_60;
                            float send_61 = ((hi_bit_45 != 0) ? score_values[5] : score_values[13]);
                            float keep_62 = ((hi_bit_45 != 0) ? score_values[13] : score_values[5]);
                            float _shfl_53 = __shfl_sync(0xFFFFFFFF, send_61, lane ^ 16);
                            float recv_63 = _shfl_53;
                            score_values[5] = keep_62 + recv_63;
                            float send_64 = ((hi_bit_45 != 0) ? score_values[6] : score_values[14]);
                            float keep_65 = ((hi_bit_45 != 0) ? score_values[14] : score_values[6]);
                            float _shfl_54 = __shfl_sync(0xFFFFFFFF, send_64, lane ^ 16);
                            float recv_66 = _shfl_54;
                            score_values[6] = keep_65 + recv_66;
                            float send_67 = ((hi_bit_45 != 0) ? score_values[7] : score_values[15]);
                            float keep_68 = ((hi_bit_45 != 0) ? score_values[15] : score_values[7]);
                            float _shfl_55 = __shfl_sync(0xFFFFFFFF, send_67, lane ^ 16);
                            float recv_69 = _shfl_55;
                            score_values[7] = keep_68 + recv_69;
                            int hi_bit_70 = lane & 8;
                            float send_71 = ((hi_bit_70 != 0) ? score_values[0] : score_values[4]);
                            float keep_72 = ((hi_bit_70 != 0) ? score_values[4] : score_values[0]);
                            float _shfl_56 = __shfl_sync(0xFFFFFFFF, send_71, lane ^ 8);
                            float recv_73 = _shfl_56;
                            score_values[0] = keep_72 + recv_73;
                            float send_74 = ((hi_bit_70 != 0) ? score_values[1] : score_values[5]);
                            float keep_75 = ((hi_bit_70 != 0) ? score_values[5] : score_values[1]);
                            float _shfl_57 = __shfl_sync(0xFFFFFFFF, send_74, lane ^ 8);
                            float recv_76 = _shfl_57;
                            score_values[1] = keep_75 + recv_76;
                            float send_77 = ((hi_bit_70 != 0) ? score_values[2] : score_values[6]);
                            float keep_78 = ((hi_bit_70 != 0) ? score_values[6] : score_values[2]);
                            float _shfl_58 = __shfl_sync(0xFFFFFFFF, send_77, lane ^ 8);
                            float recv_79 = _shfl_58;
                            score_values[2] = keep_78 + recv_79;
                            float send_80 = ((hi_bit_70 != 0) ? score_values[3] : score_values[7]);
                            float keep_81 = ((hi_bit_70 != 0) ? score_values[7] : score_values[3]);
                            float _shfl_59 = __shfl_sync(0xFFFFFFFF, send_80, lane ^ 8);
                            float recv_82 = _shfl_59;
                            score_values[3] = keep_81 + recv_82;
                            int hi_bit_83 = lane & 4;
                            float send_84 = ((hi_bit_83 != 0) ? score_values[0] : score_values[2]);
                            float keep_85 = ((hi_bit_83 != 0) ? score_values[2] : score_values[0]);
                            float _shfl_60 = __shfl_sync(0xFFFFFFFF, send_84, lane ^ 4);
                            float recv_86 = _shfl_60;
                            score_values[0] = keep_85 + recv_86;
                            float send_87 = ((hi_bit_83 != 0) ? score_values[1] : score_values[3]);
                            float keep_88 = ((hi_bit_83 != 0) ? score_values[3] : score_values[1]);
                            float _shfl_61 = __shfl_sync(0xFFFFFFFF, send_87, lane ^ 4);
                            float recv_89 = _shfl_61;
                            score_values[1] = keep_88 + recv_89;
                            int hi_bit_90 = lane & 2;
                            float send_91 = ((hi_bit_90 != 0) ? score_values[0] : score_values[1]);
                            float keep_92 = ((hi_bit_90 != 0) ? score_values[1] : score_values[0]);
                            float _shfl_62 = __shfl_sync(0xFFFFFFFF, send_91, lane ^ 2);
                            float recv_93 = _shfl_62;
                            score_values[0] = keep_92 + recv_93;
                            float _shfl_63 = __shfl_sync(0xFFFFFFFF, score_values[0], lane ^ 1);
                            float other_94 = _shfl_63;
                            score_values[0] = score_values[0] + other_94;
                            int hi_bit_95 = lane & 16;
                            float send_96 = ((hi_bit_95 != 0) ? tr_vals[0] : tr_vals[8]);
                            float keep_97 = ((hi_bit_95 != 0) ? tr_vals[8] : tr_vals[0]);
                            float _shfl_64 = __shfl_sync(0xFFFFFFFF, send_96, lane ^ 16);
                            float recv_98 = _shfl_64;
                            tr_vals[0] = keep_97 + recv_98;
                            float send_99 = ((hi_bit_95 != 0) ? tr_vals[1] : tr_vals[9]);
                            float keep_100 = ((hi_bit_95 != 0) ? tr_vals[9] : tr_vals[1]);
                            float _shfl_65 = __shfl_sync(0xFFFFFFFF, send_99, lane ^ 16);
                            float recv_101 = _shfl_65;
                            tr_vals[1] = keep_100 + recv_101;
                            float send_102 = ((hi_bit_95 != 0) ? tr_vals[2] : tr_vals[10]);
                            float keep_103 = ((hi_bit_95 != 0) ? tr_vals[10] : tr_vals[2]);
                            float _shfl_66 = __shfl_sync(0xFFFFFFFF, send_102, lane ^ 16);
                            float recv_104 = _shfl_66;
                            tr_vals[2] = keep_103 + recv_104;
                            float send_105 = ((hi_bit_95 != 0) ? tr_vals[3] : tr_vals[11]);
                            float keep_106 = ((hi_bit_95 != 0) ? tr_vals[11] : tr_vals[3]);
                            float _shfl_67 = __shfl_sync(0xFFFFFFFF, send_105, lane ^ 16);
                            float recv_107 = _shfl_67;
                            tr_vals[3] = keep_106 + recv_107;
                            float send_108 = ((hi_bit_95 != 0) ? tr_vals[4] : tr_vals[12]);
                            float keep_109 = ((hi_bit_95 != 0) ? tr_vals[12] : tr_vals[4]);
                            float _shfl_68 = __shfl_sync(0xFFFFFFFF, send_108, lane ^ 16);
                            float recv_110 = _shfl_68;
                            tr_vals[4] = keep_109 + recv_110;
                            float send_111 = ((hi_bit_95 != 0) ? tr_vals[5] : tr_vals[13]);
                            float keep_112 = ((hi_bit_95 != 0) ? tr_vals[13] : tr_vals[5]);
                            float _shfl_69 = __shfl_sync(0xFFFFFFFF, send_111, lane ^ 16);
                            float recv_113 = _shfl_69;
                            tr_vals[5] = keep_112 + recv_113;
                            float send_114 = ((hi_bit_95 != 0) ? tr_vals[6] : tr_vals[14]);
                            float keep_115 = ((hi_bit_95 != 0) ? tr_vals[14] : tr_vals[6]);
                            float _shfl_70 = __shfl_sync(0xFFFFFFFF, send_114, lane ^ 16);
                            float recv_116 = _shfl_70;
                            tr_vals[6] = keep_115 + recv_116;
                            float send_117 = ((hi_bit_95 != 0) ? tr_vals[7] : tr_vals[15]);
                            float keep_118 = ((hi_bit_95 != 0) ? tr_vals[15] : tr_vals[7]);
                            float _shfl_71 = __shfl_sync(0xFFFFFFFF, send_117, lane ^ 16);
                            float recv_119 = _shfl_71;
                            tr_vals[7] = keep_118 + recv_119;
                            int hi_bit_120 = lane & 8;
                            float send_121 = ((hi_bit_120 != 0) ? tr_vals[0] : tr_vals[4]);
                            float keep_122 = ((hi_bit_120 != 0) ? tr_vals[4] : tr_vals[0]);
                            float _shfl_72 = __shfl_sync(0xFFFFFFFF, send_121, lane ^ 8);
                            float recv_123 = _shfl_72;
                            tr_vals[0] = keep_122 + recv_123;
                            float send_124 = ((hi_bit_120 != 0) ? tr_vals[1] : tr_vals[5]);
                            float keep_125 = ((hi_bit_120 != 0) ? tr_vals[5] : tr_vals[1]);
                            float _shfl_73 = __shfl_sync(0xFFFFFFFF, send_124, lane ^ 8);
                            float recv_126 = _shfl_73;
                            tr_vals[1] = keep_125 + recv_126;
                            float send_127 = ((hi_bit_120 != 0) ? tr_vals[2] : tr_vals[6]);
                            float keep_128 = ((hi_bit_120 != 0) ? tr_vals[6] : tr_vals[2]);
                            float _shfl_74 = __shfl_sync(0xFFFFFFFF, send_127, lane ^ 8);
                            float recv_129 = _shfl_74;
                            tr_vals[2] = keep_128 + recv_129;
                            float send_130 = ((hi_bit_120 != 0) ? tr_vals[3] : tr_vals[7]);
                            float keep_131 = ((hi_bit_120 != 0) ? tr_vals[7] : tr_vals[3]);
                            float _shfl_75 = __shfl_sync(0xFFFFFFFF, send_130, lane ^ 8);
                            float recv_132 = _shfl_75;
                            tr_vals[3] = keep_131 + recv_132;
                            int hi_bit_133 = lane & 4;
                            float send_134 = ((hi_bit_133 != 0) ? tr_vals[0] : tr_vals[2]);
                            float keep_135 = ((hi_bit_133 != 0) ? tr_vals[2] : tr_vals[0]);
                            float _shfl_76 = __shfl_sync(0xFFFFFFFF, send_134, lane ^ 4);
                            float recv_136 = _shfl_76;
                            tr_vals[0] = keep_135 + recv_136;
                            float send_137 = ((hi_bit_133 != 0) ? tr_vals[1] : tr_vals[3]);
                            float keep_138 = ((hi_bit_133 != 0) ? tr_vals[3] : tr_vals[1]);
                            float _shfl_77 = __shfl_sync(0xFFFFFFFF, send_137, lane ^ 4);
                            float recv_139 = _shfl_77;
                            tr_vals[1] = keep_138 + recv_139;
                            int hi_bit_140 = lane & 2;
                            float send_141 = ((hi_bit_140 != 0) ? tr_vals[0] : tr_vals[1]);
                            float keep_142 = ((hi_bit_140 != 0) ? tr_vals[1] : tr_vals[0]);
                            float _shfl_78 = __shfl_sync(0xFFFFFFFF, send_141, lane ^ 2);
                            float recv_143 = _shfl_78;
                            tr_vals[0] = keep_142 + recv_143;
                            float _shfl_79 = __shfl_sync(0xFFFFFFFF, tr_vals[0], lane ^ 1);
                            float other_144 = _shfl_79;
                            tr_vals[0] = tr_vals[0] + other_144;
                        }
                        if ((lane & ((-1 * TILE_N) / 8 + 5)) == 0) {
                            smem_psum[local_warp * (TILE_N / 2) + (lane >> ((-1 * TILE_N) / 16 + 3))] = score_values[0];
                            smem_rsum[local_warp * (TILE_N / 2) + (lane >> ((-1 * TILE_N) / 16 + 3))] = tr_vals[0];
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(p_full_addr);
                        asm volatile("barrier.sync 11, 128;" ::: "memory");
                        if (lane < (TILE_N / 2)) {
                            float tile_sum = smem_psum[lane] + smem_psum[(TILE_N / 2) + lane] + smem_psum[TILE_N + lane] + smem_psum[((3 * TILE_N) / 2) + lane];
                            float tile_rsum = smem_rsum[lane] + smem_rsum[(TILE_N / 2) + lane] + smem_rsum[TILE_N + lane] + smem_rsum[((3 * TILE_N) / 2) + lane];
                            if (it == 0) {
                                float sink_term;
                                if constexpr (TILE_N == 16) {
                                    float _exp2_9 = approx_exp2(sink_lane - m_scaled);
                                    sink_term = _exp2_9;
                                } else {
                                    float _exp2_17 = approx_exp2(sink_lane - m_scaled);
                                    sink_term = _exp2_17;
                                }
                                l_run = sink_term;
                                r_run = sink_term;
                            }
                            l_run = l_run + tile_sum;
                            r_run = r_run + tile_rsum;
                        }
                        asm volatile("barrier.sync 11, 128;" ::: "memory");
                    }
                }
                int it_0 = 2 * it2 + 1;
                if (it_0 < tiles_per_split) {
                    mbarrier_wait_hint(tok_full_addr + 8, it2 & 1, 10000000);
                    int raw_index_1 = smem_tok_1[row];
                    int valid_1 = 1;
                    if (raw_index_1 < 0) {
                        valid_1 = 0;
                    }
                    if (valid_1 != 0) {
                        {
                            unsigned int sfw_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&sfw_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&sfw_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sfw_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sfw_1[(0) + 3]))
                                : "r"(strip_1));
                            smem_ksf32[row % 32 / 8 * 512 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = sfw_1[0];
                            smem_ksf32[row % 32 / 8 * 512 + 128 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = sfw_1[1];
                            smem_ksf32[row % 32 / 8 * 512 + 256 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = sfw_1[2];
                            smem_ksf32[row % 32 / 8 * 512 + 384 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = sfw_1[3];
                        }
                    } else if (1) {
                        smem_ksf32[row % 32 / 8 * 512 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = 0;
                        smem_ksf32[row % 32 / 8 * 512 + 128 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = 0;
                        smem_ksf32[row % 32 / 8 * 512 + 256 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = 0;
                        smem_ksf32[row % 32 / 8 * 512 + 384 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = 0;
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(kv_full_addr);
                    if (it_0 > 0) {
                        mbarrier_wait_hint(o_full_addr, 0, 10000000);
                        mbarrier_wait_hint(o_full_addr + 8, 0, 10000000);
                        mbarrier_wait_hint(o_full_addr + 16, 0, 10000000);
                        mbarrier_wait_hint(o_full_addr + 24, 0, 10000000);
                    }
                    if (valid_1 != 0) {
                        unsigned int kraw4_1[4];
                        unsigned int sfw32_1 = 0;
                        int vblock_1 = 32 * o_chunk;
                        unsigned int v8_2[4];
                        {
                            {
                                int vchunk_6 = 16 * o_chunk;
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 3]))
                                    : "r"(smem_kf4_1_addr + (unsigned int)(vchunk_6 / 8 * 16384) + (unsigned int)(row * 128 + (vchunk_6 % 8 * 16 ^ row % 8 * 16))));
                            }
                            {
                                sfw32_1 = smem_sfs32_1[row * 8 + 8 * o_chunk];
                            }
                            unsigned int scale_11 = sfw32_1 & 255;
                            {
                                v8_2[0] = cake_dsv4_qmul4_portable<5>(kraw4_1[0], scale_11);
                            }
                            {
                                v8_2[1] = cake_dsv4_qmul4_portable<6>(kraw4_1[0], scale_11);
                            }
                            {
                                v8_2[2] = cake_dsv4_qmul4_portable<5>(kraw4_1[1], scale_11);
                            }
                            {
                                v8_2[3] = cake_dsv4_qmul4_portable<6>(kraw4_1[1], scale_11);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(row * 128 + (0 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_2[(0) + 3])));
                        int vblock_0_1 = 32 * o_chunk + 1;
                        unsigned int v8_1_1[4];
                        {
                            unsigned int scale_12 = sfw32_1 >> 8 & 255;
                            {
                                v8_1_1[0] = cake_dsv4_qmul4_portable<5>(kraw4_1[2], scale_12);
                            }
                            {
                                v8_1_1[1] = cake_dsv4_qmul4_portable<6>(kraw4_1[2], scale_12);
                            }
                            {
                                v8_1_1[2] = cake_dsv4_qmul4_portable<5>(kraw4_1[3], scale_12);
                            }
                            {
                                v8_1_1[3] = cake_dsv4_qmul4_portable<6>(kraw4_1[3], scale_12);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(row * 128 + (16 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_1_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_1_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_1_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_1_1[(0) + 3])));
                        int vblock_2_1 = 32 * o_chunk + 2;
                        unsigned int v8_3_1[4];
                        {
                            {
                                int vchunk_7 = 16 * o_chunk + 1;
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 3]))
                                    : "r"(smem_kf4_1_addr + (unsigned int)(vchunk_7 / 8 * 16384) + (unsigned int)(row * 128 + (vchunk_7 % 8 * 16 ^ row % 8 * 16))));
                            }
                            unsigned int scale_13 = sfw32_1 >> 16 & 255;
                            {
                                v8_3_1[0] = cake_dsv4_qmul4_portable<5>(kraw4_1[0], scale_13);
                            }
                            {
                                v8_3_1[1] = cake_dsv4_qmul4_portable<6>(kraw4_1[0], scale_13);
                            }
                            {
                                v8_3_1[2] = cake_dsv4_qmul4_portable<5>(kraw4_1[1], scale_13);
                            }
                            {
                                v8_3_1[3] = cake_dsv4_qmul4_portable<6>(kraw4_1[1], scale_13);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(row * 128 + (32 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_3_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_3_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_3_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_3_1[(0) + 3])));
                        int vblock_4_1 = 32 * o_chunk + 3;
                        unsigned int v8_5_1[4];
                        {
                            unsigned int scale_14 = sfw32_1 >> 24 & 255;
                            {
                                v8_5_1[0] = cake_dsv4_qmul4_portable<5>(kraw4_1[2], scale_14);
                            }
                            {
                                v8_5_1[1] = cake_dsv4_qmul4_portable<6>(kraw4_1[2], scale_14);
                            }
                            {
                                v8_5_1[2] = cake_dsv4_qmul4_portable<5>(kraw4_1[3], scale_14);
                            }
                            {
                                v8_5_1[3] = cake_dsv4_qmul4_portable<6>(kraw4_1[3], scale_14);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(row * 128 + (48 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_5_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_5_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_5_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_5_1[(0) + 3])));
                        int vblock_6_1 = 32 * o_chunk + 4;
                        unsigned int v8_7_1[4];
                        {
                            {
                                int vchunk_8 = 16 * o_chunk + 2;
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 3]))
                                    : "r"(smem_kf4_1_addr + (unsigned int)(vchunk_8 / 8 * 16384) + (unsigned int)(row * 128 + (vchunk_8 % 8 * 16 ^ row % 8 * 16))));
                            }
                            {
                                sfw32_1 = smem_sfs32_1[row * 8 + 8 * o_chunk + 1];
                            }
                            unsigned int scale_15 = sfw32_1 & 255;
                            {
                                v8_7_1[0] = cake_dsv4_qmul4_portable<5>(kraw4_1[0], scale_15);
                            }
                            {
                                v8_7_1[1] = cake_dsv4_qmul4_portable<6>(kraw4_1[0], scale_15);
                            }
                            {
                                v8_7_1[2] = cake_dsv4_qmul4_portable<5>(kraw4_1[1], scale_15);
                            }
                            {
                                v8_7_1[3] = cake_dsv4_qmul4_portable<6>(kraw4_1[1], scale_15);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(row * 128 + (64 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_7_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_7_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_7_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_7_1[(0) + 3])));
                        int vblock_8_1 = 32 * o_chunk + 5;
                        unsigned int v8_9_1[4];
                        {
                            unsigned int scale_16 = sfw32_1 >> 8 & 255;
                            {
                                v8_9_1[0] = cake_dsv4_qmul4_portable<5>(kraw4_1[2], scale_16);
                            }
                            {
                                v8_9_1[1] = cake_dsv4_qmul4_portable<6>(kraw4_1[2], scale_16);
                            }
                            {
                                v8_9_1[2] = cake_dsv4_qmul4_portable<5>(kraw4_1[3], scale_16);
                            }
                            {
                                v8_9_1[3] = cake_dsv4_qmul4_portable<6>(kraw4_1[3], scale_16);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(row * 128 + (80 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_9_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_9_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_9_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_9_1[(0) + 3])));
                        int vblock_10_1 = 32 * o_chunk + 6;
                        unsigned int v8_11_1[4];
                        {
                            {
                                int vchunk_9 = 16 * o_chunk + 3;
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 3]))
                                    : "r"(smem_kf4_1_addr + (unsigned int)(vchunk_9 / 8 * 16384) + (unsigned int)(row * 128 + (vchunk_9 % 8 * 16 ^ row % 8 * 16))));
                            }
                            unsigned int scale_17 = sfw32_1 >> 16 & 255;
                            {
                                v8_11_1[0] = cake_dsv4_qmul4_portable<5>(kraw4_1[0], scale_17);
                            }
                            {
                                v8_11_1[1] = cake_dsv4_qmul4_portable<6>(kraw4_1[0], scale_17);
                            }
                            {
                                v8_11_1[2] = cake_dsv4_qmul4_portable<5>(kraw4_1[1], scale_17);
                            }
                            {
                                v8_11_1[3] = cake_dsv4_qmul4_portable<6>(kraw4_1[1], scale_17);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(row * 128 + (96 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_11_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_11_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_11_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_11_1[(0) + 3])));
                        int vblock_12_1 = 32 * o_chunk + 7;
                        unsigned int v8_13_1[4];
                        {
                            unsigned int scale_18 = sfw32_1 >> 24 & 255;
                            {
                                v8_13_1[0] = cake_dsv4_qmul4_portable<5>(kraw4_1[2], scale_18);
                            }
                            {
                                v8_13_1[1] = cake_dsv4_qmul4_portable<6>(kraw4_1[2], scale_18);
                            }
                            {
                                v8_13_1[2] = cake_dsv4_qmul4_portable<5>(kraw4_1[3], scale_18);
                            }
                            {
                                v8_13_1[3] = cake_dsv4_qmul4_portable<6>(kraw4_1[3], scale_18);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(row * 128 + (112 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_13_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_13_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_13_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_13_1[(0) + 3])));
                        int vblock_14_1 = 32 * o_chunk + 8;
                        unsigned int v8_15_1[4];
                        {
                            {
                                int vchunk_10 = 16 * o_chunk + 4;
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 3]))
                                    : "r"(smem_kf4_1_addr + (unsigned int)(vchunk_10 / 8 * 16384) + (unsigned int)(row * 128 + (vchunk_10 % 8 * 16 ^ row % 8 * 16))));
                            }
                            {
                                sfw32_1 = smem_sfs32_1[row * 8 + 8 * o_chunk + 2];
                            }
                            unsigned int scale_19 = sfw32_1 & 255;
                            {
                                v8_15_1[0] = cake_dsv4_qmul4_portable<5>(kraw4_1[0], scale_19);
                            }
                            {
                                v8_15_1[1] = cake_dsv4_qmul4_portable<6>(kraw4_1[0], scale_19);
                            }
                            {
                                v8_15_1[2] = cake_dsv4_qmul4_portable<5>(kraw4_1[1], scale_19);
                            }
                            {
                                v8_15_1[3] = cake_dsv4_qmul4_portable<6>(kraw4_1[1], scale_19);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(16384 + (row * 128 + (0 ^ row % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_15_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_15_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_15_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_15_1[(0) + 3])));
                        int vblock_16_1 = 32 * o_chunk + 9;
                        unsigned int v8_17_1[4];
                        {
                            unsigned int scale_20 = sfw32_1 >> 8 & 255;
                            {
                                v8_17_1[0] = cake_dsv4_qmul4_portable<5>(kraw4_1[2], scale_20);
                            }
                            {
                                v8_17_1[1] = cake_dsv4_qmul4_portable<6>(kraw4_1[2], scale_20);
                            }
                            {
                                v8_17_1[2] = cake_dsv4_qmul4_portable<5>(kraw4_1[3], scale_20);
                            }
                            {
                                v8_17_1[3] = cake_dsv4_qmul4_portable<6>(kraw4_1[3], scale_20);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(16384 + (row * 128 + (16 ^ row % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_17_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_17_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_17_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_17_1[(0) + 3])));
                        int vblock_18_1 = 32 * o_chunk + 10;
                        unsigned int v8_19_1[4];
                        {
                            {
                                int vchunk_11 = 16 * o_chunk + 5;
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_1[(0) + 3]))
                                    : "r"(smem_kf4_1_addr + (unsigned int)(vchunk_11 / 8 * 16384) + (unsigned int)(row * 128 + (vchunk_11 % 8 * 16 ^ row % 8 * 16))));
                            }
                            unsigned int scale_21 = sfw32_1 >> 16 & 255;
                            {
                                v8_19_1[0] = cake_dsv4_qmul4_portable<5>(kraw4_1[0], scale_21);
                            }
                            {
                                v8_19_1[1] = cake_dsv4_qmul4_portable<6>(kraw4_1[0], scale_21);
                            }
                            {
                                v8_19_1[2] = cake_dsv4_qmul4_portable<5>(kraw4_1[1], scale_21);
                            }
                            {
                                v8_19_1[3] = cake_dsv4_qmul4_portable<6>(kraw4_1[1], scale_21);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(16384 + (row * 128 + (32 ^ row % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_19_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_19_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_19_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_19_1[(0) + 3])));
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
                    mbarrier_wait_hint(s_full_addr, 1, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (it_0 + 2 < tiles_per_split) {
                        mbarrier_arrive(tok_free_addr + 8);
                    }
                    {
                        float score_values_1[(TILE_N / 2)];
                        if constexpr (TILE_N == 16) {
                            tmem_ld_x8(&score_values_1[0], taddr + (unsigned int)(tmem_row_origin << 16));
                        } else {
                            tmem_ld_x16(&score_values_1[0], taddr + (unsigned int)(tmem_row_origin << 16));
                        }
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        if (valid_1 == 0) {
                            score_values_1[0] = -CAKE_INF;
                            score_values_1[1] = -CAKE_INF;
                            score_values_1[2] = -CAKE_INF;
                            score_values_1[3] = -CAKE_INF;
                            score_values_1[4] = -CAKE_INF;
                            score_values_1[5] = -CAKE_INF;
                            score_values_1[6] = -CAKE_INF;
                            score_values_1[7] = -CAKE_INF;
                            if constexpr (TILE_N == 32) {
                                score_values_1[8] = -CAKE_INF;
                                score_values_1[9] = -CAKE_INF;
                                score_values_1[10] = -CAKE_INF;
                                score_values_1[11] = -CAKE_INF;
                                score_values_1[12] = -CAKE_INF;
                                score_values_1[13] = -CAKE_INF;
                                score_values_1[14] = -CAKE_INF;
                                score_values_1[15] = -CAKE_INF;
                            }
                        }
                        float tr_vals_1[(TILE_N / 2)];
                        if constexpr (TILE_N == 32) {
                            tr_vals_1[0] = score_values_1[0];
                            tr_vals_1[1] = score_values_1[1];
                            tr_vals_1[2] = score_values_1[2];
                            tr_vals_1[3] = score_values_1[3];
                            tr_vals_1[4] = score_values_1[4];
                            tr_vals_1[5] = score_values_1[5];
                            tr_vals_1[6] = score_values_1[6];
                            tr_vals_1[7] = score_values_1[7];
                        }
                        tr_vals_1[(TILE_N / 2 + -8)] = score_values_1[(TILE_N / 2 + -8)];
                        tr_vals_1[(TILE_N / 2 + -7)] = score_values_1[(TILE_N / 2 + -7)];
                        tr_vals_1[(TILE_N / 2 + -6)] = score_values_1[(TILE_N / 2 + -6)];
                        tr_vals_1[(TILE_N / 2 + -5)] = score_values_1[(TILE_N / 2 + -5)];
                        tr_vals_1[(TILE_N / 2 + -4)] = score_values_1[(TILE_N / 2 + -4)];
                        tr_vals_1[(TILE_N / 2 + -3)] = score_values_1[(TILE_N / 2 + -3)];
                        tr_vals_1[(TILE_N / 2 + -2)] = score_values_1[(TILE_N / 2 + -2)];
                        tr_vals_1[(TILE_N / 2 + -1)] = score_values_1[(TILE_N / 2 + -1)];
                        int hi_bit_1 = lane & 16;
                        float send_1 = ((hi_bit_1 != 0) ? tr_vals_1[0] : tr_vals_1[(TILE_N / 4)]);
                        float keep_2 = ((hi_bit_1 != 0) ? tr_vals_1[(TILE_N / 4)] : tr_vals_1[0]);
                        if constexpr (TILE_N == 16) {
                            float _shfl_43 = __shfl_sync(0xFFFFFFFF, send_1, lane ^ 16);
                            float recv_1 = _shfl_43;
                            float _max_44 = max_noftz(keep_2, recv_1);
                            tr_vals_1[0] = _max_44;
                        } else {
                            float _shfl_80 = __shfl_sync(0xFFFFFFFF, send_1, lane ^ 16);
                            float recv_1 = _shfl_80;
                            float _max_51 = max_noftz(keep_2, recv_1);
                            tr_vals_1[0] = _max_51;
                        }
                        float send_0_1 = ((hi_bit_1 != 0) ? tr_vals_1[1] : tr_vals_1[(TILE_N / 4 + 1)]);
                        float keep_1_1 = ((hi_bit_1 != 0) ? tr_vals_1[(TILE_N / 4 + 1)] : tr_vals_1[1]);
                        if constexpr (TILE_N == 16) {
                            float _shfl_44 = __shfl_sync(0xFFFFFFFF, send_0_1, lane ^ 16);
                            float recv_2_1 = _shfl_44;
                            float _max_45 = max_noftz(keep_1_1, recv_2_1);
                            tr_vals_1[1] = _max_45;
                        } else {
                            float _shfl_81 = __shfl_sync(0xFFFFFFFF, send_0_1, lane ^ 16);
                            float recv_2_1 = _shfl_81;
                            float _max_52 = max_noftz(keep_1_1, recv_2_1);
                            tr_vals_1[1] = _max_52;
                        }
                        float send_3_1 = ((hi_bit_1 != 0) ? tr_vals_1[2] : tr_vals_1[(TILE_N / 4 + 2)]);
                        float keep_4_1 = ((hi_bit_1 != 0) ? tr_vals_1[(TILE_N / 4 + 2)] : tr_vals_1[2]);
                        if constexpr (TILE_N == 16) {
                            float _shfl_45 = __shfl_sync(0xFFFFFFFF, send_3_1, lane ^ 16);
                            float recv_5_1 = _shfl_45;
                            float _max_46 = max_noftz(keep_4_1, recv_5_1);
                            tr_vals_1[2] = _max_46;
                        } else {
                            float _shfl_82 = __shfl_sync(0xFFFFFFFF, send_3_1, lane ^ 16);
                            float recv_5_1 = _shfl_82;
                            float _max_53 = max_noftz(keep_4_1, recv_5_1);
                            tr_vals_1[2] = _max_53;
                        }
                        float send_6_1 = ((hi_bit_1 != 0) ? tr_vals_1[3] : tr_vals_1[(TILE_N / 4 + 3)]);
                        float keep_7_1 = ((hi_bit_1 != 0) ? tr_vals_1[(TILE_N / 4 + 3)] : tr_vals_1[3]);
                        if constexpr (TILE_N == 16) {
                            float _shfl_46 = __shfl_sync(0xFFFFFFFF, send_6_1, lane ^ 16);
                            float recv_8_1 = _shfl_46;
                            float _max_47 = max_noftz(keep_7_1, recv_8_1);
                            tr_vals_1[3] = _max_47;
                            int hi_bit_9_1 = lane & 8;
                            float send_10_1 = ((hi_bit_9_1 != 0) ? tr_vals_1[0] : tr_vals_1[2]);
                            float keep_11_1 = ((hi_bit_9_1 != 0) ? tr_vals_1[2] : tr_vals_1[0]);
                            float _shfl_47 = __shfl_sync(0xFFFFFFFF, send_10_1, lane ^ 8);
                            float recv_12_1 = _shfl_47;
                            float _max_48 = max_noftz(keep_11_1, recv_12_1);
                            tr_vals_1[0] = _max_48;
                            float send_13_1 = ((hi_bit_9_1 != 0) ? tr_vals_1[1] : tr_vals_1[3]);
                            float keep_14_1 = ((hi_bit_9_1 != 0) ? tr_vals_1[3] : tr_vals_1[1]);
                            float _shfl_48 = __shfl_sync(0xFFFFFFFF, send_13_1, lane ^ 8);
                            float recv_15_1 = _shfl_48;
                            float _max_49 = max_noftz(keep_14_1, recv_15_1);
                            tr_vals_1[1] = _max_49;
                            int hi_bit_16_1 = lane & 4;
                            float send_17_1 = ((hi_bit_16_1 != 0) ? tr_vals_1[0] : tr_vals_1[1]);
                            float keep_18_1 = ((hi_bit_16_1 != 0) ? tr_vals_1[1] : tr_vals_1[0]);
                            float _shfl_49 = __shfl_sync(0xFFFFFFFF, send_17_1, lane ^ 4);
                            float recv_19_1 = _shfl_49;
                            float _max_50 = max_noftz(keep_18_1, recv_19_1);
                            tr_vals_1[0] = _max_50;
                            float _shfl_50 = __shfl_sync(0xFFFFFFFF, tr_vals_1[0], lane ^ 2);
                            float other_1 = _shfl_50;
                            float _max_51 = max_noftz(tr_vals_1[0], other_1);
                            tr_vals_1[0] = _max_51;
                            float _shfl_51 = __shfl_sync(0xFFFFFFFF, tr_vals_1[0], lane ^ 1);
                            float other_20_1 = _shfl_51;
                            float _max_52 = max_noftz(tr_vals_1[0], other_20_1);
                            tr_vals_1[0] = _max_52;
                        } else {
                            float _shfl_83 = __shfl_sync(0xFFFFFFFF, send_6_1, lane ^ 16);
                            float recv_8_1 = _shfl_83;
                            float _max_54 = max_noftz(keep_7_1, recv_8_1);
                            tr_vals_1[3] = _max_54;
                            float send_9_1 = ((hi_bit_1 != 0) ? tr_vals_1[4] : tr_vals_1[12]);
                            float keep_10_1 = ((hi_bit_1 != 0) ? tr_vals_1[12] : tr_vals_1[4]);
                            float _shfl_84 = __shfl_sync(0xFFFFFFFF, send_9_1, lane ^ 16);
                            float recv_11_1 = _shfl_84;
                            float _max_55 = max_noftz(keep_10_1, recv_11_1);
                            tr_vals_1[4] = _max_55;
                            float send_12_1 = ((hi_bit_1 != 0) ? tr_vals_1[5] : tr_vals_1[13]);
                            float keep_13_1 = ((hi_bit_1 != 0) ? tr_vals_1[13] : tr_vals_1[5]);
                            float _shfl_85 = __shfl_sync(0xFFFFFFFF, send_12_1, lane ^ 16);
                            float recv_14_1 = _shfl_85;
                            float _max_56 = max_noftz(keep_13_1, recv_14_1);
                            tr_vals_1[5] = _max_56;
                            float send_15_1 = ((hi_bit_1 != 0) ? tr_vals_1[6] : tr_vals_1[14]);
                            float keep_16_1 = ((hi_bit_1 != 0) ? tr_vals_1[14] : tr_vals_1[6]);
                            float _shfl_86 = __shfl_sync(0xFFFFFFFF, send_15_1, lane ^ 16);
                            float recv_17_1 = _shfl_86;
                            float _max_57 = max_noftz(keep_16_1, recv_17_1);
                            tr_vals_1[6] = _max_57;
                            float send_18_1 = ((hi_bit_1 != 0) ? tr_vals_1[7] : tr_vals_1[15]);
                            float keep_19_1 = ((hi_bit_1 != 0) ? tr_vals_1[15] : tr_vals_1[7]);
                            float _shfl_87 = __shfl_sync(0xFFFFFFFF, send_18_1, lane ^ 16);
                            float recv_20_1 = _shfl_87;
                            float _max_58 = max_noftz(keep_19_1, recv_20_1);
                            tr_vals_1[7] = _max_58;
                            int hi_bit_21_1 = lane & 8;
                            float send_22_1 = ((hi_bit_21_1 != 0) ? tr_vals_1[0] : tr_vals_1[4]);
                            float keep_23_1 = ((hi_bit_21_1 != 0) ? tr_vals_1[4] : tr_vals_1[0]);
                            float _shfl_88 = __shfl_sync(0xFFFFFFFF, send_22_1, lane ^ 8);
                            float recv_24_1 = _shfl_88;
                            float _max_59 = max_noftz(keep_23_1, recv_24_1);
                            tr_vals_1[0] = _max_59;
                            float send_25_1 = ((hi_bit_21_1 != 0) ? tr_vals_1[1] : tr_vals_1[5]);
                            float keep_26_1 = ((hi_bit_21_1 != 0) ? tr_vals_1[5] : tr_vals_1[1]);
                            float _shfl_89 = __shfl_sync(0xFFFFFFFF, send_25_1, lane ^ 8);
                            float recv_27_1 = _shfl_89;
                            float _max_60 = max_noftz(keep_26_1, recv_27_1);
                            tr_vals_1[1] = _max_60;
                            float send_28_1 = ((hi_bit_21_1 != 0) ? tr_vals_1[2] : tr_vals_1[6]);
                            float keep_29_1 = ((hi_bit_21_1 != 0) ? tr_vals_1[6] : tr_vals_1[2]);
                            float _shfl_90 = __shfl_sync(0xFFFFFFFF, send_28_1, lane ^ 8);
                            float recv_30_1 = _shfl_90;
                            float _max_61 = max_noftz(keep_29_1, recv_30_1);
                            tr_vals_1[2] = _max_61;
                            float send_31_1 = ((hi_bit_21_1 != 0) ? tr_vals_1[3] : tr_vals_1[7]);
                            float keep_32_1 = ((hi_bit_21_1 != 0) ? tr_vals_1[7] : tr_vals_1[3]);
                            float _shfl_91 = __shfl_sync(0xFFFFFFFF, send_31_1, lane ^ 8);
                            float recv_33_1 = _shfl_91;
                            float _max_62 = max_noftz(keep_32_1, recv_33_1);
                            tr_vals_1[3] = _max_62;
                            int hi_bit_34_1 = lane & 4;
                            float send_35_1 = ((hi_bit_34_1 != 0) ? tr_vals_1[0] : tr_vals_1[2]);
                            float keep_36_1 = ((hi_bit_34_1 != 0) ? tr_vals_1[2] : tr_vals_1[0]);
                            float _shfl_92 = __shfl_sync(0xFFFFFFFF, send_35_1, lane ^ 4);
                            float recv_37_1 = _shfl_92;
                            float _max_63 = max_noftz(keep_36_1, recv_37_1);
                            tr_vals_1[0] = _max_63;
                            float send_38_1 = ((hi_bit_34_1 != 0) ? tr_vals_1[1] : tr_vals_1[3]);
                            float keep_39_1 = ((hi_bit_34_1 != 0) ? tr_vals_1[3] : tr_vals_1[1]);
                            float _shfl_93 = __shfl_sync(0xFFFFFFFF, send_38_1, lane ^ 4);
                            float recv_40_1 = _shfl_93;
                            float _max_64 = max_noftz(keep_39_1, recv_40_1);
                            tr_vals_1[1] = _max_64;
                            int hi_bit_41_1 = lane & 2;
                            float send_42_1 = ((hi_bit_41_1 != 0) ? tr_vals_1[0] : tr_vals_1[1]);
                            float keep_43_1 = ((hi_bit_41_1 != 0) ? tr_vals_1[1] : tr_vals_1[0]);
                            float _shfl_94 = __shfl_sync(0xFFFFFFFF, send_42_1, lane ^ 2);
                            float recv_44_1 = _shfl_94;
                            float _max_65 = max_noftz(keep_43_1, recv_44_1);
                            tr_vals_1[0] = _max_65;
                            float _shfl_95 = __shfl_sync(0xFFFFFFFF, tr_vals_1[0], lane ^ 1);
                            float other_1 = _shfl_95;
                            float _max_66 = max_noftz(tr_vals_1[0], other_1);
                            tr_vals_1[0] = _max_66;
                        }
                        if ((lane & ((-1 * TILE_N) / 8 + 5)) == 0) {
                            smem_pmax[local_warp * (TILE_N / 2) + (lane >> ((-1 * TILE_N) / 16 + 3))] = tr_vals_1[0];
                        }
                        asm volatile("barrier.sync 11, 128;" ::: "memory");
                        float m_lane_1 = -CAKE_INF;
                        float alpha_1 = 1.0f;
                        int grow_1 = 0;
                        if (lane < (TILE_N / 2)) {
                            if constexpr (TILE_N == 16) {
                                float _max_53 = max_noftz(smem_pmax[lane], smem_pmax[8 + lane]);
                                float _max_54 = max_noftz(smem_pmax[16 + lane], smem_pmax[24 + lane]);
                                float _max_55 = max_noftz(_max_53, _max_54);
                                m_lane_1 = _max_55;
                            } else {
                                float _max_67 = max_noftz(smem_pmax[lane], smem_pmax[16 + lane]);
                                float _max_68 = max_noftz(smem_pmax[32 + lane], smem_pmax[48 + lane]);
                                float _max_69 = max_noftz(_max_67, _max_68);
                                m_lane_1 = _max_69;
                            }
                            float cand_1 = m_lane_1 * softmax_scale_log2;
                            if (it_0 == 0) {
                                if constexpr (TILE_N == 16) {
                                    float _max_56 = max_noftz(cand_1, sink_lane);
                                    cand_1 = _max_56;
                                } else {
                                    float _max_70 = max_noftz(cand_1, sink_lane);
                                    cand_1 = _max_70;
                                }
                            }
                            if constexpr (TILE_N == 16) {
                                float _max_57 = max_noftz(cand_1, m_run);
                                cand_1 = _max_57;
                            } else {
                                float _max_71 = max_noftz(cand_1, m_run);
                                cand_1 = _max_71;
                            }
                            if (it_0 == 0) {
                                grow_1 = 1;
                            }
                            if (cand_1 - m_run > 8.0f) {
                                grow_1 = 1;
                            }
                            if (grow_1 != 0) {
                                if constexpr (TILE_N == 16) {
                                    float _exp2_10 = approx_exp2(m_run - cand_1);
                                    alpha_1 = ((m_run > -CAKE_INF) ? _exp2_10 : 0.0f);
                                } else {
                                    float _exp2_18 = approx_exp2(m_run - cand_1);
                                    alpha_1 = ((m_run > -CAKE_INF) ? _exp2_18 : 0.0f);
                                }
                                l_run = l_run * alpha_1;
                                r_run = r_run * alpha_1;
                                m_run = cand_1;
                            }
                        }
                        float m_scaled_1 = ((m_run > -CAKE_INF) ? m_run : 0.0f);
                        unsigned int _vote_1 = __ballot_sync(0xFFFFFFFF, grow_1 != 0);
                        unsigned int grow_bits_1 = _vote_1;
                        if (it_0 > 0) {
                            if (grow_bits_1 != 0) {
                                float alpha_c_1[(TILE_N / 2)];
                                if constexpr (TILE_N == 16) {
                                    float _shfl_52 = __shfl_sync(0xFFFFFFFF, alpha_1, 0);
                                    alpha_c_1[0] = _shfl_52;
                                    float _shfl_53 = __shfl_sync(0xFFFFFFFF, alpha_1, 1);
                                    alpha_c_1[1] = _shfl_53;
                                    float _shfl_54 = __shfl_sync(0xFFFFFFFF, alpha_1, 2);
                                    alpha_c_1[2] = _shfl_54;
                                    float _shfl_55 = __shfl_sync(0xFFFFFFFF, alpha_1, 3);
                                    alpha_c_1[3] = _shfl_55;
                                    float _shfl_56 = __shfl_sync(0xFFFFFFFF, alpha_1, 4);
                                    alpha_c_1[4] = _shfl_56;
                                    float _shfl_57 = __shfl_sync(0xFFFFFFFF, alpha_1, 5);
                                    alpha_c_1[5] = _shfl_57;
                                    float _shfl_58 = __shfl_sync(0xFFFFFFFF, alpha_1, 6);
                                    alpha_c_1[6] = _shfl_58;
                                    float _shfl_59 = __shfl_sync(0xFFFFFFFF, alpha_1, 7);
                                    alpha_c_1[7] = _shfl_59;
                                } else {
                                    float _shfl_96 = __shfl_sync(0xFFFFFFFF, alpha_1, 0);
                                    alpha_c_1[0] = _shfl_96;
                                    float _shfl_97 = __shfl_sync(0xFFFFFFFF, alpha_1, 1);
                                    alpha_c_1[1] = _shfl_97;
                                    float _shfl_98 = __shfl_sync(0xFFFFFFFF, alpha_1, 2);
                                    alpha_c_1[2] = _shfl_98;
                                    float _shfl_99 = __shfl_sync(0xFFFFFFFF, alpha_1, 3);
                                    alpha_c_1[3] = _shfl_99;
                                    float _shfl_100 = __shfl_sync(0xFFFFFFFF, alpha_1, 4);
                                    alpha_c_1[4] = _shfl_100;
                                    float _shfl_101 = __shfl_sync(0xFFFFFFFF, alpha_1, 5);
                                    alpha_c_1[5] = _shfl_101;
                                    float _shfl_102 = __shfl_sync(0xFFFFFFFF, alpha_1, 6);
                                    alpha_c_1[6] = _shfl_102;
                                    float _shfl_103 = __shfl_sync(0xFFFFFFFF, alpha_1, 7);
                                    alpha_c_1[7] = _shfl_103;
                                    float _shfl_104 = __shfl_sync(0xFFFFFFFF, alpha_1, 8);
                                    alpha_c_1[8] = _shfl_104;
                                    float _shfl_105 = __shfl_sync(0xFFFFFFFF, alpha_1, 9);
                                    alpha_c_1[9] = _shfl_105;
                                    float _shfl_106 = __shfl_sync(0xFFFFFFFF, alpha_1, 10);
                                    alpha_c_1[10] = _shfl_106;
                                    float _shfl_107 = __shfl_sync(0xFFFFFFFF, alpha_1, 11);
                                    alpha_c_1[11] = _shfl_107;
                                    float _shfl_108 = __shfl_sync(0xFFFFFFFF, alpha_1, 12);
                                    alpha_c_1[12] = _shfl_108;
                                    float _shfl_109 = __shfl_sync(0xFFFFFFFF, alpha_1, 13);
                                    alpha_c_1[13] = _shfl_109;
                                    float _shfl_110 = __shfl_sync(0xFFFFFFFF, alpha_1, 14);
                                    alpha_c_1[14] = _shfl_110;
                                    float _shfl_111 = __shfl_sync(0xFFFFFFFF, alpha_1, 15);
                                    alpha_c_1[15] = _shfl_111;
                                }
                                float ov_1[(TILE_N / 2)];
                                if constexpr (TILE_N == 16) {
                                    tmem_ld_x8(&ov_1[0], taddr + 16 + (unsigned int)(tmem_row_origin << 16));
                                } else {
                                    tmem_ld_x16(&ov_1[0], taddr + 32 + (unsigned int)(tmem_row_origin << 16));
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                ov_1[0] = ov_1[0] * alpha_c_1[0];
                                ov_1[1] = ov_1[1] * alpha_c_1[1];
                                ov_1[2] = ov_1[2] * alpha_c_1[2];
                                ov_1[3] = ov_1[3] * alpha_c_1[3];
                                ov_1[4] = ov_1[4] * alpha_c_1[4];
                                ov_1[5] = ov_1[5] * alpha_c_1[5];
                                ov_1[6] = ov_1[6] * alpha_c_1[6];
                                ov_1[7] = ov_1[7] * alpha_c_1[7];
                                if constexpr (TILE_N == 16) {
                                    tmem_st_x8_f32(taddr + 16 + (unsigned int)(tmem_row_origin << 16), ov_1);
                                    tmem_ld_x8(&ov_1[0], taddr + 32 + (unsigned int)(tmem_row_origin << 16));
                                } else {
                                    ov_1[8] = ov_1[8] * alpha_c_1[8];
                                    ov_1[9] = ov_1[9] * alpha_c_1[9];
                                    ov_1[10] = ov_1[10] * alpha_c_1[10];
                                    ov_1[11] = ov_1[11] * alpha_c_1[11];
                                    ov_1[12] = ov_1[12] * alpha_c_1[12];
                                    ov_1[13] = ov_1[13] * alpha_c_1[13];
                                    ov_1[14] = ov_1[14] * alpha_c_1[14];
                                    ov_1[15] = ov_1[15] * alpha_c_1[15];
                                    tmem_st_x16_f32(taddr + 32 + (unsigned int)(tmem_row_origin << 16), ov_1);
                                    tmem_ld_x16(&ov_1[0], taddr + 64 + (unsigned int)(tmem_row_origin << 16));
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                ov_1[0] = ov_1[0] * alpha_c_1[0];
                                ov_1[1] = ov_1[1] * alpha_c_1[1];
                                ov_1[2] = ov_1[2] * alpha_c_1[2];
                                ov_1[3] = ov_1[3] * alpha_c_1[3];
                                ov_1[4] = ov_1[4] * alpha_c_1[4];
                                ov_1[5] = ov_1[5] * alpha_c_1[5];
                                ov_1[6] = ov_1[6] * alpha_c_1[6];
                                ov_1[7] = ov_1[7] * alpha_c_1[7];
                                if constexpr (TILE_N == 16) {
                                    tmem_st_x8_f32(taddr + 32 + (unsigned int)(tmem_row_origin << 16), ov_1);
                                    tmem_ld_x8(&ov_1[0], taddr + 48 + (unsigned int)(tmem_row_origin << 16));
                                } else {
                                    ov_1[8] = ov_1[8] * alpha_c_1[8];
                                    ov_1[9] = ov_1[9] * alpha_c_1[9];
                                    ov_1[10] = ov_1[10] * alpha_c_1[10];
                                    ov_1[11] = ov_1[11] * alpha_c_1[11];
                                    ov_1[12] = ov_1[12] * alpha_c_1[12];
                                    ov_1[13] = ov_1[13] * alpha_c_1[13];
                                    ov_1[14] = ov_1[14] * alpha_c_1[14];
                                    ov_1[15] = ov_1[15] * alpha_c_1[15];
                                    tmem_st_x16_f32(taddr + 64 + (unsigned int)(tmem_row_origin << 16), ov_1);
                                    tmem_ld_x16(&ov_1[0], taddr + 96 + (unsigned int)(tmem_row_origin << 16));
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                ov_1[0] = ov_1[0] * alpha_c_1[0];
                                ov_1[1] = ov_1[1] * alpha_c_1[1];
                                ov_1[2] = ov_1[2] * alpha_c_1[2];
                                ov_1[3] = ov_1[3] * alpha_c_1[3];
                                ov_1[4] = ov_1[4] * alpha_c_1[4];
                                ov_1[5] = ov_1[5] * alpha_c_1[5];
                                ov_1[6] = ov_1[6] * alpha_c_1[6];
                                ov_1[7] = ov_1[7] * alpha_c_1[7];
                                if constexpr (TILE_N == 16) {
                                    tmem_st_x8_f32(taddr + 48 + (unsigned int)(tmem_row_origin << 16), ov_1);
                                    tmem_ld_x8(&ov_1[0], taddr + 64 + (unsigned int)(tmem_row_origin << 16));
                                } else {
                                    ov_1[8] = ov_1[8] * alpha_c_1[8];
                                    ov_1[9] = ov_1[9] * alpha_c_1[9];
                                    ov_1[10] = ov_1[10] * alpha_c_1[10];
                                    ov_1[11] = ov_1[11] * alpha_c_1[11];
                                    ov_1[12] = ov_1[12] * alpha_c_1[12];
                                    ov_1[13] = ov_1[13] * alpha_c_1[13];
                                    ov_1[14] = ov_1[14] * alpha_c_1[14];
                                    ov_1[15] = ov_1[15] * alpha_c_1[15];
                                    tmem_st_x16_f32(taddr + 96 + (unsigned int)(tmem_row_origin << 16), ov_1);
                                    tmem_ld_x16(&ov_1[0], taddr + 128 + (unsigned int)(tmem_row_origin << 16));
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                ov_1[0] = ov_1[0] * alpha_c_1[0];
                                ov_1[1] = ov_1[1] * alpha_c_1[1];
                                ov_1[2] = ov_1[2] * alpha_c_1[2];
                                ov_1[3] = ov_1[3] * alpha_c_1[3];
                                ov_1[4] = ov_1[4] * alpha_c_1[4];
                                ov_1[5] = ov_1[5] * alpha_c_1[5];
                                ov_1[6] = ov_1[6] * alpha_c_1[6];
                                ov_1[7] = ov_1[7] * alpha_c_1[7];
                                if constexpr (TILE_N == 16) {
                                    tmem_st_x8_f32(taddr + 64 + (unsigned int)(tmem_row_origin << 16), ov_1);
                                } else {
                                    ov_1[8] = ov_1[8] * alpha_c_1[8];
                                    ov_1[9] = ov_1[9] * alpha_c_1[9];
                                    ov_1[10] = ov_1[10] * alpha_c_1[10];
                                    ov_1[11] = ov_1[11] * alpha_c_1[11];
                                    ov_1[12] = ov_1[12] * alpha_c_1[12];
                                    ov_1[13] = ov_1[13] * alpha_c_1[13];
                                    ov_1[14] = ov_1[14] * alpha_c_1[14];
                                    ov_1[15] = ov_1[15] * alpha_c_1[15];
                                    tmem_st_x16_f32(taddr + 128 + (unsigned int)(tmem_row_origin << 16), ov_1);
                                }
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                        }
                        float col_max_1[(TILE_N / 2)];
                        if constexpr (TILE_N == 16) {
                            float _shfl_60 = __shfl_sync(0xFFFFFFFF, m_scaled_1, 0);
                            col_max_1[0] = _shfl_60;
                            float _shfl_61 = __shfl_sync(0xFFFFFFFF, m_scaled_1, 1);
                            col_max_1[1] = _shfl_61;
                            float _shfl_62 = __shfl_sync(0xFFFFFFFF, m_scaled_1, 2);
                            col_max_1[2] = _shfl_62;
                            float _shfl_63 = __shfl_sync(0xFFFFFFFF, m_scaled_1, 3);
                            col_max_1[3] = _shfl_63;
                            float _shfl_64 = __shfl_sync(0xFFFFFFFF, m_scaled_1, 4);
                            col_max_1[4] = _shfl_64;
                            float _shfl_65 = __shfl_sync(0xFFFFFFFF, m_scaled_1, 5);
                            col_max_1[5] = _shfl_65;
                            float _shfl_66 = __shfl_sync(0xFFFFFFFF, m_scaled_1, 6);
                            col_max_1[6] = _shfl_66;
                            float _shfl_67 = __shfl_sync(0xFFFFFFFF, m_scaled_1, 7);
                            col_max_1[7] = _shfl_67;
                            float _exp2_11 = approx_exp2(score_values_1[0] * softmax_scale_log2 - col_max_1[0]);
                            score_values_1[0] = _exp2_11;
                            float _exp2_12 = approx_exp2(score_values_1[1] * softmax_scale_log2 - col_max_1[1]);
                            score_values_1[1] = _exp2_12;
                            float _exp2_13 = approx_exp2(score_values_1[2] * softmax_scale_log2 - col_max_1[2]);
                            score_values_1[2] = _exp2_13;
                            float _exp2_14 = approx_exp2(score_values_1[3] * softmax_scale_log2 - col_max_1[3]);
                            score_values_1[3] = _exp2_14;
                            float _exp2_15 = approx_exp2(score_values_1[4] * softmax_scale_log2 - col_max_1[4]);
                            score_values_1[4] = _exp2_15;
                            float _exp2_16 = approx_exp2(score_values_1[5] * softmax_scale_log2 - col_max_1[5]);
                            score_values_1[5] = _exp2_16;
                            float _exp2_17 = approx_exp2(score_values_1[6] * softmax_scale_log2 - col_max_1[6]);
                            score_values_1[6] = _exp2_17;
                            float _exp2_18 = approx_exp2(score_values_1[7] * softmax_scale_log2 - col_max_1[7]);
                            score_values_1[7] = _exp2_18;
                        } else {
                            float _shfl_112 = __shfl_sync(0xFFFFFFFF, m_scaled_1, 0);
                            col_max_1[0] = _shfl_112;
                            float _shfl_113 = __shfl_sync(0xFFFFFFFF, m_scaled_1, 1);
                            col_max_1[1] = _shfl_113;
                            float _shfl_114 = __shfl_sync(0xFFFFFFFF, m_scaled_1, 2);
                            col_max_1[2] = _shfl_114;
                            float _shfl_115 = __shfl_sync(0xFFFFFFFF, m_scaled_1, 3);
                            col_max_1[3] = _shfl_115;
                            float _shfl_116 = __shfl_sync(0xFFFFFFFF, m_scaled_1, 4);
                            col_max_1[4] = _shfl_116;
                            float _shfl_117 = __shfl_sync(0xFFFFFFFF, m_scaled_1, 5);
                            col_max_1[5] = _shfl_117;
                            float _shfl_118 = __shfl_sync(0xFFFFFFFF, m_scaled_1, 6);
                            col_max_1[6] = _shfl_118;
                            float _shfl_119 = __shfl_sync(0xFFFFFFFF, m_scaled_1, 7);
                            col_max_1[7] = _shfl_119;
                            float _shfl_120 = __shfl_sync(0xFFFFFFFF, m_scaled_1, 8);
                            col_max_1[8] = _shfl_120;
                            float _shfl_121 = __shfl_sync(0xFFFFFFFF, m_scaled_1, 9);
                            col_max_1[9] = _shfl_121;
                            float _shfl_122 = __shfl_sync(0xFFFFFFFF, m_scaled_1, 10);
                            col_max_1[10] = _shfl_122;
                            float _shfl_123 = __shfl_sync(0xFFFFFFFF, m_scaled_1, 11);
                            col_max_1[11] = _shfl_123;
                            float _shfl_124 = __shfl_sync(0xFFFFFFFF, m_scaled_1, 12);
                            col_max_1[12] = _shfl_124;
                            float _shfl_125 = __shfl_sync(0xFFFFFFFF, m_scaled_1, 13);
                            col_max_1[13] = _shfl_125;
                            float _shfl_126 = __shfl_sync(0xFFFFFFFF, m_scaled_1, 14);
                            col_max_1[14] = _shfl_126;
                            float _shfl_127 = __shfl_sync(0xFFFFFFFF, m_scaled_1, 15);
                            col_max_1[15] = _shfl_127;
                            float _exp2_19 = approx_exp2(score_values_1[0] * softmax_scale_log2 - col_max_1[0]);
                            score_values_1[0] = _exp2_19;
                            float _exp2_20 = approx_exp2(score_values_1[1] * softmax_scale_log2 - col_max_1[1]);
                            score_values_1[1] = _exp2_20;
                            float _exp2_21 = approx_exp2(score_values_1[2] * softmax_scale_log2 - col_max_1[2]);
                            score_values_1[2] = _exp2_21;
                            float _exp2_22 = approx_exp2(score_values_1[3] * softmax_scale_log2 - col_max_1[3]);
                            score_values_1[3] = _exp2_22;
                            float _exp2_23 = approx_exp2(score_values_1[4] * softmax_scale_log2 - col_max_1[4]);
                            score_values_1[4] = _exp2_23;
                            float _exp2_24 = approx_exp2(score_values_1[5] * softmax_scale_log2 - col_max_1[5]);
                            score_values_1[5] = _exp2_24;
                            float _exp2_25 = approx_exp2(score_values_1[6] * softmax_scale_log2 - col_max_1[6]);
                            score_values_1[6] = _exp2_25;
                            float _exp2_26 = approx_exp2(score_values_1[7] * softmax_scale_log2 - col_max_1[7]);
                            score_values_1[7] = _exp2_26;
                            float _exp2_27 = approx_exp2(score_values_1[8] * softmax_scale_log2 - col_max_1[8]);
                            score_values_1[8] = _exp2_27;
                            float _exp2_28 = approx_exp2(score_values_1[9] * softmax_scale_log2 - col_max_1[9]);
                            score_values_1[9] = _exp2_28;
                            float _exp2_29 = approx_exp2(score_values_1[10] * softmax_scale_log2 - col_max_1[10]);
                            score_values_1[10] = _exp2_29;
                            float _exp2_30 = approx_exp2(score_values_1[11] * softmax_scale_log2 - col_max_1[11]);
                            score_values_1[11] = _exp2_30;
                            float _exp2_31 = approx_exp2(score_values_1[12] * softmax_scale_log2 - col_max_1[12]);
                            score_values_1[12] = _exp2_31;
                            float _exp2_32 = approx_exp2(score_values_1[13] * softmax_scale_log2 - col_max_1[13]);
                            score_values_1[13] = _exp2_32;
                            float _exp2_33 = approx_exp2(score_values_1[14] * softmax_scale_log2 - col_max_1[14]);
                            score_values_1[14] = _exp2_33;
                            float _exp2_34 = approx_exp2(score_values_1[15] * softmax_scale_log2 - col_max_1[15]);
                            score_values_1[15] = _exp2_34;
                        }
                        {
                            if constexpr (TILE_N == 16) {
                                uint16_t _fp8_pair_106;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_106) : "f"(0.0f), "f"(score_values_1[0]));
                                uint32_t _byte_106 = (uint32_t)(_fp8_pair_106 & 0xFF);
                                uint32_t _addr_106 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(row ^ (row >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_106), "r"(_byte_106) : "memory");
                            } else {
                                uint16_t _fp8_pair_122;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_122) : "f"(0.0f), "f"(score_values_1[0]));
                                uint32_t _byte_122 = (uint32_t)(_fp8_pair_122 & 0xFF);
                                uint32_t _addr_122 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(row ^ (row >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_122), "r"(_byte_122) : "memory");
                            }
                        }
                        if constexpr (TILE_N == 16) {
                            float _fp8_rt_8;
                            uint16_t _e4m3x2_107;
                            uint32_t _f16x2_107;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_107) : "f"(0.0f), "f"(score_values_1[0]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_107) : "h"(_e4m3x2_107));
                            uint16_t _fp8_h0_107 = (uint16_t)(_f16x2_107 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_8) : "h"(_fp8_h0_107));
                            tr_vals_1[0] = _fp8_rt_8;
                        } else {
                            float _fp8_rt_16;
                            uint16_t _e4m3x2_123;
                            uint32_t _f16x2_123;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_123) : "f"(0.0f), "f"(score_values_1[0]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_123) : "h"(_e4m3x2_123));
                            uint16_t _fp8_h0_123 = (uint16_t)(_f16x2_123 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_16) : "h"(_fp8_h0_123));
                            tr_vals_1[0] = _fp8_rt_16;
                        }
                        {
                            if constexpr (TILE_N == 16) {
                                uint16_t _fp8_pair_108;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_108) : "f"(0.0f), "f"(score_values_1[1]));
                                uint32_t _byte_108 = (uint32_t)(_fp8_pair_108 & 0xFF);
                                uint32_t _addr_108 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(128 + row ^ (128 + row >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_108), "r"(_byte_108) : "memory");
                            } else {
                                uint16_t _fp8_pair_124;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_124) : "f"(0.0f), "f"(score_values_1[1]));
                                uint32_t _byte_124 = (uint32_t)(_fp8_pair_124 & 0xFF);
                                uint32_t _addr_124 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(128 + row ^ (128 + row >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_124), "r"(_byte_124) : "memory");
                            }
                        }
                        if constexpr (TILE_N == 16) {
                            float _fp8_rt_9;
                            uint16_t _e4m3x2_109;
                            uint32_t _f16x2_109;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_109) : "f"(0.0f), "f"(score_values_1[1]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_109) : "h"(_e4m3x2_109));
                            uint16_t _fp8_h0_109 = (uint16_t)(_f16x2_109 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_9) : "h"(_fp8_h0_109));
                            tr_vals_1[1] = _fp8_rt_9;
                        } else {
                            float _fp8_rt_17;
                            uint16_t _e4m3x2_125;
                            uint32_t _f16x2_125;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_125) : "f"(0.0f), "f"(score_values_1[1]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_125) : "h"(_e4m3x2_125));
                            uint16_t _fp8_h0_125 = (uint16_t)(_f16x2_125 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_17) : "h"(_fp8_h0_125));
                            tr_vals_1[1] = _fp8_rt_17;
                        }
                        {
                            if constexpr (TILE_N == 16) {
                                uint16_t _fp8_pair_110;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_110) : "f"(0.0f), "f"(score_values_1[2]));
                                uint32_t _byte_110 = (uint32_t)(_fp8_pair_110 & 0xFF);
                                uint32_t _addr_110 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(256 + row ^ (256 + row >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_110), "r"(_byte_110) : "memory");
                            } else {
                                uint16_t _fp8_pair_126;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_126) : "f"(0.0f), "f"(score_values_1[2]));
                                uint32_t _byte_126 = (uint32_t)(_fp8_pair_126 & 0xFF);
                                uint32_t _addr_126 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(256 + row ^ (256 + row >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_126), "r"(_byte_126) : "memory");
                            }
                        }
                        if constexpr (TILE_N == 16) {
                            float _fp8_rt_10;
                            uint16_t _e4m3x2_111;
                            uint32_t _f16x2_111;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_111) : "f"(0.0f), "f"(score_values_1[2]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_111) : "h"(_e4m3x2_111));
                            uint16_t _fp8_h0_111 = (uint16_t)(_f16x2_111 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_10) : "h"(_fp8_h0_111));
                            tr_vals_1[2] = _fp8_rt_10;
                        } else {
                            float _fp8_rt_18;
                            uint16_t _e4m3x2_127;
                            uint32_t _f16x2_127;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_127) : "f"(0.0f), "f"(score_values_1[2]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_127) : "h"(_e4m3x2_127));
                            uint16_t _fp8_h0_127 = (uint16_t)(_f16x2_127 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_18) : "h"(_fp8_h0_127));
                            tr_vals_1[2] = _fp8_rt_18;
                        }
                        {
                            if constexpr (TILE_N == 16) {
                                uint16_t _fp8_pair_112;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_112) : "f"(0.0f), "f"(score_values_1[3]));
                                uint32_t _byte_112 = (uint32_t)(_fp8_pair_112 & 0xFF);
                                uint32_t _addr_112 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(384 + row ^ (384 + row >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_112), "r"(_byte_112) : "memory");
                            } else {
                                uint16_t _fp8_pair_128;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_128) : "f"(0.0f), "f"(score_values_1[3]));
                                uint32_t _byte_128 = (uint32_t)(_fp8_pair_128 & 0xFF);
                                uint32_t _addr_128 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(384 + row ^ (384 + row >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_128), "r"(_byte_128) : "memory");
                            }
                        }
                        if constexpr (TILE_N == 16) {
                            float _fp8_rt_11;
                            uint16_t _e4m3x2_113;
                            uint32_t _f16x2_113;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_113) : "f"(0.0f), "f"(score_values_1[3]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_113) : "h"(_e4m3x2_113));
                            uint16_t _fp8_h0_113 = (uint16_t)(_f16x2_113 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_11) : "h"(_fp8_h0_113));
                            tr_vals_1[3] = _fp8_rt_11;
                        } else {
                            float _fp8_rt_19;
                            uint16_t _e4m3x2_129;
                            uint32_t _f16x2_129;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_129) : "f"(0.0f), "f"(score_values_1[3]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_129) : "h"(_e4m3x2_129));
                            uint16_t _fp8_h0_129 = (uint16_t)(_f16x2_129 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_19) : "h"(_fp8_h0_129));
                            tr_vals_1[3] = _fp8_rt_19;
                        }
                        {
                            if constexpr (TILE_N == 16) {
                                uint16_t _fp8_pair_114;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_114) : "f"(0.0f), "f"(score_values_1[4]));
                                uint32_t _byte_114 = (uint32_t)(_fp8_pair_114 & 0xFF);
                                uint32_t _addr_114 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(512 + row ^ (512 + row >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_114), "r"(_byte_114) : "memory");
                            } else {
                                uint16_t _fp8_pair_130;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_130) : "f"(0.0f), "f"(score_values_1[4]));
                                uint32_t _byte_130 = (uint32_t)(_fp8_pair_130 & 0xFF);
                                uint32_t _addr_130 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(512 + row ^ (512 + row >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_130), "r"(_byte_130) : "memory");
                            }
                        }
                        if constexpr (TILE_N == 16) {
                            float _fp8_rt_12;
                            uint16_t _e4m3x2_115;
                            uint32_t _f16x2_115;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_115) : "f"(0.0f), "f"(score_values_1[4]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_115) : "h"(_e4m3x2_115));
                            uint16_t _fp8_h0_115 = (uint16_t)(_f16x2_115 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_12) : "h"(_fp8_h0_115));
                            tr_vals_1[4] = _fp8_rt_12;
                        } else {
                            float _fp8_rt_20;
                            uint16_t _e4m3x2_131;
                            uint32_t _f16x2_131;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_131) : "f"(0.0f), "f"(score_values_1[4]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_131) : "h"(_e4m3x2_131));
                            uint16_t _fp8_h0_131 = (uint16_t)(_f16x2_131 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_20) : "h"(_fp8_h0_131));
                            tr_vals_1[4] = _fp8_rt_20;
                        }
                        {
                            if constexpr (TILE_N == 16) {
                                uint16_t _fp8_pair_116;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_116) : "f"(0.0f), "f"(score_values_1[5]));
                                uint32_t _byte_116 = (uint32_t)(_fp8_pair_116 & 0xFF);
                                uint32_t _addr_116 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(640 + row ^ (640 + row >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_116), "r"(_byte_116) : "memory");
                            } else {
                                uint16_t _fp8_pair_132;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_132) : "f"(0.0f), "f"(score_values_1[5]));
                                uint32_t _byte_132 = (uint32_t)(_fp8_pair_132 & 0xFF);
                                uint32_t _addr_132 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(640 + row ^ (640 + row >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_132), "r"(_byte_132) : "memory");
                            }
                        }
                        if constexpr (TILE_N == 16) {
                            float _fp8_rt_13;
                            uint16_t _e4m3x2_117;
                            uint32_t _f16x2_117;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_117) : "f"(0.0f), "f"(score_values_1[5]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_117) : "h"(_e4m3x2_117));
                            uint16_t _fp8_h0_117 = (uint16_t)(_f16x2_117 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_13) : "h"(_fp8_h0_117));
                            tr_vals_1[5] = _fp8_rt_13;
                        } else {
                            float _fp8_rt_21;
                            uint16_t _e4m3x2_133;
                            uint32_t _f16x2_133;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_133) : "f"(0.0f), "f"(score_values_1[5]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_133) : "h"(_e4m3x2_133));
                            uint16_t _fp8_h0_133 = (uint16_t)(_f16x2_133 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_21) : "h"(_fp8_h0_133));
                            tr_vals_1[5] = _fp8_rt_21;
                        }
                        {
                            if constexpr (TILE_N == 16) {
                                uint16_t _fp8_pair_118;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_118) : "f"(0.0f), "f"(score_values_1[6]));
                                uint32_t _byte_118 = (uint32_t)(_fp8_pair_118 & 0xFF);
                                uint32_t _addr_118 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(768 + row ^ (768 + row >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_118), "r"(_byte_118) : "memory");
                            } else {
                                uint16_t _fp8_pair_134;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_134) : "f"(0.0f), "f"(score_values_1[6]));
                                uint32_t _byte_134 = (uint32_t)(_fp8_pair_134 & 0xFF);
                                uint32_t _addr_134 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(768 + row ^ (768 + row >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_134), "r"(_byte_134) : "memory");
                            }
                        }
                        if constexpr (TILE_N == 16) {
                            float _fp8_rt_14;
                            uint16_t _e4m3x2_119;
                            uint32_t _f16x2_119;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_119) : "f"(0.0f), "f"(score_values_1[6]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_119) : "h"(_e4m3x2_119));
                            uint16_t _fp8_h0_119 = (uint16_t)(_f16x2_119 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_14) : "h"(_fp8_h0_119));
                            tr_vals_1[6] = _fp8_rt_14;
                        } else {
                            float _fp8_rt_22;
                            uint16_t _e4m3x2_135;
                            uint32_t _f16x2_135;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_135) : "f"(0.0f), "f"(score_values_1[6]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_135) : "h"(_e4m3x2_135));
                            uint16_t _fp8_h0_135 = (uint16_t)(_f16x2_135 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_22) : "h"(_fp8_h0_135));
                            tr_vals_1[6] = _fp8_rt_22;
                        }
                        {
                            if constexpr (TILE_N == 16) {
                                uint16_t _fp8_pair_120;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_120) : "f"(0.0f), "f"(score_values_1[7]));
                                uint32_t _byte_120 = (uint32_t)(_fp8_pair_120 & 0xFF);
                                uint32_t _addr_120 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(896 + row ^ (896 + row >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_120), "r"(_byte_120) : "memory");
                            } else {
                                uint16_t _fp8_pair_136;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_136) : "f"(0.0f), "f"(score_values_1[7]));
                                uint32_t _byte_136 = (uint32_t)(_fp8_pair_136 & 0xFF);
                                uint32_t _addr_136 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(896 + row ^ (896 + row >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_136), "r"(_byte_136) : "memory");
                            }
                        }
                        if constexpr (TILE_N == 16) {
                            float _fp8_rt_15;
                            uint16_t _e4m3x2_121;
                            uint32_t _f16x2_121;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_121) : "f"(0.0f), "f"(score_values_1[7]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_121) : "h"(_e4m3x2_121));
                            uint16_t _fp8_h0_121 = (uint16_t)(_f16x2_121 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_15) : "h"(_fp8_h0_121));
                            tr_vals_1[7] = _fp8_rt_15;
                            int hi_bit_21_1 = lane & 16;
                            float send_22_1 = ((hi_bit_21_1 != 0) ? score_values_1[0] : score_values_1[4]);
                            float keep_23_1 = ((hi_bit_21_1 != 0) ? score_values_1[4] : score_values_1[0]);
                            float _shfl_68 = __shfl_sync(0xFFFFFFFF, send_22_1, lane ^ 16);
                            float recv_24_1 = _shfl_68;
                            score_values_1[0] = keep_23_1 + recv_24_1;
                            float send_25_1 = ((hi_bit_21_1 != 0) ? score_values_1[1] : score_values_1[5]);
                            float keep_26_1 = ((hi_bit_21_1 != 0) ? score_values_1[5] : score_values_1[1]);
                            float _shfl_69 = __shfl_sync(0xFFFFFFFF, send_25_1, lane ^ 16);
                            float recv_27_1 = _shfl_69;
                            score_values_1[1] = keep_26_1 + recv_27_1;
                            float send_28_1 = ((hi_bit_21_1 != 0) ? score_values_1[2] : score_values_1[6]);
                            float keep_29_1 = ((hi_bit_21_1 != 0) ? score_values_1[6] : score_values_1[2]);
                            float _shfl_70 = __shfl_sync(0xFFFFFFFF, send_28_1, lane ^ 16);
                            float recv_30_1 = _shfl_70;
                            score_values_1[2] = keep_29_1 + recv_30_1;
                            float send_31_1 = ((hi_bit_21_1 != 0) ? score_values_1[3] : score_values_1[7]);
                            float keep_32_1 = ((hi_bit_21_1 != 0) ? score_values_1[7] : score_values_1[3]);
                            float _shfl_71 = __shfl_sync(0xFFFFFFFF, send_31_1, lane ^ 16);
                            float recv_33_1 = _shfl_71;
                            score_values_1[3] = keep_32_1 + recv_33_1;
                            int hi_bit_34_1 = lane & 8;
                            float send_35_1 = ((hi_bit_34_1 != 0) ? score_values_1[0] : score_values_1[2]);
                            float keep_36_1 = ((hi_bit_34_1 != 0) ? score_values_1[2] : score_values_1[0]);
                            float _shfl_72 = __shfl_sync(0xFFFFFFFF, send_35_1, lane ^ 8);
                            float recv_37_1 = _shfl_72;
                            score_values_1[0] = keep_36_1 + recv_37_1;
                            float send_38_1 = ((hi_bit_34_1 != 0) ? score_values_1[1] : score_values_1[3]);
                            float keep_39_1 = ((hi_bit_34_1 != 0) ? score_values_1[3] : score_values_1[1]);
                            float _shfl_73 = __shfl_sync(0xFFFFFFFF, send_38_1, lane ^ 8);
                            float recv_40_1 = _shfl_73;
                            score_values_1[1] = keep_39_1 + recv_40_1;
                            int hi_bit_41_1 = lane & 4;
                            float send_42_1 = ((hi_bit_41_1 != 0) ? score_values_1[0] : score_values_1[1]);
                            float keep_43_1 = ((hi_bit_41_1 != 0) ? score_values_1[1] : score_values_1[0]);
                            float _shfl_74 = __shfl_sync(0xFFFFFFFF, send_42_1, lane ^ 4);
                            float recv_44_1 = _shfl_74;
                            score_values_1[0] = keep_43_1 + recv_44_1;
                            float _shfl_75 = __shfl_sync(0xFFFFFFFF, score_values_1[0], lane ^ 2);
                            float other_45_1 = _shfl_75;
                            score_values_1[0] = score_values_1[0] + other_45_1;
                            float _shfl_76 = __shfl_sync(0xFFFFFFFF, score_values_1[0], lane ^ 1);
                            float other_46_1 = _shfl_76;
                            score_values_1[0] = score_values_1[0] + other_46_1;
                            int hi_bit_47_1 = lane & 16;
                            float send_48_1 = ((hi_bit_47_1 != 0) ? tr_vals_1[0] : tr_vals_1[4]);
                            float keep_49_1 = ((hi_bit_47_1 != 0) ? tr_vals_1[4] : tr_vals_1[0]);
                            float _shfl_77 = __shfl_sync(0xFFFFFFFF, send_48_1, lane ^ 16);
                            float recv_50_1 = _shfl_77;
                            tr_vals_1[0] = keep_49_1 + recv_50_1;
                            float send_51_1 = ((hi_bit_47_1 != 0) ? tr_vals_1[1] : tr_vals_1[5]);
                            float keep_52_1 = ((hi_bit_47_1 != 0) ? tr_vals_1[5] : tr_vals_1[1]);
                            float _shfl_78 = __shfl_sync(0xFFFFFFFF, send_51_1, lane ^ 16);
                            float recv_53_1 = _shfl_78;
                            tr_vals_1[1] = keep_52_1 + recv_53_1;
                            float send_54_1 = ((hi_bit_47_1 != 0) ? tr_vals_1[2] : tr_vals_1[6]);
                            float keep_55_1 = ((hi_bit_47_1 != 0) ? tr_vals_1[6] : tr_vals_1[2]);
                            float _shfl_79 = __shfl_sync(0xFFFFFFFF, send_54_1, lane ^ 16);
                            float recv_56_1 = _shfl_79;
                            tr_vals_1[2] = keep_55_1 + recv_56_1;
                            float send_57_1 = ((hi_bit_47_1 != 0) ? tr_vals_1[3] : tr_vals_1[7]);
                            float keep_58_1 = ((hi_bit_47_1 != 0) ? tr_vals_1[7] : tr_vals_1[3]);
                            float _shfl_80 = __shfl_sync(0xFFFFFFFF, send_57_1, lane ^ 16);
                            float recv_59_1 = _shfl_80;
                            tr_vals_1[3] = keep_58_1 + recv_59_1;
                            int hi_bit_60_1 = lane & 8;
                            float send_61_1 = ((hi_bit_60_1 != 0) ? tr_vals_1[0] : tr_vals_1[2]);
                            float keep_62_1 = ((hi_bit_60_1 != 0) ? tr_vals_1[2] : tr_vals_1[0]);
                            float _shfl_81 = __shfl_sync(0xFFFFFFFF, send_61_1, lane ^ 8);
                            float recv_63_1 = _shfl_81;
                            tr_vals_1[0] = keep_62_1 + recv_63_1;
                            float send_64_1 = ((hi_bit_60_1 != 0) ? tr_vals_1[1] : tr_vals_1[3]);
                            float keep_65_1 = ((hi_bit_60_1 != 0) ? tr_vals_1[3] : tr_vals_1[1]);
                            float _shfl_82 = __shfl_sync(0xFFFFFFFF, send_64_1, lane ^ 8);
                            float recv_66_1 = _shfl_82;
                            tr_vals_1[1] = keep_65_1 + recv_66_1;
                            int hi_bit_67_1 = lane & 4;
                            float send_68_1 = ((hi_bit_67_1 != 0) ? tr_vals_1[0] : tr_vals_1[1]);
                            float keep_69_1 = ((hi_bit_67_1 != 0) ? tr_vals_1[1] : tr_vals_1[0]);
                            float _shfl_83 = __shfl_sync(0xFFFFFFFF, send_68_1, lane ^ 4);
                            float recv_70_1 = _shfl_83;
                            tr_vals_1[0] = keep_69_1 + recv_70_1;
                            float _shfl_84 = __shfl_sync(0xFFFFFFFF, tr_vals_1[0], lane ^ 2);
                            float other_71_1 = _shfl_84;
                            tr_vals_1[0] = tr_vals_1[0] + other_71_1;
                            float _shfl_85 = __shfl_sync(0xFFFFFFFF, tr_vals_1[0], lane ^ 1);
                            float other_72_1 = _shfl_85;
                            tr_vals_1[0] = tr_vals_1[0] + other_72_1;
                        } else {
                            float _fp8_rt_23;
                            uint16_t _e4m3x2_137;
                            uint32_t _f16x2_137;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_137) : "f"(0.0f), "f"(score_values_1[7]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_137) : "h"(_e4m3x2_137));
                            uint16_t _fp8_h0_137 = (uint16_t)(_f16x2_137 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_23) : "h"(_fp8_h0_137));
                            tr_vals_1[7] = _fp8_rt_23;
                            {
                                uint16_t _fp8_pair_138;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_138) : "f"(0.0f), "f"(score_values_1[8]));
                                uint32_t _byte_138 = (uint32_t)(_fp8_pair_138 & 0xFF);
                                uint32_t _addr_138 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1024 + row ^ (1024 + row >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_138), "r"(_byte_138) : "memory");
                            }
                            float _fp8_rt_24;
                            uint16_t _e4m3x2_139;
                            uint32_t _f16x2_139;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_139) : "f"(0.0f), "f"(score_values_1[8]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_139) : "h"(_e4m3x2_139));
                            uint16_t _fp8_h0_139 = (uint16_t)(_f16x2_139 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_24) : "h"(_fp8_h0_139));
                            tr_vals_1[8] = _fp8_rt_24;
                            {
                                uint16_t _fp8_pair_140;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_140) : "f"(0.0f), "f"(score_values_1[9]));
                                uint32_t _byte_140 = (uint32_t)(_fp8_pair_140 & 0xFF);
                                uint32_t _addr_140 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1152 + row ^ (1152 + row >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_140), "r"(_byte_140) : "memory");
                            }
                            float _fp8_rt_25;
                            uint16_t _e4m3x2_141;
                            uint32_t _f16x2_141;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_141) : "f"(0.0f), "f"(score_values_1[9]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_141) : "h"(_e4m3x2_141));
                            uint16_t _fp8_h0_141 = (uint16_t)(_f16x2_141 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_25) : "h"(_fp8_h0_141));
                            tr_vals_1[9] = _fp8_rt_25;
                            {
                                uint16_t _fp8_pair_142;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_142) : "f"(0.0f), "f"(score_values_1[10]));
                                uint32_t _byte_142 = (uint32_t)(_fp8_pair_142 & 0xFF);
                                uint32_t _addr_142 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1280 + row ^ (1280 + row >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_142), "r"(_byte_142) : "memory");
                            }
                            float _fp8_rt_26;
                            uint16_t _e4m3x2_143;
                            uint32_t _f16x2_143;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_143) : "f"(0.0f), "f"(score_values_1[10]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_143) : "h"(_e4m3x2_143));
                            uint16_t _fp8_h0_143 = (uint16_t)(_f16x2_143 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_26) : "h"(_fp8_h0_143));
                            tr_vals_1[10] = _fp8_rt_26;
                            {
                                uint16_t _fp8_pair_144;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_144) : "f"(0.0f), "f"(score_values_1[11]));
                                uint32_t _byte_144 = (uint32_t)(_fp8_pair_144 & 0xFF);
                                uint32_t _addr_144 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1408 + row ^ (1408 + row >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_144), "r"(_byte_144) : "memory");
                            }
                            float _fp8_rt_27;
                            uint16_t _e4m3x2_145;
                            uint32_t _f16x2_145;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_145) : "f"(0.0f), "f"(score_values_1[11]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_145) : "h"(_e4m3x2_145));
                            uint16_t _fp8_h0_145 = (uint16_t)(_f16x2_145 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_27) : "h"(_fp8_h0_145));
                            tr_vals_1[11] = _fp8_rt_27;
                            {
                                uint16_t _fp8_pair_146;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_146) : "f"(0.0f), "f"(score_values_1[12]));
                                uint32_t _byte_146 = (uint32_t)(_fp8_pair_146 & 0xFF);
                                uint32_t _addr_146 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1536 + row ^ (1536 + row >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_146), "r"(_byte_146) : "memory");
                            }
                            float _fp8_rt_28;
                            uint16_t _e4m3x2_147;
                            uint32_t _f16x2_147;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_147) : "f"(0.0f), "f"(score_values_1[12]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_147) : "h"(_e4m3x2_147));
                            uint16_t _fp8_h0_147 = (uint16_t)(_f16x2_147 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_28) : "h"(_fp8_h0_147));
                            tr_vals_1[12] = _fp8_rt_28;
                            {
                                uint16_t _fp8_pair_148;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_148) : "f"(0.0f), "f"(score_values_1[13]));
                                uint32_t _byte_148 = (uint32_t)(_fp8_pair_148 & 0xFF);
                                uint32_t _addr_148 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1664 + row ^ (1664 + row >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_148), "r"(_byte_148) : "memory");
                            }
                            float _fp8_rt_29;
                            uint16_t _e4m3x2_149;
                            uint32_t _f16x2_149;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_149) : "f"(0.0f), "f"(score_values_1[13]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_149) : "h"(_e4m3x2_149));
                            uint16_t _fp8_h0_149 = (uint16_t)(_f16x2_149 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_29) : "h"(_fp8_h0_149));
                            tr_vals_1[13] = _fp8_rt_29;
                            {
                                uint16_t _fp8_pair_150;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_150) : "f"(0.0f), "f"(score_values_1[14]));
                                uint32_t _byte_150 = (uint32_t)(_fp8_pair_150 & 0xFF);
                                uint32_t _addr_150 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1792 + row ^ (1792 + row >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_150), "r"(_byte_150) : "memory");
                            }
                            float _fp8_rt_30;
                            uint16_t _e4m3x2_151;
                            uint32_t _f16x2_151;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_151) : "f"(0.0f), "f"(score_values_1[14]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_151) : "h"(_e4m3x2_151));
                            uint16_t _fp8_h0_151 = (uint16_t)(_f16x2_151 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_30) : "h"(_fp8_h0_151));
                            tr_vals_1[14] = _fp8_rt_30;
                            {
                                uint16_t _fp8_pair_152;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_152) : "f"(0.0f), "f"(score_values_1[15]));
                                uint32_t _byte_152 = (uint32_t)(_fp8_pair_152 & 0xFF);
                                uint32_t _addr_152 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1920 + row ^ (1920 + row >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_152), "r"(_byte_152) : "memory");
                            }
                            float _fp8_rt_31;
                            uint16_t _e4m3x2_153;
                            uint32_t _f16x2_153;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_153) : "f"(0.0f), "f"(score_values_1[15]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_153) : "h"(_e4m3x2_153));
                            uint16_t _fp8_h0_153 = (uint16_t)(_f16x2_153 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_31) : "h"(_fp8_h0_153));
                            tr_vals_1[15] = _fp8_rt_31;
                            int hi_bit_45_1 = lane & 16;
                            float send_46_1 = ((hi_bit_45_1 != 0) ? score_values_1[0] : score_values_1[8]);
                            float keep_47_1 = ((hi_bit_45_1 != 0) ? score_values_1[8] : score_values_1[0]);
                            float _shfl_128 = __shfl_sync(0xFFFFFFFF, send_46_1, lane ^ 16);
                            float recv_48_1 = _shfl_128;
                            score_values_1[0] = keep_47_1 + recv_48_1;
                            float send_49_1 = ((hi_bit_45_1 != 0) ? score_values_1[1] : score_values_1[9]);
                            float keep_50_1 = ((hi_bit_45_1 != 0) ? score_values_1[9] : score_values_1[1]);
                            float _shfl_129 = __shfl_sync(0xFFFFFFFF, send_49_1, lane ^ 16);
                            float recv_51_1 = _shfl_129;
                            score_values_1[1] = keep_50_1 + recv_51_1;
                            float send_52_1 = ((hi_bit_45_1 != 0) ? score_values_1[2] : score_values_1[10]);
                            float keep_53_1 = ((hi_bit_45_1 != 0) ? score_values_1[10] : score_values_1[2]);
                            float _shfl_130 = __shfl_sync(0xFFFFFFFF, send_52_1, lane ^ 16);
                            float recv_54_1 = _shfl_130;
                            score_values_1[2] = keep_53_1 + recv_54_1;
                            float send_55_1 = ((hi_bit_45_1 != 0) ? score_values_1[3] : score_values_1[11]);
                            float keep_56_1 = ((hi_bit_45_1 != 0) ? score_values_1[11] : score_values_1[3]);
                            float _shfl_131 = __shfl_sync(0xFFFFFFFF, send_55_1, lane ^ 16);
                            float recv_57_1 = _shfl_131;
                            score_values_1[3] = keep_56_1 + recv_57_1;
                            float send_58_1 = ((hi_bit_45_1 != 0) ? score_values_1[4] : score_values_1[12]);
                            float keep_59_1 = ((hi_bit_45_1 != 0) ? score_values_1[12] : score_values_1[4]);
                            float _shfl_132 = __shfl_sync(0xFFFFFFFF, send_58_1, lane ^ 16);
                            float recv_60_1 = _shfl_132;
                            score_values_1[4] = keep_59_1 + recv_60_1;
                            float send_61_1 = ((hi_bit_45_1 != 0) ? score_values_1[5] : score_values_1[13]);
                            float keep_62_1 = ((hi_bit_45_1 != 0) ? score_values_1[13] : score_values_1[5]);
                            float _shfl_133 = __shfl_sync(0xFFFFFFFF, send_61_1, lane ^ 16);
                            float recv_63_1 = _shfl_133;
                            score_values_1[5] = keep_62_1 + recv_63_1;
                            float send_64_1 = ((hi_bit_45_1 != 0) ? score_values_1[6] : score_values_1[14]);
                            float keep_65_1 = ((hi_bit_45_1 != 0) ? score_values_1[14] : score_values_1[6]);
                            float _shfl_134 = __shfl_sync(0xFFFFFFFF, send_64_1, lane ^ 16);
                            float recv_66_1 = _shfl_134;
                            score_values_1[6] = keep_65_1 + recv_66_1;
                            float send_67_1 = ((hi_bit_45_1 != 0) ? score_values_1[7] : score_values_1[15]);
                            float keep_68_1 = ((hi_bit_45_1 != 0) ? score_values_1[15] : score_values_1[7]);
                            float _shfl_135 = __shfl_sync(0xFFFFFFFF, send_67_1, lane ^ 16);
                            float recv_69_1 = _shfl_135;
                            score_values_1[7] = keep_68_1 + recv_69_1;
                            int hi_bit_70_1 = lane & 8;
                            float send_71_1 = ((hi_bit_70_1 != 0) ? score_values_1[0] : score_values_1[4]);
                            float keep_72_1 = ((hi_bit_70_1 != 0) ? score_values_1[4] : score_values_1[0]);
                            float _shfl_136 = __shfl_sync(0xFFFFFFFF, send_71_1, lane ^ 8);
                            float recv_73_1 = _shfl_136;
                            score_values_1[0] = keep_72_1 + recv_73_1;
                            float send_74_1 = ((hi_bit_70_1 != 0) ? score_values_1[1] : score_values_1[5]);
                            float keep_75_1 = ((hi_bit_70_1 != 0) ? score_values_1[5] : score_values_1[1]);
                            float _shfl_137 = __shfl_sync(0xFFFFFFFF, send_74_1, lane ^ 8);
                            float recv_76_1 = _shfl_137;
                            score_values_1[1] = keep_75_1 + recv_76_1;
                            float send_77_1 = ((hi_bit_70_1 != 0) ? score_values_1[2] : score_values_1[6]);
                            float keep_78_1 = ((hi_bit_70_1 != 0) ? score_values_1[6] : score_values_1[2]);
                            float _shfl_138 = __shfl_sync(0xFFFFFFFF, send_77_1, lane ^ 8);
                            float recv_79_1 = _shfl_138;
                            score_values_1[2] = keep_78_1 + recv_79_1;
                            float send_80_1 = ((hi_bit_70_1 != 0) ? score_values_1[3] : score_values_1[7]);
                            float keep_81_1 = ((hi_bit_70_1 != 0) ? score_values_1[7] : score_values_1[3]);
                            float _shfl_139 = __shfl_sync(0xFFFFFFFF, send_80_1, lane ^ 8);
                            float recv_82_1 = _shfl_139;
                            score_values_1[3] = keep_81_1 + recv_82_1;
                            int hi_bit_83_1 = lane & 4;
                            float send_84_1 = ((hi_bit_83_1 != 0) ? score_values_1[0] : score_values_1[2]);
                            float keep_85_1 = ((hi_bit_83_1 != 0) ? score_values_1[2] : score_values_1[0]);
                            float _shfl_140 = __shfl_sync(0xFFFFFFFF, send_84_1, lane ^ 4);
                            float recv_86_1 = _shfl_140;
                            score_values_1[0] = keep_85_1 + recv_86_1;
                            float send_87_1 = ((hi_bit_83_1 != 0) ? score_values_1[1] : score_values_1[3]);
                            float keep_88_1 = ((hi_bit_83_1 != 0) ? score_values_1[3] : score_values_1[1]);
                            float _shfl_141 = __shfl_sync(0xFFFFFFFF, send_87_1, lane ^ 4);
                            float recv_89_1 = _shfl_141;
                            score_values_1[1] = keep_88_1 + recv_89_1;
                            int hi_bit_90_1 = lane & 2;
                            float send_91_1 = ((hi_bit_90_1 != 0) ? score_values_1[0] : score_values_1[1]);
                            float keep_92_1 = ((hi_bit_90_1 != 0) ? score_values_1[1] : score_values_1[0]);
                            float _shfl_142 = __shfl_sync(0xFFFFFFFF, send_91_1, lane ^ 2);
                            float recv_93_1 = _shfl_142;
                            score_values_1[0] = keep_92_1 + recv_93_1;
                            float _shfl_143 = __shfl_sync(0xFFFFFFFF, score_values_1[0], lane ^ 1);
                            float other_94_1 = _shfl_143;
                            score_values_1[0] = score_values_1[0] + other_94_1;
                            int hi_bit_95_1 = lane & 16;
                            float send_96_1 = ((hi_bit_95_1 != 0) ? tr_vals_1[0] : tr_vals_1[8]);
                            float keep_97_1 = ((hi_bit_95_1 != 0) ? tr_vals_1[8] : tr_vals_1[0]);
                            float _shfl_144 = __shfl_sync(0xFFFFFFFF, send_96_1, lane ^ 16);
                            float recv_98_1 = _shfl_144;
                            tr_vals_1[0] = keep_97_1 + recv_98_1;
                            float send_99_1 = ((hi_bit_95_1 != 0) ? tr_vals_1[1] : tr_vals_1[9]);
                            float keep_100_1 = ((hi_bit_95_1 != 0) ? tr_vals_1[9] : tr_vals_1[1]);
                            float _shfl_145 = __shfl_sync(0xFFFFFFFF, send_99_1, lane ^ 16);
                            float recv_101_1 = _shfl_145;
                            tr_vals_1[1] = keep_100_1 + recv_101_1;
                            float send_102_1 = ((hi_bit_95_1 != 0) ? tr_vals_1[2] : tr_vals_1[10]);
                            float keep_103_1 = ((hi_bit_95_1 != 0) ? tr_vals_1[10] : tr_vals_1[2]);
                            float _shfl_146 = __shfl_sync(0xFFFFFFFF, send_102_1, lane ^ 16);
                            float recv_104_1 = _shfl_146;
                            tr_vals_1[2] = keep_103_1 + recv_104_1;
                            float send_105_1 = ((hi_bit_95_1 != 0) ? tr_vals_1[3] : tr_vals_1[11]);
                            float keep_106_1 = ((hi_bit_95_1 != 0) ? tr_vals_1[11] : tr_vals_1[3]);
                            float _shfl_147 = __shfl_sync(0xFFFFFFFF, send_105_1, lane ^ 16);
                            float recv_107_1 = _shfl_147;
                            tr_vals_1[3] = keep_106_1 + recv_107_1;
                            float send_108_1 = ((hi_bit_95_1 != 0) ? tr_vals_1[4] : tr_vals_1[12]);
                            float keep_109_1 = ((hi_bit_95_1 != 0) ? tr_vals_1[12] : tr_vals_1[4]);
                            float _shfl_148 = __shfl_sync(0xFFFFFFFF, send_108_1, lane ^ 16);
                            float recv_110_1 = _shfl_148;
                            tr_vals_1[4] = keep_109_1 + recv_110_1;
                            float send_111_1 = ((hi_bit_95_1 != 0) ? tr_vals_1[5] : tr_vals_1[13]);
                            float keep_112_1 = ((hi_bit_95_1 != 0) ? tr_vals_1[13] : tr_vals_1[5]);
                            float _shfl_149 = __shfl_sync(0xFFFFFFFF, send_111_1, lane ^ 16);
                            float recv_113_1 = _shfl_149;
                            tr_vals_1[5] = keep_112_1 + recv_113_1;
                            float send_114_1 = ((hi_bit_95_1 != 0) ? tr_vals_1[6] : tr_vals_1[14]);
                            float keep_115_1 = ((hi_bit_95_1 != 0) ? tr_vals_1[14] : tr_vals_1[6]);
                            float _shfl_150 = __shfl_sync(0xFFFFFFFF, send_114_1, lane ^ 16);
                            float recv_116_1 = _shfl_150;
                            tr_vals_1[6] = keep_115_1 + recv_116_1;
                            float send_117_1 = ((hi_bit_95_1 != 0) ? tr_vals_1[7] : tr_vals_1[15]);
                            float keep_118_1 = ((hi_bit_95_1 != 0) ? tr_vals_1[15] : tr_vals_1[7]);
                            float _shfl_151 = __shfl_sync(0xFFFFFFFF, send_117_1, lane ^ 16);
                            float recv_119_1 = _shfl_151;
                            tr_vals_1[7] = keep_118_1 + recv_119_1;
                            int hi_bit_120_1 = lane & 8;
                            float send_121_1 = ((hi_bit_120_1 != 0) ? tr_vals_1[0] : tr_vals_1[4]);
                            float keep_122_1 = ((hi_bit_120_1 != 0) ? tr_vals_1[4] : tr_vals_1[0]);
                            float _shfl_152 = __shfl_sync(0xFFFFFFFF, send_121_1, lane ^ 8);
                            float recv_123_1 = _shfl_152;
                            tr_vals_1[0] = keep_122_1 + recv_123_1;
                            float send_124_1 = ((hi_bit_120_1 != 0) ? tr_vals_1[1] : tr_vals_1[5]);
                            float keep_125_1 = ((hi_bit_120_1 != 0) ? tr_vals_1[5] : tr_vals_1[1]);
                            float _shfl_153 = __shfl_sync(0xFFFFFFFF, send_124_1, lane ^ 8);
                            float recv_126_1 = _shfl_153;
                            tr_vals_1[1] = keep_125_1 + recv_126_1;
                            float send_127_1 = ((hi_bit_120_1 != 0) ? tr_vals_1[2] : tr_vals_1[6]);
                            float keep_128_1 = ((hi_bit_120_1 != 0) ? tr_vals_1[6] : tr_vals_1[2]);
                            float _shfl_154 = __shfl_sync(0xFFFFFFFF, send_127_1, lane ^ 8);
                            float recv_129_1 = _shfl_154;
                            tr_vals_1[2] = keep_128_1 + recv_129_1;
                            float send_130_1 = ((hi_bit_120_1 != 0) ? tr_vals_1[3] : tr_vals_1[7]);
                            float keep_131_1 = ((hi_bit_120_1 != 0) ? tr_vals_1[7] : tr_vals_1[3]);
                            float _shfl_155 = __shfl_sync(0xFFFFFFFF, send_130_1, lane ^ 8);
                            float recv_132_1 = _shfl_155;
                            tr_vals_1[3] = keep_131_1 + recv_132_1;
                            int hi_bit_133_1 = lane & 4;
                            float send_134_1 = ((hi_bit_133_1 != 0) ? tr_vals_1[0] : tr_vals_1[2]);
                            float keep_135_1 = ((hi_bit_133_1 != 0) ? tr_vals_1[2] : tr_vals_1[0]);
                            float _shfl_156 = __shfl_sync(0xFFFFFFFF, send_134_1, lane ^ 4);
                            float recv_136_1 = _shfl_156;
                            tr_vals_1[0] = keep_135_1 + recv_136_1;
                            float send_137_1 = ((hi_bit_133_1 != 0) ? tr_vals_1[1] : tr_vals_1[3]);
                            float keep_138_1 = ((hi_bit_133_1 != 0) ? tr_vals_1[3] : tr_vals_1[1]);
                            float _shfl_157 = __shfl_sync(0xFFFFFFFF, send_137_1, lane ^ 4);
                            float recv_139_1 = _shfl_157;
                            tr_vals_1[1] = keep_138_1 + recv_139_1;
                            int hi_bit_140_1 = lane & 2;
                            float send_141_1 = ((hi_bit_140_1 != 0) ? tr_vals_1[0] : tr_vals_1[1]);
                            float keep_142_1 = ((hi_bit_140_1 != 0) ? tr_vals_1[1] : tr_vals_1[0]);
                            float _shfl_158 = __shfl_sync(0xFFFFFFFF, send_141_1, lane ^ 2);
                            float recv_143_1 = _shfl_158;
                            tr_vals_1[0] = keep_142_1 + recv_143_1;
                            float _shfl_159 = __shfl_sync(0xFFFFFFFF, tr_vals_1[0], lane ^ 1);
                            float other_144_1 = _shfl_159;
                            tr_vals_1[0] = tr_vals_1[0] + other_144_1;
                        }
                        if ((lane & ((-1 * TILE_N) / 8 + 5)) == 0) {
                            smem_psum[local_warp * (TILE_N / 2) + (lane >> ((-1 * TILE_N) / 16 + 3))] = score_values_1[0];
                            smem_rsum[local_warp * (TILE_N / 2) + (lane >> ((-1 * TILE_N) / 16 + 3))] = tr_vals_1[0];
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(p_full_addr);
                        asm volatile("barrier.sync 11, 128;" ::: "memory");
                        if (lane < (TILE_N / 2)) {
                            float tile_sum_1 = smem_psum[lane] + smem_psum[(TILE_N / 2) + lane] + smem_psum[TILE_N + lane] + smem_psum[((3 * TILE_N) / 2) + lane];
                            float tile_rsum_1 = smem_rsum[lane] + smem_rsum[(TILE_N / 2) + lane] + smem_rsum[TILE_N + lane] + smem_rsum[((3 * TILE_N) / 2) + lane];
                            if (it_0 == 0) {
                                float sink_term_1;
                                if constexpr (TILE_N == 16) {
                                    float _exp2_19 = approx_exp2(sink_lane - m_scaled_1);
                                    sink_term_1 = _exp2_19;
                                } else {
                                    float _exp2_35 = approx_exp2(sink_lane - m_scaled_1);
                                    sink_term_1 = _exp2_35;
                                }
                                l_run = sink_term_1;
                                r_run = sink_term_1;
                            }
                            l_run = l_run + tile_sum_1;
                            r_run = r_run + tile_rsum_1;
                        }
                        asm volatile("barrier.sync 11, 128;" ::: "memory");
                    }
                }
            }
            int last_par = tiles_per_split - 1 & 1;
            {
                float norm_lane = 0.0f;
                if (lane < (TILE_N / 2)) {
                    if (r_run > 0.0f) {
                        float _rcp_0 = approx_rcp(r_run);
                        norm_lane = _rcp_0 * output_scale;
                    }
                    if (local_warp == 0) {
                        if (o_chunk == 0 && head_base + lane < num_heads) {
                            int lse_offset = (query_idx * num_heads + head_base + lane) * num_splits + split_idx;
                            float _log2_0;
                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(l_run));
                            partial_lse[lse_offset] = ((l_run > 0.0f) ? (m_run + _log2_0) * lse_partial_scale : -CAKE_INF);
                        }
                    }
                }
                float norm_c[(TILE_N / 2)];
                if constexpr (TILE_N == 16) {
                    float _shfl_86 = __shfl_sync(0xFFFFFFFF, norm_lane, 0);
                    norm_c[0] = _shfl_86;
                    float _shfl_87 = __shfl_sync(0xFFFFFFFF, norm_lane, 1);
                    norm_c[1] = _shfl_87;
                    float _shfl_88 = __shfl_sync(0xFFFFFFFF, norm_lane, 2);
                    norm_c[2] = _shfl_88;
                    float _shfl_89 = __shfl_sync(0xFFFFFFFF, norm_lane, 3);
                    norm_c[3] = _shfl_89;
                    float _shfl_90 = __shfl_sync(0xFFFFFFFF, norm_lane, 4);
                    norm_c[4] = _shfl_90;
                    float _shfl_91 = __shfl_sync(0xFFFFFFFF, norm_lane, 5);
                    norm_c[5] = _shfl_91;
                    float _shfl_92 = __shfl_sync(0xFFFFFFFF, norm_lane, 6);
                    norm_c[6] = _shfl_92;
                    float _shfl_93 = __shfl_sync(0xFFFFFFFF, norm_lane, 7);
                    norm_c[7] = _shfl_93;
                } else {
                    float _shfl_160 = __shfl_sync(0xFFFFFFFF, norm_lane, 0);
                    norm_c[0] = _shfl_160;
                    float _shfl_161 = __shfl_sync(0xFFFFFFFF, norm_lane, 1);
                    norm_c[1] = _shfl_161;
                    float _shfl_162 = __shfl_sync(0xFFFFFFFF, norm_lane, 2);
                    norm_c[2] = _shfl_162;
                    float _shfl_163 = __shfl_sync(0xFFFFFFFF, norm_lane, 3);
                    norm_c[3] = _shfl_163;
                    float _shfl_164 = __shfl_sync(0xFFFFFFFF, norm_lane, 4);
                    norm_c[4] = _shfl_164;
                    float _shfl_165 = __shfl_sync(0xFFFFFFFF, norm_lane, 5);
                    norm_c[5] = _shfl_165;
                    float _shfl_166 = __shfl_sync(0xFFFFFFFF, norm_lane, 6);
                    norm_c[6] = _shfl_166;
                    float _shfl_167 = __shfl_sync(0xFFFFFFFF, norm_lane, 7);
                    norm_c[7] = _shfl_167;
                    float _shfl_168 = __shfl_sync(0xFFFFFFFF, norm_lane, 8);
                    norm_c[8] = _shfl_168;
                    float _shfl_169 = __shfl_sync(0xFFFFFFFF, norm_lane, 9);
                    norm_c[9] = _shfl_169;
                    float _shfl_170 = __shfl_sync(0xFFFFFFFF, norm_lane, 10);
                    norm_c[10] = _shfl_170;
                    float _shfl_171 = __shfl_sync(0xFFFFFFFF, norm_lane, 11);
                    norm_c[11] = _shfl_171;
                    float _shfl_172 = __shfl_sync(0xFFFFFFFF, norm_lane, 12);
                    norm_c[12] = _shfl_172;
                    float _shfl_173 = __shfl_sync(0xFFFFFFFF, norm_lane, 13);
                    norm_c[13] = _shfl_173;
                    float _shfl_174 = __shfl_sync(0xFFFFFFFF, norm_lane, 14);
                    norm_c[14] = _shfl_174;
                    float _shfl_175 = __shfl_sync(0xFFFFFFFF, norm_lane, 15);
                    norm_c[15] = _shfl_175;
                }
                float o_values[(TILE_N / 2)];
                int dim = 0;
                long long out_off = 0;
                mbarrier_wait_hint(o_full_addr, last_par, 10000000);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if constexpr (TILE_N == 16) {
                    tmem_ld_x8(&o_values[0], taddr + 16 + (unsigned int)(tmem_row_origin << 16));
                } else {
                    tmem_ld_x16(&o_values[0], taddr + 32 + (unsigned int)(tmem_row_origin << 16));
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                dim = o_chunk * 4 * 128 + row;
                if (head_base < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[0] * norm_c[0];
                }
                if (head_base + 1 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 1) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[1] * norm_c[1];
                }
                if (head_base + 2 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 2) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[2] * norm_c[2];
                }
                if (head_base + 3 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 3) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[3] * norm_c[3];
                }
                if (head_base + 4 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 4) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[4] * norm_c[4];
                }
                if (head_base + 5 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 5) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[5] * norm_c[5];
                }
                if (head_base + 6 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 6) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[6] * norm_c[6];
                }
                if (head_base + 7 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 7) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[7] * norm_c[7];
                }
                if constexpr (TILE_N == 32) {
                    if (head_base + 8 < num_heads) {
                        out_off = ((long long)(query_idx * num_heads + head_base + 8) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                        partial_O[out_off] = o_values[8] * norm_c[8];
                    }
                    if (head_base + 9 < num_heads) {
                        out_off = ((long long)(query_idx * num_heads + head_base + 9) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                        partial_O[out_off] = o_values[9] * norm_c[9];
                    }
                    if (head_base + 10 < num_heads) {
                        out_off = ((long long)(query_idx * num_heads + head_base + 10) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                        partial_O[out_off] = o_values[10] * norm_c[10];
                    }
                    if (head_base + 11 < num_heads) {
                        out_off = ((long long)(query_idx * num_heads + head_base + 11) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                        partial_O[out_off] = o_values[11] * norm_c[11];
                    }
                    if (head_base + 12 < num_heads) {
                        out_off = ((long long)(query_idx * num_heads + head_base + 12) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                        partial_O[out_off] = o_values[12] * norm_c[12];
                    }
                    if (head_base + 13 < num_heads) {
                        out_off = ((long long)(query_idx * num_heads + head_base + 13) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                        partial_O[out_off] = o_values[13] * norm_c[13];
                    }
                    if (head_base + 14 < num_heads) {
                        out_off = ((long long)(query_idx * num_heads + head_base + 14) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                        partial_O[out_off] = o_values[14] * norm_c[14];
                    }
                    if (head_base + 15 < num_heads) {
                        out_off = ((long long)(query_idx * num_heads + head_base + 15) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                        partial_O[out_off] = o_values[15] * norm_c[15];
                    }
                }
                mbarrier_wait_hint(o_full_addr + 8, last_par, 10000000);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if constexpr (TILE_N == 16) {
                    tmem_ld_x8(&o_values[0], taddr + 32 + (unsigned int)(tmem_row_origin << 16));
                } else {
                    tmem_ld_x16(&o_values[0], taddr + 64 + (unsigned int)(tmem_row_origin << 16));
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                dim = (o_chunk * 4 + 1) * 128 + row;
                if (head_base < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[0] * norm_c[0];
                }
                if (head_base + 1 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 1) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[1] * norm_c[1];
                }
                if (head_base + 2 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 2) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[2] * norm_c[2];
                }
                if (head_base + 3 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 3) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[3] * norm_c[3];
                }
                if (head_base + 4 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 4) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[4] * norm_c[4];
                }
                if (head_base + 5 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 5) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[5] * norm_c[5];
                }
                if (head_base + 6 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 6) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[6] * norm_c[6];
                }
                if (head_base + 7 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 7) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[7] * norm_c[7];
                }
                if constexpr (TILE_N == 32) {
                    if (head_base + 8 < num_heads) {
                        out_off = ((long long)(query_idx * num_heads + head_base + 8) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                        partial_O[out_off] = o_values[8] * norm_c[8];
                    }
                    if (head_base + 9 < num_heads) {
                        out_off = ((long long)(query_idx * num_heads + head_base + 9) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                        partial_O[out_off] = o_values[9] * norm_c[9];
                    }
                    if (head_base + 10 < num_heads) {
                        out_off = ((long long)(query_idx * num_heads + head_base + 10) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                        partial_O[out_off] = o_values[10] * norm_c[10];
                    }
                    if (head_base + 11 < num_heads) {
                        out_off = ((long long)(query_idx * num_heads + head_base + 11) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                        partial_O[out_off] = o_values[11] * norm_c[11];
                    }
                    if (head_base + 12 < num_heads) {
                        out_off = ((long long)(query_idx * num_heads + head_base + 12) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                        partial_O[out_off] = o_values[12] * norm_c[12];
                    }
                    if (head_base + 13 < num_heads) {
                        out_off = ((long long)(query_idx * num_heads + head_base + 13) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                        partial_O[out_off] = o_values[13] * norm_c[13];
                    }
                    if (head_base + 14 < num_heads) {
                        out_off = ((long long)(query_idx * num_heads + head_base + 14) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                        partial_O[out_off] = o_values[14] * norm_c[14];
                    }
                    if (head_base + 15 < num_heads) {
                        out_off = ((long long)(query_idx * num_heads + head_base + 15) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                        partial_O[out_off] = o_values[15] * norm_c[15];
                    }
                }
                mbarrier_wait_hint(o_full_addr + 16, last_par, 10000000);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if constexpr (TILE_N == 16) {
                    tmem_ld_x8(&o_values[0], taddr + 48 + (unsigned int)(tmem_row_origin << 16));
                } else {
                    tmem_ld_x16(&o_values[0], taddr + 96 + (unsigned int)(tmem_row_origin << 16));
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                dim = (o_chunk * 4 + 2) * 128 + row;
                if (head_base < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[0] * norm_c[0];
                }
                if (head_base + 1 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 1) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[1] * norm_c[1];
                }
                if (head_base + 2 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 2) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[2] * norm_c[2];
                }
                if (head_base + 3 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 3) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[3] * norm_c[3];
                }
                if (head_base + 4 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 4) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[4] * norm_c[4];
                }
                if (head_base + 5 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 5) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[5] * norm_c[5];
                }
                if (head_base + 6 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 6) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[6] * norm_c[6];
                }
                if (head_base + 7 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 7) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[7] * norm_c[7];
                }
                if constexpr (TILE_N == 32) {
                    if (head_base + 8 < num_heads) {
                        out_off = ((long long)(query_idx * num_heads + head_base + 8) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                        partial_O[out_off] = o_values[8] * norm_c[8];
                    }
                    if (head_base + 9 < num_heads) {
                        out_off = ((long long)(query_idx * num_heads + head_base + 9) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                        partial_O[out_off] = o_values[9] * norm_c[9];
                    }
                    if (head_base + 10 < num_heads) {
                        out_off = ((long long)(query_idx * num_heads + head_base + 10) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                        partial_O[out_off] = o_values[10] * norm_c[10];
                    }
                    if (head_base + 11 < num_heads) {
                        out_off = ((long long)(query_idx * num_heads + head_base + 11) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                        partial_O[out_off] = o_values[11] * norm_c[11];
                    }
                    if (head_base + 12 < num_heads) {
                        out_off = ((long long)(query_idx * num_heads + head_base + 12) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                        partial_O[out_off] = o_values[12] * norm_c[12];
                    }
                    if (head_base + 13 < num_heads) {
                        out_off = ((long long)(query_idx * num_heads + head_base + 13) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                        partial_O[out_off] = o_values[13] * norm_c[13];
                    }
                    if (head_base + 14 < num_heads) {
                        out_off = ((long long)(query_idx * num_heads + head_base + 14) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                        partial_O[out_off] = o_values[14] * norm_c[14];
                    }
                    if (head_base + 15 < num_heads) {
                        out_off = ((long long)(query_idx * num_heads + head_base + 15) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                        partial_O[out_off] = o_values[15] * norm_c[15];
                    }
                }
                mbarrier_wait_hint(o_full_addr + 24, last_par, 10000000);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if constexpr (TILE_N == 16) {
                    tmem_ld_x8(&o_values[0], taddr + 64 + (unsigned int)(tmem_row_origin << 16));
                } else {
                    tmem_ld_x16(&o_values[0], taddr + 128 + (unsigned int)(tmem_row_origin << 16));
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                dim = (o_chunk * 4 + 3) * 128 + row;
                if (head_base < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[0] * norm_c[0];
                }
                if (head_base + 1 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 1) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[1] * norm_c[1];
                }
                if (head_base + 2 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 2) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[2] * norm_c[2];
                }
                if (head_base + 3 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 3) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[3] * norm_c[3];
                }
                if (head_base + 4 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 4) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[4] * norm_c[4];
                }
                if (head_base + 5 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 5) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[5] * norm_c[5];
                }
                if (head_base + 6 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 6) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[6] * norm_c[6];
                }
                if (head_base + 7 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 7) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[7] * norm_c[7];
                }
                if constexpr (TILE_N == 32) {
                    if (head_base + 8 < num_heads) {
                        out_off = ((long long)(query_idx * num_heads + head_base + 8) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                        partial_O[out_off] = o_values[8] * norm_c[8];
                    }
                    if (head_base + 9 < num_heads) {
                        out_off = ((long long)(query_idx * num_heads + head_base + 9) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                        partial_O[out_off] = o_values[9] * norm_c[9];
                    }
                    if (head_base + 10 < num_heads) {
                        out_off = ((long long)(query_idx * num_heads + head_base + 10) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                        partial_O[out_off] = o_values[10] * norm_c[10];
                    }
                    if (head_base + 11 < num_heads) {
                        out_off = ((long long)(query_idx * num_heads + head_base + 11) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                        partial_O[out_off] = o_values[11] * norm_c[11];
                    }
                    if (head_base + 12 < num_heads) {
                        out_off = ((long long)(query_idx * num_heads + head_base + 12) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                        partial_O[out_off] = o_values[12] * norm_c[12];
                    }
                    if (head_base + 13 < num_heads) {
                        out_off = ((long long)(query_idx * num_heads + head_base + 13) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                        partial_O[out_off] = o_values[13] * norm_c[13];
                    }
                    if (head_base + 14 < num_heads) {
                        out_off = ((long long)(query_idx * num_heads + head_base + 14) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                        partial_O[out_off] = o_values[14] * norm_c[14];
                    }
                    if (head_base + 15 < num_heads) {
                        out_off = ((long long)(query_idx * num_heads + head_base + 15) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                        partial_O[out_off] = o_values[15] * norm_c[15];
                    }
                }
                mbarrier_arrive(tmem_dealloc_addr);
            }
        }
    }
    // ---- Role: compute1 ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 144;");
        // compute1_main
        {
            const int local_warp_1 = warp - 4;
            int o_chunk_1 = 0;
            int work_idx_1 = blockIdx.x;
            int head_tile_1 = work_idx_1 % num_head_tiles;
            int split_work_1 = work_idx_1 / num_head_tiles;
            int split_idx_1 = split_work_1 % num_splits;
            int query_idx_1 = split_work_1 / num_splits;
            int head_base_1 = head_tile_1 * 128;
            const int row_1 = local_warp_1 * 32 + lane;
            const int tmem_row_origin_1 = local_warp_1 * 32;
            asm volatile("barrier.sync 9, 384;" ::: "memory");
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
            const int q_warp_1 = warp;
            for (int i_1 = 0; i_1 < 1; i_1++) {
                int unit_1 = q_warp_1 + 12 * i_1;
                if (unit_1 < 7) {
                    int q_block_1 = unit_1 / 7;
                    int kset_1 = unit_1 - q_block_1 * 7;
                    int q_row_1 = q_block_1 * 32 + lane;
                    if (head_base_1 + q_block_1 * 32 < num_heads && q_row_1 < TILE_N) {
                        int q_row_addr_1 = smem_qstage_addr + (unsigned int)(kset_1 * 128 * TILE_N) + (unsigned int)(q_row_1 * 128);
                        unsigned int sf_word_1 = 0;
                        for (int bp_1 = 0; bp_1 < 2; bp_1++) {
                            unsigned int words_1[4];
                            unsigned int qa_1[4];
                            unsigned int qb_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&qa_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_1[(0) + 3]))
                                : "r"(q_row_addr_1 + (4 * bp_1 ^ q_row_1 % 8) * 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&qb_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_2[(0) + 3]))
                                : "r"(q_row_addr_1 + (4 * bp_1 + 1 ^ q_row_1 % 8) * 16));
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
                            float _fabs_32 = fabsf(qv_1[0]);
                            float _fabs_33 = fabsf(qv_1[1]);
                            if constexpr (TILE_N == 16) {
                                float _max_58 = max_noftz(_fabs_32, _fabs_33);
                                m8_1[0] = _max_58;
                            } else {
                                float _max_72 = max_noftz(_fabs_32, _fabs_33);
                                m8_1[0] = _max_72;
                            }
                            float _fabs_34 = fabsf(qv_1[2]);
                            float _fabs_35 = fabsf(qv_1[3]);
                            if constexpr (TILE_N == 16) {
                                float _max_59 = max_noftz(_fabs_34, _fabs_35);
                                m8_1[1] = _max_59;
                            } else {
                                float _max_73 = max_noftz(_fabs_34, _fabs_35);
                                m8_1[1] = _max_73;
                            }
                            float _fabs_36 = fabsf(qv_1[4]);
                            float _fabs_37 = fabsf(qv_1[5]);
                            if constexpr (TILE_N == 16) {
                                float _max_60 = max_noftz(_fabs_36, _fabs_37);
                                m8_1[2] = _max_60;
                            } else {
                                float _max_74 = max_noftz(_fabs_36, _fabs_37);
                                m8_1[2] = _max_74;
                            }
                            float _fabs_38 = fabsf(qv_1[6]);
                            float _fabs_39 = fabsf(qv_1[7]);
                            if constexpr (TILE_N == 16) {
                                float _max_61 = max_noftz(_fabs_38, _fabs_39);
                                m8_1[3] = _max_61;
                            } else {
                                float _max_75 = max_noftz(_fabs_38, _fabs_39);
                                m8_1[3] = _max_75;
                            }
                            float _fabs_40 = fabsf(qv_1[8]);
                            float _fabs_41 = fabsf(qv_1[9]);
                            if constexpr (TILE_N == 16) {
                                float _max_62 = max_noftz(_fabs_40, _fabs_41);
                                m8_1[4] = _max_62;
                            } else {
                                float _max_76 = max_noftz(_fabs_40, _fabs_41);
                                m8_1[4] = _max_76;
                            }
                            float _fabs_42 = fabsf(qv_1[10]);
                            float _fabs_43 = fabsf(qv_1[11]);
                            if constexpr (TILE_N == 16) {
                                float _max_63 = max_noftz(_fabs_42, _fabs_43);
                                m8_1[5] = _max_63;
                            } else {
                                float _max_77 = max_noftz(_fabs_42, _fabs_43);
                                m8_1[5] = _max_77;
                            }
                            float _fabs_44 = fabsf(qv_1[12]);
                            float _fabs_45 = fabsf(qv_1[13]);
                            if constexpr (TILE_N == 16) {
                                float _max_64 = max_noftz(_fabs_44, _fabs_45);
                                m8_1[6] = _max_64;
                            } else {
                                float _max_78 = max_noftz(_fabs_44, _fabs_45);
                                m8_1[6] = _max_78;
                            }
                            float _fabs_46 = fabsf(qv_1[14]);
                            float _fabs_47 = fabsf(qv_1[15]);
                            if constexpr (TILE_N == 16) {
                                float _max_65 = max_noftz(_fabs_46, _fabs_47);
                                m8_1[7] = _max_65;
                            } else {
                                float _max_79 = max_noftz(_fabs_46, _fabs_47);
                                m8_1[7] = _max_79;
                            }
                            float m4_1[4];
                            float amax_1;
                            if constexpr (TILE_N == 16) {
                                float _max_66 = max_noftz(m8_1[0], m8_1[1]);
                                m4_1[0] = _max_66;
                                float _max_67 = max_noftz(m8_1[2], m8_1[3]);
                                m4_1[1] = _max_67;
                                float _max_68 = max_noftz(m8_1[4], m8_1[5]);
                                m4_1[2] = _max_68;
                                float _max_69 = max_noftz(m8_1[6], m8_1[7]);
                                m4_1[3] = _max_69;
                                float _max_70 = max_noftz(m4_1[0], m4_1[1]);
                                float _max_71 = max_noftz(m4_1[2], m4_1[3]);
                                float _max_72 = max_noftz(_max_70, _max_71);
                                amax_1 = _max_72;
                            } else {
                                float _max_80 = max_noftz(m8_1[0], m8_1[1]);
                                m4_1[0] = _max_80;
                                float _max_81 = max_noftz(m8_1[2], m8_1[3]);
                                m4_1[1] = _max_81;
                                float _max_82 = max_noftz(m8_1[4], m8_1[5]);
                                m4_1[2] = _max_82;
                                float _max_83 = max_noftz(m8_1[6], m8_1[7]);
                                m4_1[3] = _max_83;
                                float _max_84 = max_noftz(m4_1[0], m4_1[1]);
                                float _max_85 = max_noftz(m4_1[2], m4_1[3]);
                                float _max_86 = max_noftz(_max_84, _max_85);
                                amax_1 = _max_86;
                            }
                            float sc_1 = amax_1 * inv_six_1;
                            uint16_t _e4m3x2_f32_354;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_354) : "f"(0.0f), "f"(sc_1));
                            uint16_t sc_pair_1 = _e4m3x2_f32_354;
                            unsigned int sc_byte_1 = (unsigned int)sc_pair_1 & 255;
                            unsigned int sc_exp_1 = sc_byte_1 >> 3 & 15;
                            unsigned int sc_man_1 = sc_byte_1 & 7;
                            float inv_1 = 0.0f;
                            if (sc_exp_1 == 0) {
                                inv_1 = __uint_as_float(smem_rcptab[8 + sc_man_1]) * 512.0f;
                            } else {
                                inv_1 = __uint_as_float(smem_rcptab[sc_man_1]) * __uint_as_float(134 - sc_exp_1 << 23);
                            }
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_0 = {inv_1, inv_1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_1)[_ls], _scale2_0);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++) {
                                qv_1[_ls] = qv_1[_ls] * inv_1;
                            }
                            #endif
                            uint32_t _fp4_pair_16;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_16) : "f"(qv_1[0]), "f"(qv_1[1]));
                            uint32_t _fp4_pair_17;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_17) : "f"(qv_1[2]), "f"(qv_1[3]));
                            uint32_t _fp4_pair_18;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_18) : "f"(qv_1[4]), "f"(qv_1[5]));
                            uint32_t _fp4_pair_19;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_19) : "f"(qv_1[6]), "f"(qv_1[7]));
                            uint32_t _fp4_pair_20;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_20) : "f"(qv_1[8]), "f"(qv_1[9]));
                            uint32_t _fp4_pair_21;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_21) : "f"(qv_1[10]), "f"(qv_1[11]));
                            uint32_t _fp4_pair_22;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_22) : "f"(qv_1[12]), "f"(qv_1[13]));
                            uint32_t _fp4_pair_23;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_23) : "f"(qv_1[14]), "f"(qv_1[15]));
                            words_1[0] = _fp4_pair_16 | _fp4_pair_17 << 8 | _fp4_pair_18 << 16 | _fp4_pair_19 << 24;
                            words_1[1] = _fp4_pair_20 | _fp4_pair_21 << 8 | _fp4_pair_22 << 16 | _fp4_pair_23 << 24;
                            sf_word_1 = sf_word_1 | sc_byte_1 << (unsigned int)(8 * (2 * bp_1));
                            unsigned int qa_0_1[4];
                            unsigned int qb_1_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&qa_0_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0_1[(0) + 3]))
                                : "r"(q_row_addr_1 + (4 * bp_1 + 2 ^ q_row_1 % 8) * 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&qb_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1_1[(0) + 3]))
                                : "r"(q_row_addr_1 + (4 * bp_1 + 2 + 1 ^ q_row_1 % 8) * 16));
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
                            float _fabs_48 = fabsf(qv_2_1[0]);
                            float _fabs_49 = fabsf(qv_2_1[1]);
                            if constexpr (TILE_N == 16) {
                                float _max_73 = max_noftz(_fabs_48, _fabs_49);
                                m8_3_1[0] = _max_73;
                            } else {
                                float _max_87 = max_noftz(_fabs_48, _fabs_49);
                                m8_3_1[0] = _max_87;
                            }
                            float _fabs_50 = fabsf(qv_2_1[2]);
                            float _fabs_51 = fabsf(qv_2_1[3]);
                            if constexpr (TILE_N == 16) {
                                float _max_74 = max_noftz(_fabs_50, _fabs_51);
                                m8_3_1[1] = _max_74;
                            } else {
                                float _max_88 = max_noftz(_fabs_50, _fabs_51);
                                m8_3_1[1] = _max_88;
                            }
                            float _fabs_52 = fabsf(qv_2_1[4]);
                            float _fabs_53 = fabsf(qv_2_1[5]);
                            if constexpr (TILE_N == 16) {
                                float _max_75 = max_noftz(_fabs_52, _fabs_53);
                                m8_3_1[2] = _max_75;
                            } else {
                                float _max_89 = max_noftz(_fabs_52, _fabs_53);
                                m8_3_1[2] = _max_89;
                            }
                            float _fabs_54 = fabsf(qv_2_1[6]);
                            float _fabs_55 = fabsf(qv_2_1[7]);
                            if constexpr (TILE_N == 16) {
                                float _max_76 = max_noftz(_fabs_54, _fabs_55);
                                m8_3_1[3] = _max_76;
                            } else {
                                float _max_90 = max_noftz(_fabs_54, _fabs_55);
                                m8_3_1[3] = _max_90;
                            }
                            float _fabs_56 = fabsf(qv_2_1[8]);
                            float _fabs_57 = fabsf(qv_2_1[9]);
                            if constexpr (TILE_N == 16) {
                                float _max_77 = max_noftz(_fabs_56, _fabs_57);
                                m8_3_1[4] = _max_77;
                            } else {
                                float _max_91 = max_noftz(_fabs_56, _fabs_57);
                                m8_3_1[4] = _max_91;
                            }
                            float _fabs_58 = fabsf(qv_2_1[10]);
                            float _fabs_59 = fabsf(qv_2_1[11]);
                            if constexpr (TILE_N == 16) {
                                float _max_78 = max_noftz(_fabs_58, _fabs_59);
                                m8_3_1[5] = _max_78;
                            } else {
                                float _max_92 = max_noftz(_fabs_58, _fabs_59);
                                m8_3_1[5] = _max_92;
                            }
                            float _fabs_60 = fabsf(qv_2_1[12]);
                            float _fabs_61 = fabsf(qv_2_1[13]);
                            if constexpr (TILE_N == 16) {
                                float _max_79 = max_noftz(_fabs_60, _fabs_61);
                                m8_3_1[6] = _max_79;
                            } else {
                                float _max_93 = max_noftz(_fabs_60, _fabs_61);
                                m8_3_1[6] = _max_93;
                            }
                            float _fabs_62 = fabsf(qv_2_1[14]);
                            float _fabs_63 = fabsf(qv_2_1[15]);
                            if constexpr (TILE_N == 16) {
                                float _max_80 = max_noftz(_fabs_62, _fabs_63);
                                m8_3_1[7] = _max_80;
                            } else {
                                float _max_94 = max_noftz(_fabs_62, _fabs_63);
                                m8_3_1[7] = _max_94;
                            }
                            float m4_4_1[4];
                            float amax_5_1;
                            if constexpr (TILE_N == 16) {
                                float _max_81 = max_noftz(m8_3_1[0], m8_3_1[1]);
                                m4_4_1[0] = _max_81;
                                float _max_82 = max_noftz(m8_3_1[2], m8_3_1[3]);
                                m4_4_1[1] = _max_82;
                                float _max_83 = max_noftz(m8_3_1[4], m8_3_1[5]);
                                m4_4_1[2] = _max_83;
                                float _max_84 = max_noftz(m8_3_1[6], m8_3_1[7]);
                                m4_4_1[3] = _max_84;
                                float _max_85 = max_noftz(m4_4_1[0], m4_4_1[1]);
                                float _max_86 = max_noftz(m4_4_1[2], m4_4_1[3]);
                                float _max_87 = max_noftz(_max_85, _max_86);
                                amax_5_1 = _max_87;
                            } else {
                                float _max_95 = max_noftz(m8_3_1[0], m8_3_1[1]);
                                m4_4_1[0] = _max_95;
                                float _max_96 = max_noftz(m8_3_1[2], m8_3_1[3]);
                                m4_4_1[1] = _max_96;
                                float _max_97 = max_noftz(m8_3_1[4], m8_3_1[5]);
                                m4_4_1[2] = _max_97;
                                float _max_98 = max_noftz(m8_3_1[6], m8_3_1[7]);
                                m4_4_1[3] = _max_98;
                                float _max_99 = max_noftz(m4_4_1[0], m4_4_1[1]);
                                float _max_100 = max_noftz(m4_4_1[2], m4_4_1[3]);
                                float _max_101 = max_noftz(_max_99, _max_100);
                                amax_5_1 = _max_101;
                            }
                            float sc_6_1 = amax_5_1 * inv_six_1;
                            uint16_t _e4m3x2_f32_355;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_355) : "f"(0.0f), "f"(sc_6_1));
                            uint16_t sc_pair_7_1 = _e4m3x2_f32_355;
                            unsigned int sc_byte_8_1 = (unsigned int)sc_pair_7_1 & 255;
                            unsigned int sc_exp_9_1 = sc_byte_8_1 >> 3 & 15;
                            unsigned int sc_man_10_1 = sc_byte_8_1 & 7;
                            float inv_11_1 = 0.0f;
                            if (sc_exp_9_1 == 0) {
                                inv_11_1 = __uint_as_float(smem_rcptab[8 + sc_man_10_1]) * 512.0f;
                            } else {
                                inv_11_1 = __uint_as_float(smem_rcptab[sc_man_10_1]) * __uint_as_float(134 - sc_exp_9_1 << 23);
                            }
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_1 = {inv_11_1, inv_11_1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_2_1)[_ls], _scale2_1);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++) {
                                qv_2_1[_ls] = qv_2_1[_ls] * inv_11_1;
                            }
                            #endif
                            uint32_t _fp4_pair_24;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_24) : "f"(qv_2_1[0]), "f"(qv_2_1[1]));
                            uint32_t _fp4_pair_25;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_25) : "f"(qv_2_1[2]), "f"(qv_2_1[3]));
                            uint32_t _fp4_pair_26;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_26) : "f"(qv_2_1[4]), "f"(qv_2_1[5]));
                            uint32_t _fp4_pair_27;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_27) : "f"(qv_2_1[6]), "f"(qv_2_1[7]));
                            uint32_t _fp4_pair_28;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_28) : "f"(qv_2_1[8]), "f"(qv_2_1[9]));
                            uint32_t _fp4_pair_29;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_29) : "f"(qv_2_1[10]), "f"(qv_2_1[11]));
                            uint32_t _fp4_pair_30;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_30) : "f"(qv_2_1[12]), "f"(qv_2_1[13]));
                            uint32_t _fp4_pair_31;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_31) : "f"(qv_2_1[14]), "f"(qv_2_1[15]));
                            words_1[2] = _fp4_pair_24 | _fp4_pair_25 << 8 | _fp4_pair_26 << 16 | _fp4_pair_27 << 24;
                            words_1[3] = _fp4_pair_28 | _fp4_pair_29 << 8 | _fp4_pair_30 << 16 | _fp4_pair_31 << 24;
                            sf_word_1 = sf_word_1 | sc_byte_8_1 << (unsigned int)(8 * (2 * bp_1 + 1));
                            int chunk_1 = 2 * kset_1 + bp_1;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(smem_qf4_addr + (unsigned int)(chunk_1 / 8 * 4096 + (q_row_1 * 128 + (chunk_1 % 8 * 16 ^ q_row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3])));
                        }
                        smem_qsf32[kset_1 / 4 * 2048 + q_row_1 % 32 / 8 * 512 + kset_1 % 4 * 128 + q_row_1 % 8 * 16 + q_row_1 / 32 % 4 * 4 >> 2] = sf_word_1;
                    } else {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_qf4_addr + (unsigned int)(2 * kset_1 / 8 * 4096 + (q_row_1 * 128 + (2 * kset_1 % 8 * 16 ^ q_row_1 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_qf4_addr + (unsigned int)((2 * kset_1 + 1) / 8 * 4096 + (q_row_1 * 128 + ((2 * kset_1 + 1) % 8 * 16 ^ q_row_1 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                        smem_qsf32[kset_1 / 4 * 2048 + q_row_1 % 32 / 8 * 512 + kset_1 % 4 * 128 + q_row_1 % 8 * 16 + q_row_1 / 32 % 4 * 4 >> 2] = 0;
                    }
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(q_ready_addr);
            asm volatile("barrier.sync 9, 384;" ::: "memory");
            mbarrier_arrive(tok_free_addr + 8);
            float softmax_scale_log2_1 = bmm1_scale[0] * 1.4426950408889634f;
            float output_scale_1 = bmm2_scale[0];
            int strip_0_1 = smem_sfs_0_addr + (unsigned int)(row_1 * 32);
            int strip_1_1 = smem_sfs_1_addr + (unsigned int)(row_1 * 32);
            float m_run_1 = -CAKE_INF;
            float l_run_1 = 0.0f;
            float r_run_1 = 0.0f;
            float sink_lane_1 = -CAKE_INF;
            if (lane < (TILE_N / 4)) {
                if (has_sinks != 0 && split_idx_1 == 0 && head_base_1 + (TILE_N / 2) + lane < num_heads) {
                    sink_lane_1 = sinks[head_base_1 + (TILE_N / 2) + lane] * 1.4426950408889634f;
                }
            }
            for (int it2_1 = 0; it2_1 < (tiles_per_split + 1) / 2; it2_1++) {
                int par2_1 = it2_1 & 1;
                int it_1 = 2 * it2_1;
                if (it_1 < tiles_per_split) {
                    mbarrier_wait_hint(tok_full_addr, it2_1 & 1, 10000000);
                    int raw_index_2 = smem_tok_0[row_1];
                    int valid_2 = 1;
                    if (raw_index_2 < 0) {
                        valid_2 = 0;
                    }
                    if (valid_2 != 0) {
                        {
                            unsigned int sfw_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&sfw_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&sfw_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sfw_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sfw_2[(0) + 3]))
                                : "r"(strip_0_1 + 16));
                            smem_ksf32[2048 + row_1 % 32 / 8 * 512 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = sfw_2[0];
                            smem_ksf32[2048 + row_1 % 32 / 8 * 512 + 128 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = sfw_2[1];
                            smem_ksf32[2048 + row_1 % 32 / 8 * 512 + 256 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = sfw_2[2];
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
                        mbarrier_wait_hint(o_full_addr, 1, 10000000);
                        mbarrier_wait_hint(o_full_addr + 8, 1, 10000000);
                        mbarrier_wait_hint(o_full_addr + 16, 1, 10000000);
                        mbarrier_wait_hint(o_full_addr + 24, 1, 10000000);
                    }
                    if (valid_2 != 0) {
                        unsigned int kraw4_2[4];
                        unsigned int sfw32_2 = 0;
                        int vblock_3 = 32 * o_chunk_1 + 11;
                        unsigned int v8_4[4];
                        {
                            {
                                int vchunk_12 = 16 * o_chunk_1 + 5;
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[(0) + 3]))
                                    : "r"(smem_kf4_0_addr + (unsigned int)(vchunk_12 / 8 * 16384) + (unsigned int)(row_1 * 128 + (vchunk_12 % 8 * 16 ^ row_1 % 8 * 16))));
                            }
                            {
                                sfw32_2 = smem_sfs32_0[row_1 * 8 + 8 * o_chunk_1 + 2];
                            }
                            unsigned int scale_22 = sfw32_2 >> 24 & 255;
                            {
                                v8_4[0] = cake_dsv4_qmul4_portable<5>(kraw4_2[2], scale_22);
                            }
                            {
                                v8_4[1] = cake_dsv4_qmul4_portable<6>(kraw4_2[2], scale_22);
                            }
                            {
                                v8_4[2] = cake_dsv4_qmul4_portable<5>(kraw4_2[3], scale_22);
                            }
                            {
                                v8_4[3] = cake_dsv4_qmul4_portable<6>(kraw4_2[3], scale_22);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(16384 + (row_1 * 128 + (48 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_4[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_4[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_4[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_4[(0) + 3])));
                        int vblock_0_2 = 32 * o_chunk_1 + 12;
                        unsigned int v8_1_2[4];
                        {
                            {
                                int vchunk_13 = 16 * o_chunk_1 + 6;
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[(0) + 3]))
                                    : "r"(smem_kf4_0_addr + (unsigned int)(vchunk_13 / 8 * 16384) + (unsigned int)(row_1 * 128 + (vchunk_13 % 8 * 16 ^ row_1 % 8 * 16))));
                            }
                            {
                                sfw32_2 = smem_sfs32_0[row_1 * 8 + 8 * o_chunk_1 + 3];
                            }
                            unsigned int scale_23 = sfw32_2 & 255;
                            {
                                v8_1_2[0] = cake_dsv4_qmul4_portable<5>(kraw4_2[0], scale_23);
                            }
                            {
                                v8_1_2[1] = cake_dsv4_qmul4_portable<6>(kraw4_2[0], scale_23);
                            }
                            {
                                v8_1_2[2] = cake_dsv4_qmul4_portable<5>(kraw4_2[1], scale_23);
                            }
                            {
                                v8_1_2[3] = cake_dsv4_qmul4_portable<6>(kraw4_2[1], scale_23);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(16384 + (row_1 * 128 + (64 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_1_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_1_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_1_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_1_2[(0) + 3])));
                        int vblock_2_2 = 32 * o_chunk_1 + 13;
                        unsigned int v8_3_2[4];
                        {
                            unsigned int scale_24 = sfw32_2 >> 8 & 255;
                            {
                                v8_3_2[0] = cake_dsv4_qmul4_portable<5>(kraw4_2[2], scale_24);
                            }
                            {
                                v8_3_2[1] = cake_dsv4_qmul4_portable<6>(kraw4_2[2], scale_24);
                            }
                            {
                                v8_3_2[2] = cake_dsv4_qmul4_portable<5>(kraw4_2[3], scale_24);
                            }
                            {
                                v8_3_2[3] = cake_dsv4_qmul4_portable<6>(kraw4_2[3], scale_24);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(16384 + (row_1 * 128 + (80 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_3_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_3_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_3_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_3_2[(0) + 3])));
                        int vblock_4_2 = 32 * o_chunk_1 + 14;
                        unsigned int v8_5_2[4];
                        {
                            {
                                int vchunk_14 = 16 * o_chunk_1 + 7;
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[(0) + 3]))
                                    : "r"(smem_kf4_0_addr + (unsigned int)(vchunk_14 / 8 * 16384) + (unsigned int)(row_1 * 128 + (vchunk_14 % 8 * 16 ^ row_1 % 8 * 16))));
                            }
                            unsigned int scale_25 = sfw32_2 >> 16 & 255;
                            {
                                v8_5_2[0] = cake_dsv4_qmul4_portable<5>(kraw4_2[0], scale_25);
                            }
                            {
                                v8_5_2[1] = cake_dsv4_qmul4_portable<6>(kraw4_2[0], scale_25);
                            }
                            {
                                v8_5_2[2] = cake_dsv4_qmul4_portable<5>(kraw4_2[1], scale_25);
                            }
                            {
                                v8_5_2[3] = cake_dsv4_qmul4_portable<6>(kraw4_2[1], scale_25);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(16384 + (row_1 * 128 + (96 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_5_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_5_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_5_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_5_2[(0) + 3])));
                        int vblock_6_2 = 32 * o_chunk_1 + 15;
                        unsigned int v8_7_2[4];
                        {
                            unsigned int scale_26 = sfw32_2 >> 24 & 255;
                            {
                                v8_7_2[0] = cake_dsv4_qmul4_portable<5>(kraw4_2[2], scale_26);
                            }
                            {
                                v8_7_2[1] = cake_dsv4_qmul4_portable<6>(kraw4_2[2], scale_26);
                            }
                            {
                                v8_7_2[2] = cake_dsv4_qmul4_portable<5>(kraw4_2[3], scale_26);
                            }
                            {
                                v8_7_2[3] = cake_dsv4_qmul4_portable<6>(kraw4_2[3], scale_26);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(16384 + (row_1 * 128 + (112 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_7_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_7_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_7_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_7_2[(0) + 3])));
                        int vblock_8_2 = 32 * o_chunk_1 + 16;
                        unsigned int v8_9_2[4];
                        {
                            {
                                int vchunk_15 = 16 * o_chunk_1 + 8;
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[(0) + 3]))
                                    : "r"(smem_kf4_0_addr + (unsigned int)(vchunk_15 / 8 * 16384) + (unsigned int)(row_1 * 128 + (vchunk_15 % 8 * 16 ^ row_1 % 8 * 16))));
                            }
                            {
                                sfw32_2 = smem_sfs32_0[row_1 * 8 + 8 * o_chunk_1 + 4];
                            }
                            unsigned int scale_27 = sfw32_2 & 255;
                            {
                                v8_9_2[0] = cake_dsv4_qmul4_portable<5>(kraw4_2[0], scale_27);
                            }
                            {
                                v8_9_2[1] = cake_dsv4_qmul4_portable<6>(kraw4_2[0], scale_27);
                            }
                            {
                                v8_9_2[2] = cake_dsv4_qmul4_portable<5>(kraw4_2[1], scale_27);
                            }
                            {
                                v8_9_2[3] = cake_dsv4_qmul4_portable<6>(kraw4_2[1], scale_27);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (0 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_9_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_9_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_9_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_9_2[(0) + 3])));
                        int vblock_10_2 = 32 * o_chunk_1 + 17;
                        unsigned int v8_11_2[4];
                        {
                            unsigned int scale_28 = sfw32_2 >> 8 & 255;
                            {
                                v8_11_2[0] = cake_dsv4_qmul4_portable<5>(kraw4_2[2], scale_28);
                            }
                            {
                                v8_11_2[1] = cake_dsv4_qmul4_portable<6>(kraw4_2[2], scale_28);
                            }
                            {
                                v8_11_2[2] = cake_dsv4_qmul4_portable<5>(kraw4_2[3], scale_28);
                            }
                            {
                                v8_11_2[3] = cake_dsv4_qmul4_portable<6>(kraw4_2[3], scale_28);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (16 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_11_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_11_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_11_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_11_2[(0) + 3])));
                        int vblock_12_2 = 32 * o_chunk_1 + 18;
                        unsigned int v8_13_2[4];
                        {
                            {
                                int vchunk_16 = 16 * o_chunk_1 + 9;
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[(0) + 3]))
                                    : "r"(smem_kf4_0_addr + (unsigned int)(vchunk_16 / 8 * 16384) + (unsigned int)(row_1 * 128 + (vchunk_16 % 8 * 16 ^ row_1 % 8 * 16))));
                            }
                            unsigned int scale_29 = sfw32_2 >> 16 & 255;
                            {
                                v8_13_2[0] = cake_dsv4_qmul4_portable<5>(kraw4_2[0], scale_29);
                            }
                            {
                                v8_13_2[1] = cake_dsv4_qmul4_portable<6>(kraw4_2[0], scale_29);
                            }
                            {
                                v8_13_2[2] = cake_dsv4_qmul4_portable<5>(kraw4_2[1], scale_29);
                            }
                            {
                                v8_13_2[3] = cake_dsv4_qmul4_portable<6>(kraw4_2[1], scale_29);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (32 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_13_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_13_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_13_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_13_2[(0) + 3])));
                        int vblock_14_2 = 32 * o_chunk_1 + 19;
                        unsigned int v8_15_2[4];
                        {
                            unsigned int scale_30 = sfw32_2 >> 24 & 255;
                            {
                                v8_15_2[0] = cake_dsv4_qmul4_portable<5>(kraw4_2[2], scale_30);
                            }
                            {
                                v8_15_2[1] = cake_dsv4_qmul4_portable<6>(kraw4_2[2], scale_30);
                            }
                            {
                                v8_15_2[2] = cake_dsv4_qmul4_portable<5>(kraw4_2[3], scale_30);
                            }
                            {
                                v8_15_2[3] = cake_dsv4_qmul4_portable<6>(kraw4_2[3], scale_30);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (48 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_15_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_15_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_15_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_15_2[(0) + 3])));
                        int vblock_16_2 = 32 * o_chunk_1 + 20;
                        unsigned int v8_17_2[4];
                        {
                            {
                                int vchunk_17 = 16 * o_chunk_1 + 10;
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_2[(0) + 3]))
                                    : "r"(smem_kf4_0_addr + (unsigned int)(vchunk_17 / 8 * 16384) + (unsigned int)(row_1 * 128 + (vchunk_17 % 8 * 16 ^ row_1 % 8 * 16))));
                            }
                            {
                                sfw32_2 = smem_sfs32_0[row_1 * 8 + 8 * o_chunk_1 + 5];
                            }
                            unsigned int scale_31 = sfw32_2 & 255;
                            {
                                v8_17_2[0] = cake_dsv4_qmul4_portable<5>(kraw4_2[0], scale_31);
                            }
                            {
                                v8_17_2[1] = cake_dsv4_qmul4_portable<6>(kraw4_2[0], scale_31);
                            }
                            {
                                v8_17_2[2] = cake_dsv4_qmul4_portable<5>(kraw4_2[1], scale_31);
                            }
                            {
                                v8_17_2[3] = cake_dsv4_qmul4_portable<6>(kraw4_2[1], scale_31);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (64 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_17_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_17_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_17_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_17_2[(0) + 3])));
                        int vblock_18_2 = 32 * o_chunk_1 + 21;
                        unsigned int v8_19_2[4];
                        {
                            unsigned int scale_32 = sfw32_2 >> 8 & 255;
                            {
                                v8_19_2[0] = cake_dsv4_qmul4_portable<5>(kraw4_2[2], scale_32);
                            }
                            {
                                v8_19_2[1] = cake_dsv4_qmul4_portable<6>(kraw4_2[2], scale_32);
                            }
                            {
                                v8_19_2[2] = cake_dsv4_qmul4_portable<5>(kraw4_2[3], scale_32);
                            }
                            {
                                v8_19_2[3] = cake_dsv4_qmul4_portable<6>(kraw4_2[3], scale_32);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (80 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_19_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_19_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_19_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_19_2[(0) + 3])));
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
                    mbarrier_wait_hint(s_full_addr, 0, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (it_1 + 2 < tiles_per_split) {
                        mbarrier_arrive(tok_free_addr);
                    }
                    {
                        float score_values_2[(TILE_N / 4)];
                        if constexpr (TILE_N == 16) {
                            tmem_ld_x4(&score_values_2[0], taddr + 8 + (unsigned int)(tmem_row_origin_1 << 16));
                        } else {
                            tmem_ld_x8(&score_values_2[0], taddr + 16 + (unsigned int)(tmem_row_origin_1 << 16));
                        }
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        if (valid_2 == 0) {
                            score_values_2[0] = -CAKE_INF;
                            score_values_2[1] = -CAKE_INF;
                            score_values_2[2] = -CAKE_INF;
                            score_values_2[3] = -CAKE_INF;
                            if constexpr (TILE_N == 32) {
                                score_values_2[4] = -CAKE_INF;
                                score_values_2[5] = -CAKE_INF;
                                score_values_2[6] = -CAKE_INF;
                                score_values_2[7] = -CAKE_INF;
                            }
                        }
                        float tr_vals_2[(TILE_N / 4)];
                        if constexpr (TILE_N == 32) {
                            tr_vals_2[0] = score_values_2[0];
                            tr_vals_2[1] = score_values_2[1];
                            tr_vals_2[2] = score_values_2[2];
                            tr_vals_2[3] = score_values_2[3];
                        }
                        tr_vals_2[(TILE_N / 4 + -4)] = score_values_2[(TILE_N / 4 + -4)];
                        tr_vals_2[(TILE_N / 4 + -3)] = score_values_2[(TILE_N / 4 + -3)];
                        tr_vals_2[(TILE_N / 4 + -2)] = score_values_2[(TILE_N / 4 + -2)];
                        tr_vals_2[(TILE_N / 4 + -1)] = score_values_2[(TILE_N / 4 + -1)];
                        int hi_bit_2 = lane & 16;
                        float send_2 = ((hi_bit_2 != 0) ? tr_vals_2[0] : tr_vals_2[(TILE_N / 8)]);
                        float keep_3 = ((hi_bit_2 != 0) ? tr_vals_2[(TILE_N / 8)] : tr_vals_2[0]);
                        if constexpr (TILE_N == 16) {
                            float _shfl_94 = __shfl_sync(0xFFFFFFFF, send_2, lane ^ 16);
                            float recv_3 = _shfl_94;
                            float _max_88 = max_noftz(keep_3, recv_3);
                            tr_vals_2[0] = _max_88;
                        } else {
                            float _shfl_176 = __shfl_sync(0xFFFFFFFF, send_2, lane ^ 16);
                            float recv_3 = _shfl_176;
                            float _max_102 = max_noftz(keep_3, recv_3);
                            tr_vals_2[0] = _max_102;
                        }
                        float send_0_2 = ((hi_bit_2 != 0) ? tr_vals_2[1] : tr_vals_2[(TILE_N / 8 + 1)]);
                        float keep_1_2 = ((hi_bit_2 != 0) ? tr_vals_2[(TILE_N / 8 + 1)] : tr_vals_2[1]);
                        if constexpr (TILE_N == 16) {
                            float _shfl_95 = __shfl_sync(0xFFFFFFFF, send_0_2, lane ^ 16);
                            float recv_2_2 = _shfl_95;
                            float _max_89 = max_noftz(keep_1_2, recv_2_2);
                            tr_vals_2[1] = _max_89;
                            int hi_bit_3 = lane & 8;
                            float send_4 = ((hi_bit_3 != 0) ? tr_vals_2[0] : tr_vals_2[1]);
                            float keep_5 = ((hi_bit_3 != 0) ? tr_vals_2[1] : tr_vals_2[0]);
                            float _shfl_96 = __shfl_sync(0xFFFFFFFF, send_4, lane ^ 8);
                            float recv_6 = _shfl_96;
                            float _max_90 = max_noftz(keep_5, recv_6);
                            tr_vals_2[0] = _max_90;
                            float _shfl_97 = __shfl_sync(0xFFFFFFFF, tr_vals_2[0], lane ^ 4);
                            float other_2 = _shfl_97;
                            float _max_91 = max_noftz(tr_vals_2[0], other_2);
                            tr_vals_2[0] = _max_91;
                            float _shfl_98 = __shfl_sync(0xFFFFFFFF, tr_vals_2[0], lane ^ 2);
                            float other_7 = _shfl_98;
                            float _max_92 = max_noftz(tr_vals_2[0], other_7);
                            tr_vals_2[0] = _max_92;
                            float _shfl_99 = __shfl_sync(0xFFFFFFFF, tr_vals_2[0], lane ^ 1);
                            float other_8 = _shfl_99;
                            float _max_93 = max_noftz(tr_vals_2[0], other_8);
                            tr_vals_2[0] = _max_93;
                        } else {
                            float _shfl_177 = __shfl_sync(0xFFFFFFFF, send_0_2, lane ^ 16);
                            float recv_2_2 = _shfl_177;
                            float _max_103 = max_noftz(keep_1_2, recv_2_2);
                            tr_vals_2[1] = _max_103;
                            float send_3_2 = ((hi_bit_2 != 0) ? tr_vals_2[2] : tr_vals_2[6]);
                            float keep_4_2 = ((hi_bit_2 != 0) ? tr_vals_2[6] : tr_vals_2[2]);
                            float _shfl_178 = __shfl_sync(0xFFFFFFFF, send_3_2, lane ^ 16);
                            float recv_5_2 = _shfl_178;
                            float _max_104 = max_noftz(keep_4_2, recv_5_2);
                            tr_vals_2[2] = _max_104;
                            float send_6_2 = ((hi_bit_2 != 0) ? tr_vals_2[3] : tr_vals_2[7]);
                            float keep_7_2 = ((hi_bit_2 != 0) ? tr_vals_2[7] : tr_vals_2[3]);
                            float _shfl_179 = __shfl_sync(0xFFFFFFFF, send_6_2, lane ^ 16);
                            float recv_8_2 = _shfl_179;
                            float _max_105 = max_noftz(keep_7_2, recv_8_2);
                            tr_vals_2[3] = _max_105;
                            int hi_bit_9 = lane & 8;
                            float send_10 = ((hi_bit_9 != 0) ? tr_vals_2[0] : tr_vals_2[2]);
                            float keep_11 = ((hi_bit_9 != 0) ? tr_vals_2[2] : tr_vals_2[0]);
                            float _shfl_180 = __shfl_sync(0xFFFFFFFF, send_10, lane ^ 8);
                            float recv_12 = _shfl_180;
                            float _max_106 = max_noftz(keep_11, recv_12);
                            tr_vals_2[0] = _max_106;
                            float send_13 = ((hi_bit_9 != 0) ? tr_vals_2[1] : tr_vals_2[3]);
                            float keep_14 = ((hi_bit_9 != 0) ? tr_vals_2[3] : tr_vals_2[1]);
                            float _shfl_181 = __shfl_sync(0xFFFFFFFF, send_13, lane ^ 8);
                            float recv_15 = _shfl_181;
                            float _max_107 = max_noftz(keep_14, recv_15);
                            tr_vals_2[1] = _max_107;
                            int hi_bit_16 = lane & 4;
                            float send_17 = ((hi_bit_16 != 0) ? tr_vals_2[0] : tr_vals_2[1]);
                            float keep_18 = ((hi_bit_16 != 0) ? tr_vals_2[1] : tr_vals_2[0]);
                            float _shfl_182 = __shfl_sync(0xFFFFFFFF, send_17, lane ^ 4);
                            float recv_19 = _shfl_182;
                            float _max_108 = max_noftz(keep_18, recv_19);
                            tr_vals_2[0] = _max_108;
                            float _shfl_183 = __shfl_sync(0xFFFFFFFF, tr_vals_2[0], lane ^ 2);
                            float other_2 = _shfl_183;
                            float _max_109 = max_noftz(tr_vals_2[0], other_2);
                            tr_vals_2[0] = _max_109;
                            float _shfl_184 = __shfl_sync(0xFFFFFFFF, tr_vals_2[0], lane ^ 1);
                            float other_20 = _shfl_184;
                            float _max_110 = max_noftz(tr_vals_2[0], other_20);
                            tr_vals_2[0] = _max_110;
                        }
                        if ((lane & ((-1 * TILE_N) / 4 + 11)) == 0) {
                            smem_pmax[(4 + local_warp_1) * (TILE_N / 2) + (lane >> ((-1 * TILE_N) / 16 + 4))] = tr_vals_2[0];
                        }
                        asm volatile("barrier.sync 12, 128;" ::: "memory");
                        float m_lane_2 = -CAKE_INF;
                        float alpha_2 = 1.0f;
                        int grow_2 = 0;
                        if (lane < (TILE_N / 4)) {
                            if constexpr (TILE_N == 16) {
                                float _max_94 = max_noftz(smem_pmax[32 + lane], smem_pmax[40 + lane]);
                                float _max_95 = max_noftz(smem_pmax[48 + lane], smem_pmax[56 + lane]);
                                float _max_96 = max_noftz(_max_94, _max_95);
                                m_lane_2 = _max_96;
                            } else {
                                float _max_111 = max_noftz(smem_pmax[64 + lane], smem_pmax[80 + lane]);
                                float _max_112 = max_noftz(smem_pmax[96 + lane], smem_pmax[112 + lane]);
                                float _max_113 = max_noftz(_max_111, _max_112);
                                m_lane_2 = _max_113;
                            }
                            float cand_2 = m_lane_2 * softmax_scale_log2_1;
                            if (it_1 == 0) {
                                if constexpr (TILE_N == 16) {
                                    float _max_97 = max_noftz(cand_2, sink_lane_1);
                                    cand_2 = _max_97;
                                } else {
                                    float _max_114 = max_noftz(cand_2, sink_lane_1);
                                    cand_2 = _max_114;
                                }
                            }
                            if constexpr (TILE_N == 16) {
                                float _max_98 = max_noftz(cand_2, m_run_1);
                                cand_2 = _max_98;
                            } else {
                                float _max_115 = max_noftz(cand_2, m_run_1);
                                cand_2 = _max_115;
                            }
                            if (it_1 == 0) {
                                grow_2 = 1;
                            }
                            if (cand_2 - m_run_1 > 8.0f) {
                                grow_2 = 1;
                            }
                            if (grow_2 != 0) {
                                if constexpr (TILE_N == 16) {
                                    float _exp2_20 = approx_exp2(m_run_1 - cand_2);
                                    alpha_2 = ((m_run_1 > -CAKE_INF) ? _exp2_20 : 0.0f);
                                } else {
                                    float _exp2_36 = approx_exp2(m_run_1 - cand_2);
                                    alpha_2 = ((m_run_1 > -CAKE_INF) ? _exp2_36 : 0.0f);
                                }
                                l_run_1 = l_run_1 * alpha_2;
                                r_run_1 = r_run_1 * alpha_2;
                                m_run_1 = cand_2;
                            }
                        }
                        float m_scaled_2 = ((m_run_1 > -CAKE_INF) ? m_run_1 : 0.0f);
                        unsigned int _vote_2 = __ballot_sync(0xFFFFFFFF, grow_2 != 0);
                        unsigned int grow_bits_2 = _vote_2;
                        if (it_1 > 0) {
                            if (grow_bits_2 != 0) {
                                float alpha_c_2[(TILE_N / 4)];
                                if constexpr (TILE_N == 16) {
                                    float _shfl_100 = __shfl_sync(0xFFFFFFFF, alpha_2, 0);
                                    alpha_c_2[0] = _shfl_100;
                                    float _shfl_101 = __shfl_sync(0xFFFFFFFF, alpha_2, 1);
                                    alpha_c_2[1] = _shfl_101;
                                    float _shfl_102 = __shfl_sync(0xFFFFFFFF, alpha_2, 2);
                                    alpha_c_2[2] = _shfl_102;
                                    float _shfl_103 = __shfl_sync(0xFFFFFFFF, alpha_2, 3);
                                    alpha_c_2[3] = _shfl_103;
                                } else {
                                    float _shfl_185 = __shfl_sync(0xFFFFFFFF, alpha_2, 0);
                                    alpha_c_2[0] = _shfl_185;
                                    float _shfl_186 = __shfl_sync(0xFFFFFFFF, alpha_2, 1);
                                    alpha_c_2[1] = _shfl_186;
                                    float _shfl_187 = __shfl_sync(0xFFFFFFFF, alpha_2, 2);
                                    alpha_c_2[2] = _shfl_187;
                                    float _shfl_188 = __shfl_sync(0xFFFFFFFF, alpha_2, 3);
                                    alpha_c_2[3] = _shfl_188;
                                    float _shfl_189 = __shfl_sync(0xFFFFFFFF, alpha_2, 4);
                                    alpha_c_2[4] = _shfl_189;
                                    float _shfl_190 = __shfl_sync(0xFFFFFFFF, alpha_2, 5);
                                    alpha_c_2[5] = _shfl_190;
                                    float _shfl_191 = __shfl_sync(0xFFFFFFFF, alpha_2, 6);
                                    alpha_c_2[6] = _shfl_191;
                                    float _shfl_192 = __shfl_sync(0xFFFFFFFF, alpha_2, 7);
                                    alpha_c_2[7] = _shfl_192;
                                }
                                float ov_2[(TILE_N / 4)];
                                if constexpr (TILE_N == 16) {
                                    tmem_ld_x4(&ov_2[0], taddr + 16 + 8 + (unsigned int)(tmem_row_origin_1 << 16));
                                } else {
                                    tmem_ld_x8(&ov_2[0], taddr + 32 + 16 + (unsigned int)(tmem_row_origin_1 << 16));
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                ov_2[0] = ov_2[0] * alpha_c_2[0];
                                ov_2[1] = ov_2[1] * alpha_c_2[1];
                                ov_2[2] = ov_2[2] * alpha_c_2[2];
                                ov_2[3] = ov_2[3] * alpha_c_2[3];
                                if constexpr (TILE_N == 16) {
                                    tmem_st_x4_f32(taddr + 16 + 8 + (unsigned int)(tmem_row_origin_1 << 16), ov_2);
                                    tmem_ld_x4(&ov_2[0], taddr + 32 + 8 + (unsigned int)(tmem_row_origin_1 << 16));
                                } else {
                                    ov_2[4] = ov_2[4] * alpha_c_2[4];
                                    ov_2[5] = ov_2[5] * alpha_c_2[5];
                                    ov_2[6] = ov_2[6] * alpha_c_2[6];
                                    ov_2[7] = ov_2[7] * alpha_c_2[7];
                                    tmem_st_x8_f32(taddr + 32 + 16 + (unsigned int)(tmem_row_origin_1 << 16), ov_2);
                                    tmem_ld_x8(&ov_2[0], taddr + 64 + 16 + (unsigned int)(tmem_row_origin_1 << 16));
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                ov_2[0] = ov_2[0] * alpha_c_2[0];
                                ov_2[1] = ov_2[1] * alpha_c_2[1];
                                ov_2[2] = ov_2[2] * alpha_c_2[2];
                                ov_2[3] = ov_2[3] * alpha_c_2[3];
                                if constexpr (TILE_N == 16) {
                                    tmem_st_x4_f32(taddr + 32 + 8 + (unsigned int)(tmem_row_origin_1 << 16), ov_2);
                                    tmem_ld_x4(&ov_2[0], taddr + 48 + 8 + (unsigned int)(tmem_row_origin_1 << 16));
                                } else {
                                    ov_2[4] = ov_2[4] * alpha_c_2[4];
                                    ov_2[5] = ov_2[5] * alpha_c_2[5];
                                    ov_2[6] = ov_2[6] * alpha_c_2[6];
                                    ov_2[7] = ov_2[7] * alpha_c_2[7];
                                    tmem_st_x8_f32(taddr + 64 + 16 + (unsigned int)(tmem_row_origin_1 << 16), ov_2);
                                    tmem_ld_x8(&ov_2[0], taddr + 96 + 16 + (unsigned int)(tmem_row_origin_1 << 16));
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                ov_2[0] = ov_2[0] * alpha_c_2[0];
                                ov_2[1] = ov_2[1] * alpha_c_2[1];
                                ov_2[2] = ov_2[2] * alpha_c_2[2];
                                ov_2[3] = ov_2[3] * alpha_c_2[3];
                                if constexpr (TILE_N == 16) {
                                    tmem_st_x4_f32(taddr + 48 + 8 + (unsigned int)(tmem_row_origin_1 << 16), ov_2);
                                    tmem_ld_x4(&ov_2[0], taddr + 64 + 8 + (unsigned int)(tmem_row_origin_1 << 16));
                                } else {
                                    ov_2[4] = ov_2[4] * alpha_c_2[4];
                                    ov_2[5] = ov_2[5] * alpha_c_2[5];
                                    ov_2[6] = ov_2[6] * alpha_c_2[6];
                                    ov_2[7] = ov_2[7] * alpha_c_2[7];
                                    tmem_st_x8_f32(taddr + 96 + 16 + (unsigned int)(tmem_row_origin_1 << 16), ov_2);
                                    tmem_ld_x8(&ov_2[0], taddr + 128 + 16 + (unsigned int)(tmem_row_origin_1 << 16));
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                ov_2[0] = ov_2[0] * alpha_c_2[0];
                                ov_2[1] = ov_2[1] * alpha_c_2[1];
                                ov_2[2] = ov_2[2] * alpha_c_2[2];
                                ov_2[3] = ov_2[3] * alpha_c_2[3];
                                if constexpr (TILE_N == 16) {
                                    tmem_st_x4_f32(taddr + 64 + 8 + (unsigned int)(tmem_row_origin_1 << 16), ov_2);
                                } else {
                                    ov_2[4] = ov_2[4] * alpha_c_2[4];
                                    ov_2[5] = ov_2[5] * alpha_c_2[5];
                                    ov_2[6] = ov_2[6] * alpha_c_2[6];
                                    ov_2[7] = ov_2[7] * alpha_c_2[7];
                                    tmem_st_x8_f32(taddr + 128 + 16 + (unsigned int)(tmem_row_origin_1 << 16), ov_2);
                                }
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                        }
                        float col_max_2[(TILE_N / 4)];
                        if constexpr (TILE_N == 16) {
                            float _shfl_104 = __shfl_sync(0xFFFFFFFF, m_scaled_2, 0);
                            col_max_2[0] = _shfl_104;
                            float _shfl_105 = __shfl_sync(0xFFFFFFFF, m_scaled_2, 1);
                            col_max_2[1] = _shfl_105;
                            float _shfl_106 = __shfl_sync(0xFFFFFFFF, m_scaled_2, 2);
                            col_max_2[2] = _shfl_106;
                            float _shfl_107 = __shfl_sync(0xFFFFFFFF, m_scaled_2, 3);
                            col_max_2[3] = _shfl_107;
                            float _exp2_21 = approx_exp2(score_values_2[0] * softmax_scale_log2_1 - col_max_2[0]);
                            score_values_2[0] = _exp2_21;
                            float _exp2_22 = approx_exp2(score_values_2[1] * softmax_scale_log2_1 - col_max_2[1]);
                            score_values_2[1] = _exp2_22;
                            float _exp2_23 = approx_exp2(score_values_2[2] * softmax_scale_log2_1 - col_max_2[2]);
                            score_values_2[2] = _exp2_23;
                            float _exp2_24 = approx_exp2(score_values_2[3] * softmax_scale_log2_1 - col_max_2[3]);
                            score_values_2[3] = _exp2_24;
                        } else {
                            float _shfl_193 = __shfl_sync(0xFFFFFFFF, m_scaled_2, 0);
                            col_max_2[0] = _shfl_193;
                            float _shfl_194 = __shfl_sync(0xFFFFFFFF, m_scaled_2, 1);
                            col_max_2[1] = _shfl_194;
                            float _shfl_195 = __shfl_sync(0xFFFFFFFF, m_scaled_2, 2);
                            col_max_2[2] = _shfl_195;
                            float _shfl_196 = __shfl_sync(0xFFFFFFFF, m_scaled_2, 3);
                            col_max_2[3] = _shfl_196;
                            float _shfl_197 = __shfl_sync(0xFFFFFFFF, m_scaled_2, 4);
                            col_max_2[4] = _shfl_197;
                            float _shfl_198 = __shfl_sync(0xFFFFFFFF, m_scaled_2, 5);
                            col_max_2[5] = _shfl_198;
                            float _shfl_199 = __shfl_sync(0xFFFFFFFF, m_scaled_2, 6);
                            col_max_2[6] = _shfl_199;
                            float _shfl_200 = __shfl_sync(0xFFFFFFFF, m_scaled_2, 7);
                            col_max_2[7] = _shfl_200;
                            float _exp2_37 = approx_exp2(score_values_2[0] * softmax_scale_log2_1 - col_max_2[0]);
                            score_values_2[0] = _exp2_37;
                            float _exp2_38 = approx_exp2(score_values_2[1] * softmax_scale_log2_1 - col_max_2[1]);
                            score_values_2[1] = _exp2_38;
                            float _exp2_39 = approx_exp2(score_values_2[2] * softmax_scale_log2_1 - col_max_2[2]);
                            score_values_2[2] = _exp2_39;
                            float _exp2_40 = approx_exp2(score_values_2[3] * softmax_scale_log2_1 - col_max_2[3]);
                            score_values_2[3] = _exp2_40;
                            float _exp2_41 = approx_exp2(score_values_2[4] * softmax_scale_log2_1 - col_max_2[4]);
                            score_values_2[4] = _exp2_41;
                            float _exp2_42 = approx_exp2(score_values_2[5] * softmax_scale_log2_1 - col_max_2[5]);
                            score_values_2[5] = _exp2_42;
                            float _exp2_43 = approx_exp2(score_values_2[6] * softmax_scale_log2_1 - col_max_2[6]);
                            score_values_2[6] = _exp2_43;
                            float _exp2_44 = approx_exp2(score_values_2[7] * softmax_scale_log2_1 - col_max_2[7]);
                            score_values_2[7] = _exp2_44;
                        }
                        {
                            uint16_t _fp8_pair_46;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                : "=h"(_fp8_pair_46) : "f"(0.0f), "f"(score_values_2[0]));
                            uint32_t _byte_46 = (uint32_t)(_fp8_pair_46 & 0xFF);
                            uint32_t _addr_46 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(64 * TILE_N + row_1 ^ (64 * TILE_N + row_1 >> 7 & 7) << 4)));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_46), "r"(_byte_46) : "memory");
                        }
                        float _fp8_rt_16;
                        float _fp8_rt_32;
                        uint16_t _e4m3x2_47;
                        uint32_t _f16x2_47;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_47) : "f"(0.0f), "f"(score_values_2[0]));
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_47) : "h"(_e4m3x2_47));
                        uint16_t _fp8_h0_47 = (uint16_t)(_f16x2_47 & 0xFFFFu);
                        if constexpr (TILE_N == 16) {
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_16) : "h"(_fp8_h0_47));
                            tr_vals_2[0] = _fp8_rt_16;
                        } else {
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_32) : "h"(_fp8_h0_47));
                            tr_vals_2[0] = _fp8_rt_32;
                        }
                        {
                            uint16_t _fp8_pair_48;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                : "=h"(_fp8_pair_48) : "f"(0.0f), "f"(score_values_2[1]));
                            uint32_t _byte_48 = (uint32_t)(_fp8_pair_48 & 0xFF);
                            uint32_t _addr_48 = static_cast<uint32_t>((smem_p_addr + (unsigned int)((64 * TILE_N + 128) + row_1 ^ ((64 * TILE_N + 128) + row_1 >> 7 & 7) << 4)));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_48), "r"(_byte_48) : "memory");
                        }
                        float _fp8_rt_17;
                        float _fp8_rt_33;
                        uint16_t _e4m3x2_49;
                        uint32_t _f16x2_49;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_49) : "f"(0.0f), "f"(score_values_2[1]));
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_49) : "h"(_e4m3x2_49));
                        uint16_t _fp8_h0_49 = (uint16_t)(_f16x2_49 & 0xFFFFu);
                        if constexpr (TILE_N == 16) {
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_17) : "h"(_fp8_h0_49));
                            tr_vals_2[1] = _fp8_rt_17;
                        } else {
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_33) : "h"(_fp8_h0_49));
                            tr_vals_2[1] = _fp8_rt_33;
                        }
                        {
                            uint16_t _fp8_pair_50;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                : "=h"(_fp8_pair_50) : "f"(0.0f), "f"(score_values_2[2]));
                            uint32_t _byte_50 = (uint32_t)(_fp8_pair_50 & 0xFF);
                            uint32_t _addr_50 = static_cast<uint32_t>((smem_p_addr + (unsigned int)((64 * TILE_N + 256) + row_1 ^ ((64 * TILE_N + 256) + row_1 >> 7 & 7) << 4)));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_50), "r"(_byte_50) : "memory");
                        }
                        float _fp8_rt_18;
                        float _fp8_rt_34;
                        uint16_t _e4m3x2_51;
                        uint32_t _f16x2_51;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_51) : "f"(0.0f), "f"(score_values_2[2]));
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_51) : "h"(_e4m3x2_51));
                        uint16_t _fp8_h0_51 = (uint16_t)(_f16x2_51 & 0xFFFFu);
                        if constexpr (TILE_N == 16) {
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_18) : "h"(_fp8_h0_51));
                            tr_vals_2[2] = _fp8_rt_18;
                        } else {
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_34) : "h"(_fp8_h0_51));
                            tr_vals_2[2] = _fp8_rt_34;
                        }
                        {
                            uint16_t _fp8_pair_52;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                : "=h"(_fp8_pair_52) : "f"(0.0f), "f"(score_values_2[3]));
                            uint32_t _byte_52 = (uint32_t)(_fp8_pair_52 & 0xFF);
                            uint32_t _addr_52 = static_cast<uint32_t>((smem_p_addr + (unsigned int)((64 * TILE_N + 384) + row_1 ^ ((64 * TILE_N + 384) + row_1 >> 7 & 7) << 4)));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_52), "r"(_byte_52) : "memory");
                        }
                        float _fp8_rt_19;
                        float _fp8_rt_35;
                        uint16_t _e4m3x2_53;
                        uint32_t _f16x2_53;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_53) : "f"(0.0f), "f"(score_values_2[3]));
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_53) : "h"(_e4m3x2_53));
                        uint16_t _fp8_h0_53 = (uint16_t)(_f16x2_53 & 0xFFFFu);
                        if constexpr (TILE_N == 16) {
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_19) : "h"(_fp8_h0_53));
                            tr_vals_2[3] = _fp8_rt_19;
                            int hi_bit_9_2 = lane & 16;
                            float send_10_2 = ((hi_bit_9_2 != 0) ? score_values_2[0] : score_values_2[2]);
                            float keep_11_2 = ((hi_bit_9_2 != 0) ? score_values_2[2] : score_values_2[0]);
                            float _shfl_108 = __shfl_sync(0xFFFFFFFF, send_10_2, lane ^ 16);
                            float recv_12_2 = _shfl_108;
                            score_values_2[0] = keep_11_2 + recv_12_2;
                            float send_13_2 = ((hi_bit_9_2 != 0) ? score_values_2[1] : score_values_2[3]);
                            float keep_14_2 = ((hi_bit_9_2 != 0) ? score_values_2[3] : score_values_2[1]);
                            float _shfl_109 = __shfl_sync(0xFFFFFFFF, send_13_2, lane ^ 16);
                            float recv_15_2 = _shfl_109;
                            score_values_2[1] = keep_14_2 + recv_15_2;
                            int hi_bit_16_2 = lane & 8;
                            float send_17_2 = ((hi_bit_16_2 != 0) ? score_values_2[0] : score_values_2[1]);
                            float keep_18_2 = ((hi_bit_16_2 != 0) ? score_values_2[1] : score_values_2[0]);
                            float _shfl_110 = __shfl_sync(0xFFFFFFFF, send_17_2, lane ^ 8);
                            float recv_19_2 = _shfl_110;
                            score_values_2[0] = keep_18_2 + recv_19_2;
                            float _shfl_111 = __shfl_sync(0xFFFFFFFF, score_values_2[0], lane ^ 4);
                            float other_20_2 = _shfl_111;
                            score_values_2[0] = score_values_2[0] + other_20_2;
                            float _shfl_112 = __shfl_sync(0xFFFFFFFF, score_values_2[0], lane ^ 2);
                            float other_21 = _shfl_112;
                            score_values_2[0] = score_values_2[0] + other_21;
                            float _shfl_113 = __shfl_sync(0xFFFFFFFF, score_values_2[0], lane ^ 1);
                            float other_22 = _shfl_113;
                            score_values_2[0] = score_values_2[0] + other_22;
                            int hi_bit_23 = lane & 16;
                            float send_24 = ((hi_bit_23 != 0) ? tr_vals_2[0] : tr_vals_2[2]);
                            float keep_25 = ((hi_bit_23 != 0) ? tr_vals_2[2] : tr_vals_2[0]);
                            float _shfl_114 = __shfl_sync(0xFFFFFFFF, send_24, lane ^ 16);
                            float recv_26 = _shfl_114;
                            tr_vals_2[0] = keep_25 + recv_26;
                            float send_27 = ((hi_bit_23 != 0) ? tr_vals_2[1] : tr_vals_2[3]);
                            float keep_28 = ((hi_bit_23 != 0) ? tr_vals_2[3] : tr_vals_2[1]);
                            float _shfl_115 = __shfl_sync(0xFFFFFFFF, send_27, lane ^ 16);
                            float recv_29 = _shfl_115;
                            tr_vals_2[1] = keep_28 + recv_29;
                            int hi_bit_30 = lane & 8;
                            float send_31_2 = ((hi_bit_30 != 0) ? tr_vals_2[0] : tr_vals_2[1]);
                            float keep_32_2 = ((hi_bit_30 != 0) ? tr_vals_2[1] : tr_vals_2[0]);
                            float _shfl_116 = __shfl_sync(0xFFFFFFFF, send_31_2, lane ^ 8);
                            float recv_33_2 = _shfl_116;
                            tr_vals_2[0] = keep_32_2 + recv_33_2;
                            float _shfl_117 = __shfl_sync(0xFFFFFFFF, tr_vals_2[0], lane ^ 4);
                            float other_34 = _shfl_117;
                            tr_vals_2[0] = tr_vals_2[0] + other_34;
                            float _shfl_118 = __shfl_sync(0xFFFFFFFF, tr_vals_2[0], lane ^ 2);
                            float other_35 = _shfl_118;
                            tr_vals_2[0] = tr_vals_2[0] + other_35;
                            float _shfl_119 = __shfl_sync(0xFFFFFFFF, tr_vals_2[0], lane ^ 1);
                            float other_36 = _shfl_119;
                            tr_vals_2[0] = tr_vals_2[0] + other_36;
                        } else {
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_35) : "h"(_fp8_h0_53));
                            tr_vals_2[3] = _fp8_rt_35;
                            {
                                uint16_t _fp8_pair_54;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_54) : "f"(0.0f), "f"(score_values_2[4]));
                                uint32_t _byte_54 = (uint32_t)(_fp8_pair_54 & 0xFF);
                                uint32_t _addr_54 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(2560 + row_1 ^ (2560 + row_1 >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_54), "r"(_byte_54) : "memory");
                            }
                            float _fp8_rt_36;
                            uint16_t _e4m3x2_55;
                            uint32_t _f16x2_55;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_55) : "f"(0.0f), "f"(score_values_2[4]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_55) : "h"(_e4m3x2_55));
                            uint16_t _fp8_h0_55 = (uint16_t)(_f16x2_55 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_36) : "h"(_fp8_h0_55));
                            tr_vals_2[4] = _fp8_rt_36;
                            {
                                uint16_t _fp8_pair_56;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_56) : "f"(0.0f), "f"(score_values_2[5]));
                                uint32_t _byte_56 = (uint32_t)(_fp8_pair_56 & 0xFF);
                                uint32_t _addr_56 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(2688 + row_1 ^ (2688 + row_1 >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_56), "r"(_byte_56) : "memory");
                            }
                            float _fp8_rt_37;
                            uint16_t _e4m3x2_57;
                            uint32_t _f16x2_57;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_57) : "f"(0.0f), "f"(score_values_2[5]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_57) : "h"(_e4m3x2_57));
                            uint16_t _fp8_h0_57 = (uint16_t)(_f16x2_57 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_37) : "h"(_fp8_h0_57));
                            tr_vals_2[5] = _fp8_rt_37;
                            {
                                uint16_t _fp8_pair_58;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_58) : "f"(0.0f), "f"(score_values_2[6]));
                                uint32_t _byte_58 = (uint32_t)(_fp8_pair_58 & 0xFF);
                                uint32_t _addr_58 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(2816 + row_1 ^ (2816 + row_1 >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_58), "r"(_byte_58) : "memory");
                            }
                            float _fp8_rt_38;
                            uint16_t _e4m3x2_59;
                            uint32_t _f16x2_59;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_59) : "f"(0.0f), "f"(score_values_2[6]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_59) : "h"(_e4m3x2_59));
                            uint16_t _fp8_h0_59 = (uint16_t)(_f16x2_59 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_38) : "h"(_fp8_h0_59));
                            tr_vals_2[6] = _fp8_rt_38;
                            {
                                uint16_t _fp8_pair_60;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_60) : "f"(0.0f), "f"(score_values_2[7]));
                                uint32_t _byte_60 = (uint32_t)(_fp8_pair_60 & 0xFF);
                                uint32_t _addr_60 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(2944 + row_1 ^ (2944 + row_1 >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_60), "r"(_byte_60) : "memory");
                            }
                            float _fp8_rt_39;
                            uint16_t _e4m3x2_61;
                            uint32_t _f16x2_61;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_61) : "f"(0.0f), "f"(score_values_2[7]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_61) : "h"(_e4m3x2_61));
                            uint16_t _fp8_h0_61 = (uint16_t)(_f16x2_61 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_39) : "h"(_fp8_h0_61));
                            tr_vals_2[7] = _fp8_rt_39;
                            int hi_bit_21_2 = lane & 16;
                            float send_22_2 = ((hi_bit_21_2 != 0) ? score_values_2[0] : score_values_2[4]);
                            float keep_23_2 = ((hi_bit_21_2 != 0) ? score_values_2[4] : score_values_2[0]);
                            float _shfl_201 = __shfl_sync(0xFFFFFFFF, send_22_2, lane ^ 16);
                            float recv_24_2 = _shfl_201;
                            score_values_2[0] = keep_23_2 + recv_24_2;
                            float send_25_2 = ((hi_bit_21_2 != 0) ? score_values_2[1] : score_values_2[5]);
                            float keep_26_2 = ((hi_bit_21_2 != 0) ? score_values_2[5] : score_values_2[1]);
                            float _shfl_202 = __shfl_sync(0xFFFFFFFF, send_25_2, lane ^ 16);
                            float recv_27_2 = _shfl_202;
                            score_values_2[1] = keep_26_2 + recv_27_2;
                            float send_28_2 = ((hi_bit_21_2 != 0) ? score_values_2[2] : score_values_2[6]);
                            float keep_29_2 = ((hi_bit_21_2 != 0) ? score_values_2[6] : score_values_2[2]);
                            float _shfl_203 = __shfl_sync(0xFFFFFFFF, send_28_2, lane ^ 16);
                            float recv_30_2 = _shfl_203;
                            score_values_2[2] = keep_29_2 + recv_30_2;
                            float send_31_2 = ((hi_bit_21_2 != 0) ? score_values_2[3] : score_values_2[7]);
                            float keep_32_2 = ((hi_bit_21_2 != 0) ? score_values_2[7] : score_values_2[3]);
                            float _shfl_204 = __shfl_sync(0xFFFFFFFF, send_31_2, lane ^ 16);
                            float recv_33_2 = _shfl_204;
                            score_values_2[3] = keep_32_2 + recv_33_2;
                            int hi_bit_34_2 = lane & 8;
                            float send_35_2 = ((hi_bit_34_2 != 0) ? score_values_2[0] : score_values_2[2]);
                            float keep_36_2 = ((hi_bit_34_2 != 0) ? score_values_2[2] : score_values_2[0]);
                            float _shfl_205 = __shfl_sync(0xFFFFFFFF, send_35_2, lane ^ 8);
                            float recv_37_2 = _shfl_205;
                            score_values_2[0] = keep_36_2 + recv_37_2;
                            float send_38_2 = ((hi_bit_34_2 != 0) ? score_values_2[1] : score_values_2[3]);
                            float keep_39_2 = ((hi_bit_34_2 != 0) ? score_values_2[3] : score_values_2[1]);
                            float _shfl_206 = __shfl_sync(0xFFFFFFFF, send_38_2, lane ^ 8);
                            float recv_40_2 = _shfl_206;
                            score_values_2[1] = keep_39_2 + recv_40_2;
                            int hi_bit_41_2 = lane & 4;
                            float send_42_2 = ((hi_bit_41_2 != 0) ? score_values_2[0] : score_values_2[1]);
                            float keep_43_2 = ((hi_bit_41_2 != 0) ? score_values_2[1] : score_values_2[0]);
                            float _shfl_207 = __shfl_sync(0xFFFFFFFF, send_42_2, lane ^ 4);
                            float recv_44_2 = _shfl_207;
                            score_values_2[0] = keep_43_2 + recv_44_2;
                            float _shfl_208 = __shfl_sync(0xFFFFFFFF, score_values_2[0], lane ^ 2);
                            float other_45 = _shfl_208;
                            score_values_2[0] = score_values_2[0] + other_45;
                            float _shfl_209 = __shfl_sync(0xFFFFFFFF, score_values_2[0], lane ^ 1);
                            float other_46 = _shfl_209;
                            score_values_2[0] = score_values_2[0] + other_46;
                            int hi_bit_47 = lane & 16;
                            float send_48 = ((hi_bit_47 != 0) ? tr_vals_2[0] : tr_vals_2[4]);
                            float keep_49 = ((hi_bit_47 != 0) ? tr_vals_2[4] : tr_vals_2[0]);
                            float _shfl_210 = __shfl_sync(0xFFFFFFFF, send_48, lane ^ 16);
                            float recv_50 = _shfl_210;
                            tr_vals_2[0] = keep_49 + recv_50;
                            float send_51 = ((hi_bit_47 != 0) ? tr_vals_2[1] : tr_vals_2[5]);
                            float keep_52 = ((hi_bit_47 != 0) ? tr_vals_2[5] : tr_vals_2[1]);
                            float _shfl_211 = __shfl_sync(0xFFFFFFFF, send_51, lane ^ 16);
                            float recv_53 = _shfl_211;
                            tr_vals_2[1] = keep_52 + recv_53;
                            float send_54 = ((hi_bit_47 != 0) ? tr_vals_2[2] : tr_vals_2[6]);
                            float keep_55 = ((hi_bit_47 != 0) ? tr_vals_2[6] : tr_vals_2[2]);
                            float _shfl_212 = __shfl_sync(0xFFFFFFFF, send_54, lane ^ 16);
                            float recv_56 = _shfl_212;
                            tr_vals_2[2] = keep_55 + recv_56;
                            float send_57 = ((hi_bit_47 != 0) ? tr_vals_2[3] : tr_vals_2[7]);
                            float keep_58 = ((hi_bit_47 != 0) ? tr_vals_2[7] : tr_vals_2[3]);
                            float _shfl_213 = __shfl_sync(0xFFFFFFFF, send_57, lane ^ 16);
                            float recv_59 = _shfl_213;
                            tr_vals_2[3] = keep_58 + recv_59;
                            int hi_bit_60 = lane & 8;
                            float send_61_2 = ((hi_bit_60 != 0) ? tr_vals_2[0] : tr_vals_2[2]);
                            float keep_62_2 = ((hi_bit_60 != 0) ? tr_vals_2[2] : tr_vals_2[0]);
                            float _shfl_214 = __shfl_sync(0xFFFFFFFF, send_61_2, lane ^ 8);
                            float recv_63_2 = _shfl_214;
                            tr_vals_2[0] = keep_62_2 + recv_63_2;
                            float send_64_2 = ((hi_bit_60 != 0) ? tr_vals_2[1] : tr_vals_2[3]);
                            float keep_65_2 = ((hi_bit_60 != 0) ? tr_vals_2[3] : tr_vals_2[1]);
                            float _shfl_215 = __shfl_sync(0xFFFFFFFF, send_64_2, lane ^ 8);
                            float recv_66_2 = _shfl_215;
                            tr_vals_2[1] = keep_65_2 + recv_66_2;
                            int hi_bit_67 = lane & 4;
                            float send_68 = ((hi_bit_67 != 0) ? tr_vals_2[0] : tr_vals_2[1]);
                            float keep_69 = ((hi_bit_67 != 0) ? tr_vals_2[1] : tr_vals_2[0]);
                            float _shfl_216 = __shfl_sync(0xFFFFFFFF, send_68, lane ^ 4);
                            float recv_70 = _shfl_216;
                            tr_vals_2[0] = keep_69 + recv_70;
                            float _shfl_217 = __shfl_sync(0xFFFFFFFF, tr_vals_2[0], lane ^ 2);
                            float other_71 = _shfl_217;
                            tr_vals_2[0] = tr_vals_2[0] + other_71;
                            float _shfl_218 = __shfl_sync(0xFFFFFFFF, tr_vals_2[0], lane ^ 1);
                            float other_72 = _shfl_218;
                            tr_vals_2[0] = tr_vals_2[0] + other_72;
                        }
                        if ((lane & ((-1 * TILE_N) / 4 + 11)) == 0) {
                            smem_psum[(4 + local_warp_1) * (TILE_N / 2) + (lane >> ((-1 * TILE_N) / 16 + 4))] = score_values_2[0];
                            smem_rsum[(4 + local_warp_1) * (TILE_N / 2) + (lane >> ((-1 * TILE_N) / 16 + 4))] = tr_vals_2[0];
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(p_full_addr);
                        asm volatile("barrier.sync 12, 128;" ::: "memory");
                        if (lane < (TILE_N / 4)) {
                            float tile_sum_2 = smem_psum[2 * TILE_N + lane] + smem_psum[((5 * TILE_N) / 2) + lane] + smem_psum[3 * TILE_N + lane] + smem_psum[((7 * TILE_N) / 2) + lane];
                            float tile_rsum_2 = smem_rsum[2 * TILE_N + lane] + smem_rsum[((5 * TILE_N) / 2) + lane] + smem_rsum[3 * TILE_N + lane] + smem_rsum[((7 * TILE_N) / 2) + lane];
                            if (it_1 == 0) {
                                float sink_term_2;
                                if constexpr (TILE_N == 16) {
                                    float _exp2_25 = approx_exp2(sink_lane_1 - m_scaled_2);
                                    sink_term_2 = _exp2_25;
                                } else {
                                    float _exp2_45 = approx_exp2(sink_lane_1 - m_scaled_2);
                                    sink_term_2 = _exp2_45;
                                }
                                l_run_1 = sink_term_2;
                                r_run_1 = sink_term_2;
                            }
                            l_run_1 = l_run_1 + tile_sum_2;
                            r_run_1 = r_run_1 + tile_rsum_2;
                        }
                        asm volatile("barrier.sync 12, 128;" ::: "memory");
                    }
                }
                int it_0_1 = 2 * it2_1 + 1;
                if (it_0_1 < tiles_per_split) {
                    mbarrier_wait_hint(tok_full_addr + 8, it2_1 & 1, 10000000);
                    int raw_index_3 = smem_tok_1[row_1];
                    int valid_3 = 1;
                    if (raw_index_3 < 0) {
                        valid_3 = 0;
                    }
                    if (valid_3 != 0) {
                        {
                            unsigned int sfw_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&sfw_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&sfw_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sfw_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sfw_3[(0) + 3]))
                                : "r"(strip_1_1 + 16));
                            smem_ksf32[2048 + row_1 % 32 / 8 * 512 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = sfw_3[0];
                            smem_ksf32[2048 + row_1 % 32 / 8 * 512 + 128 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = sfw_3[1];
                            smem_ksf32[2048 + row_1 % 32 / 8 * 512 + 256 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = sfw_3[2];
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
                    if (it_0_1 > 0) {
                        mbarrier_wait_hint(o_full_addr, 0, 10000000);
                        mbarrier_wait_hint(o_full_addr + 8, 0, 10000000);
                        mbarrier_wait_hint(o_full_addr + 16, 0, 10000000);
                        mbarrier_wait_hint(o_full_addr + 24, 0, 10000000);
                    }
                    if (valid_3 != 0) {
                        unsigned int kraw4_3[4];
                        unsigned int sfw32_3 = 0;
                        int vblock_5 = 32 * o_chunk_1 + 11;
                        unsigned int v8_6[4];
                        {
                            {
                                int vchunk_18 = 16 * o_chunk_1 + 5;
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&kraw4_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_3[(0) + 3]))
                                    : "r"(smem_kf4_1_addr + (unsigned int)(vchunk_18 / 8 * 16384) + (unsigned int)(row_1 * 128 + (vchunk_18 % 8 * 16 ^ row_1 % 8 * 16))));
                            }
                            {
                                sfw32_3 = smem_sfs32_1[row_1 * 8 + 8 * o_chunk_1 + 2];
                            }
                            unsigned int scale_33 = sfw32_3 >> 24 & 255;
                            {
                                v8_6[0] = cake_dsv4_qmul4_portable<5>(kraw4_3[2], scale_33);
                            }
                            {
                                v8_6[1] = cake_dsv4_qmul4_portable<6>(kraw4_3[2], scale_33);
                            }
                            {
                                v8_6[2] = cake_dsv4_qmul4_portable<5>(kraw4_3[3], scale_33);
                            }
                            {
                                v8_6[3] = cake_dsv4_qmul4_portable<6>(kraw4_3[3], scale_33);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(16384 + (row_1 * 128 + (48 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_6[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_6[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_6[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_6[(0) + 3])));
                        int vblock_0_3 = 32 * o_chunk_1 + 12;
                        unsigned int v8_1_3[4];
                        {
                            {
                                int vchunk_19 = 16 * o_chunk_1 + 6;
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&kraw4_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_3[(0) + 3]))
                                    : "r"(smem_kf4_1_addr + (unsigned int)(vchunk_19 / 8 * 16384) + (unsigned int)(row_1 * 128 + (vchunk_19 % 8 * 16 ^ row_1 % 8 * 16))));
                            }
                            {
                                sfw32_3 = smem_sfs32_1[row_1 * 8 + 8 * o_chunk_1 + 3];
                            }
                            unsigned int scale_34 = sfw32_3 & 255;
                            {
                                v8_1_3[0] = cake_dsv4_qmul4_portable<5>(kraw4_3[0], scale_34);
                            }
                            {
                                v8_1_3[1] = cake_dsv4_qmul4_portable<6>(kraw4_3[0], scale_34);
                            }
                            {
                                v8_1_3[2] = cake_dsv4_qmul4_portable<5>(kraw4_3[1], scale_34);
                            }
                            {
                                v8_1_3[3] = cake_dsv4_qmul4_portable<6>(kraw4_3[1], scale_34);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(16384 + (row_1 * 128 + (64 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_1_3[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_1_3[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_1_3[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_1_3[(0) + 3])));
                        int vblock_2_3 = 32 * o_chunk_1 + 13;
                        unsigned int v8_3_3[4];
                        {
                            unsigned int scale_35 = sfw32_3 >> 8 & 255;
                            {
                                v8_3_3[0] = cake_dsv4_qmul4_portable<5>(kraw4_3[2], scale_35);
                            }
                            {
                                v8_3_3[1] = cake_dsv4_qmul4_portable<6>(kraw4_3[2], scale_35);
                            }
                            {
                                v8_3_3[2] = cake_dsv4_qmul4_portable<5>(kraw4_3[3], scale_35);
                            }
                            {
                                v8_3_3[3] = cake_dsv4_qmul4_portable<6>(kraw4_3[3], scale_35);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(16384 + (row_1 * 128 + (80 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_3_3[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_3_3[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_3_3[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_3_3[(0) + 3])));
                        int vblock_4_3 = 32 * o_chunk_1 + 14;
                        unsigned int v8_5_3[4];
                        {
                            {
                                int vchunk_20 = 16 * o_chunk_1 + 7;
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&kraw4_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_3[(0) + 3]))
                                    : "r"(smem_kf4_1_addr + (unsigned int)(vchunk_20 / 8 * 16384) + (unsigned int)(row_1 * 128 + (vchunk_20 % 8 * 16 ^ row_1 % 8 * 16))));
                            }
                            unsigned int scale_36 = sfw32_3 >> 16 & 255;
                            {
                                v8_5_3[0] = cake_dsv4_qmul4_portable<5>(kraw4_3[0], scale_36);
                            }
                            {
                                v8_5_3[1] = cake_dsv4_qmul4_portable<6>(kraw4_3[0], scale_36);
                            }
                            {
                                v8_5_3[2] = cake_dsv4_qmul4_portable<5>(kraw4_3[1], scale_36);
                            }
                            {
                                v8_5_3[3] = cake_dsv4_qmul4_portable<6>(kraw4_3[1], scale_36);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(16384 + (row_1 * 128 + (96 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_5_3[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_5_3[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_5_3[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_5_3[(0) + 3])));
                        int vblock_6_3 = 32 * o_chunk_1 + 15;
                        unsigned int v8_7_3[4];
                        {
                            unsigned int scale_37 = sfw32_3 >> 24 & 255;
                            {
                                v8_7_3[0] = cake_dsv4_qmul4_portable<5>(kraw4_3[2], scale_37);
                            }
                            {
                                v8_7_3[1] = cake_dsv4_qmul4_portable<6>(kraw4_3[2], scale_37);
                            }
                            {
                                v8_7_3[2] = cake_dsv4_qmul4_portable<5>(kraw4_3[3], scale_37);
                            }
                            {
                                v8_7_3[3] = cake_dsv4_qmul4_portable<6>(kraw4_3[3], scale_37);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(16384 + (row_1 * 128 + (112 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_7_3[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_7_3[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_7_3[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_7_3[(0) + 3])));
                        int vblock_8_3 = 32 * o_chunk_1 + 16;
                        unsigned int v8_9_3[4];
                        {
                            {
                                int vchunk_21 = 16 * o_chunk_1 + 8;
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&kraw4_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_3[(0) + 3]))
                                    : "r"(smem_kf4_1_addr + (unsigned int)(vchunk_21 / 8 * 16384) + (unsigned int)(row_1 * 128 + (vchunk_21 % 8 * 16 ^ row_1 % 8 * 16))));
                            }
                            {
                                sfw32_3 = smem_sfs32_1[row_1 * 8 + 8 * o_chunk_1 + 4];
                            }
                            unsigned int scale_38 = sfw32_3 & 255;
                            {
                                v8_9_3[0] = cake_dsv4_qmul4_portable<5>(kraw4_3[0], scale_38);
                            }
                            {
                                v8_9_3[1] = cake_dsv4_qmul4_portable<6>(kraw4_3[0], scale_38);
                            }
                            {
                                v8_9_3[2] = cake_dsv4_qmul4_portable<5>(kraw4_3[1], scale_38);
                            }
                            {
                                v8_9_3[3] = cake_dsv4_qmul4_portable<6>(kraw4_3[1], scale_38);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (0 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_9_3[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_9_3[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_9_3[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_9_3[(0) + 3])));
                        int vblock_10_3 = 32 * o_chunk_1 + 17;
                        unsigned int v8_11_3[4];
                        {
                            unsigned int scale_39 = sfw32_3 >> 8 & 255;
                            {
                                v8_11_3[0] = cake_dsv4_qmul4_portable<5>(kraw4_3[2], scale_39);
                            }
                            {
                                v8_11_3[1] = cake_dsv4_qmul4_portable<6>(kraw4_3[2], scale_39);
                            }
                            {
                                v8_11_3[2] = cake_dsv4_qmul4_portable<5>(kraw4_3[3], scale_39);
                            }
                            {
                                v8_11_3[3] = cake_dsv4_qmul4_portable<6>(kraw4_3[3], scale_39);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (16 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_11_3[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_11_3[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_11_3[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_11_3[(0) + 3])));
                        int vblock_12_3 = 32 * o_chunk_1 + 18;
                        unsigned int v8_13_3[4];
                        {
                            {
                                int vchunk_22 = 16 * o_chunk_1 + 9;
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&kraw4_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_3[(0) + 3]))
                                    : "r"(smem_kf4_1_addr + (unsigned int)(vchunk_22 / 8 * 16384) + (unsigned int)(row_1 * 128 + (vchunk_22 % 8 * 16 ^ row_1 % 8 * 16))));
                            }
                            unsigned int scale_40 = sfw32_3 >> 16 & 255;
                            {
                                v8_13_3[0] = cake_dsv4_qmul4_portable<5>(kraw4_3[0], scale_40);
                            }
                            {
                                v8_13_3[1] = cake_dsv4_qmul4_portable<6>(kraw4_3[0], scale_40);
                            }
                            {
                                v8_13_3[2] = cake_dsv4_qmul4_portable<5>(kraw4_3[1], scale_40);
                            }
                            {
                                v8_13_3[3] = cake_dsv4_qmul4_portable<6>(kraw4_3[1], scale_40);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (32 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_13_3[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_13_3[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_13_3[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_13_3[(0) + 3])));
                        int vblock_14_3 = 32 * o_chunk_1 + 19;
                        unsigned int v8_15_3[4];
                        {
                            unsigned int scale_41 = sfw32_3 >> 24 & 255;
                            {
                                v8_15_3[0] = cake_dsv4_qmul4_portable<5>(kraw4_3[2], scale_41);
                            }
                            {
                                v8_15_3[1] = cake_dsv4_qmul4_portable<6>(kraw4_3[2], scale_41);
                            }
                            {
                                v8_15_3[2] = cake_dsv4_qmul4_portable<5>(kraw4_3[3], scale_41);
                            }
                            {
                                v8_15_3[3] = cake_dsv4_qmul4_portable<6>(kraw4_3[3], scale_41);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (48 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_15_3[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_15_3[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_15_3[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_15_3[(0) + 3])));
                        int vblock_16_3 = 32 * o_chunk_1 + 20;
                        unsigned int v8_17_3[4];
                        {
                            {
                                int vchunk_23 = 16 * o_chunk_1 + 10;
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&kraw4_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_3[(0) + 3]))
                                    : "r"(smem_kf4_1_addr + (unsigned int)(vchunk_23 / 8 * 16384) + (unsigned int)(row_1 * 128 + (vchunk_23 % 8 * 16 ^ row_1 % 8 * 16))));
                            }
                            {
                                sfw32_3 = smem_sfs32_1[row_1 * 8 + 8 * o_chunk_1 + 5];
                            }
                            unsigned int scale_42 = sfw32_3 & 255;
                            {
                                v8_17_3[0] = cake_dsv4_qmul4_portable<5>(kraw4_3[0], scale_42);
                            }
                            {
                                v8_17_3[1] = cake_dsv4_qmul4_portable<6>(kraw4_3[0], scale_42);
                            }
                            {
                                v8_17_3[2] = cake_dsv4_qmul4_portable<5>(kraw4_3[1], scale_42);
                            }
                            {
                                v8_17_3[3] = cake_dsv4_qmul4_portable<6>(kraw4_3[1], scale_42);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (64 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_17_3[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_17_3[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_17_3[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_17_3[(0) + 3])));
                        int vblock_18_3 = 32 * o_chunk_1 + 21;
                        unsigned int v8_19_3[4];
                        {
                            unsigned int scale_43 = sfw32_3 >> 8 & 255;
                            {
                                v8_19_3[0] = cake_dsv4_qmul4_portable<5>(kraw4_3[2], scale_43);
                            }
                            {
                                v8_19_3[1] = cake_dsv4_qmul4_portable<6>(kraw4_3[2], scale_43);
                            }
                            {
                                v8_19_3[2] = cake_dsv4_qmul4_portable<5>(kraw4_3[3], scale_43);
                            }
                            {
                                v8_19_3[3] = cake_dsv4_qmul4_portable<6>(kraw4_3[3], scale_43);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (80 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_19_3[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_19_3[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_19_3[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_19_3[(0) + 3])));
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
                    mbarrier_wait_hint(s_full_addr, 1, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (it_0_1 + 2 < tiles_per_split) {
                        mbarrier_arrive(tok_free_addr + 8);
                    }
                    {
                        float score_values_3[(TILE_N / 4)];
                        if constexpr (TILE_N == 16) {
                            tmem_ld_x4(&score_values_3[0], taddr + 8 + (unsigned int)(tmem_row_origin_1 << 16));
                        } else {
                            tmem_ld_x8(&score_values_3[0], taddr + 16 + (unsigned int)(tmem_row_origin_1 << 16));
                        }
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        if (valid_3 == 0) {
                            score_values_3[0] = -CAKE_INF;
                            score_values_3[1] = -CAKE_INF;
                            score_values_3[2] = -CAKE_INF;
                            score_values_3[3] = -CAKE_INF;
                            if constexpr (TILE_N == 32) {
                                score_values_3[4] = -CAKE_INF;
                                score_values_3[5] = -CAKE_INF;
                                score_values_3[6] = -CAKE_INF;
                                score_values_3[7] = -CAKE_INF;
                            }
                        }
                        float tr_vals_3[(TILE_N / 4)];
                        tr_vals_3[0] = score_values_3[0];
                        tr_vals_3[1] = score_values_3[1];
                        tr_vals_3[2] = score_values_3[2];
                        tr_vals_3[3] = score_values_3[3];
                        if constexpr (TILE_N == 16) {
                            int hi_bit_4 = lane & 16;
                            float send_5 = ((hi_bit_4 != 0) ? tr_vals_3[0] : tr_vals_3[2]);
                            float keep_6 = ((hi_bit_4 != 0) ? tr_vals_3[2] : tr_vals_3[0]);
                            float _shfl_120 = __shfl_sync(0xFFFFFFFF, send_5, lane ^ 16);
                            float recv_4 = _shfl_120;
                            float _max_99 = max_noftz(keep_6, recv_4);
                            tr_vals_3[0] = _max_99;
                            float send_0_3 = ((hi_bit_4 != 0) ? tr_vals_3[1] : tr_vals_3[3]);
                            float keep_1_3 = ((hi_bit_4 != 0) ? tr_vals_3[3] : tr_vals_3[1]);
                            float _shfl_121 = __shfl_sync(0xFFFFFFFF, send_0_3, lane ^ 16);
                            float recv_2_3 = _shfl_121;
                            float _max_100 = max_noftz(keep_1_3, recv_2_3);
                            tr_vals_3[1] = _max_100;
                            int hi_bit_3_1 = lane & 8;
                            float send_4_1 = ((hi_bit_3_1 != 0) ? tr_vals_3[0] : tr_vals_3[1]);
                            float keep_5_1 = ((hi_bit_3_1 != 0) ? tr_vals_3[1] : tr_vals_3[0]);
                            float _shfl_122 = __shfl_sync(0xFFFFFFFF, send_4_1, lane ^ 8);
                            float recv_6_1 = _shfl_122;
                            float _max_101 = max_noftz(keep_5_1, recv_6_1);
                            tr_vals_3[0] = _max_101;
                            float _shfl_123 = __shfl_sync(0xFFFFFFFF, tr_vals_3[0], lane ^ 4);
                            float other_3 = _shfl_123;
                            float _max_102 = max_noftz(tr_vals_3[0], other_3);
                            tr_vals_3[0] = _max_102;
                            float _shfl_124 = __shfl_sync(0xFFFFFFFF, tr_vals_3[0], lane ^ 2);
                            float other_7_1 = _shfl_124;
                            float _max_103 = max_noftz(tr_vals_3[0], other_7_1);
                            tr_vals_3[0] = _max_103;
                            float _shfl_125 = __shfl_sync(0xFFFFFFFF, tr_vals_3[0], lane ^ 1);
                            float other_8_1 = _shfl_125;
                            float _max_104 = max_noftz(tr_vals_3[0], other_8_1);
                            tr_vals_3[0] = _max_104;
                        } else {
                            tr_vals_3[4] = score_values_3[4];
                            tr_vals_3[5] = score_values_3[5];
                            tr_vals_3[6] = score_values_3[6];
                            tr_vals_3[7] = score_values_3[7];
                            int hi_bit_3 = lane & 16;
                            float send_4 = ((hi_bit_3 != 0) ? tr_vals_3[0] : tr_vals_3[4]);
                            float keep_5 = ((hi_bit_3 != 0) ? tr_vals_3[4] : tr_vals_3[0]);
                            float _shfl_219 = __shfl_sync(0xFFFFFFFF, send_4, lane ^ 16);
                            float recv_4 = _shfl_219;
                            float _max_116 = max_noftz(keep_5, recv_4);
                            tr_vals_3[0] = _max_116;
                            float send_0_3 = ((hi_bit_3 != 0) ? tr_vals_3[1] : tr_vals_3[5]);
                            float keep_1_3 = ((hi_bit_3 != 0) ? tr_vals_3[5] : tr_vals_3[1]);
                            float _shfl_220 = __shfl_sync(0xFFFFFFFF, send_0_3, lane ^ 16);
                            float recv_2_3 = _shfl_220;
                            float _max_117 = max_noftz(keep_1_3, recv_2_3);
                            tr_vals_3[1] = _max_117;
                            float send_3_3 = ((hi_bit_3 != 0) ? tr_vals_3[2] : tr_vals_3[6]);
                            float keep_4_3 = ((hi_bit_3 != 0) ? tr_vals_3[6] : tr_vals_3[2]);
                            float _shfl_221 = __shfl_sync(0xFFFFFFFF, send_3_3, lane ^ 16);
                            float recv_5_3 = _shfl_221;
                            float _max_118 = max_noftz(keep_4_3, recv_5_3);
                            tr_vals_3[2] = _max_118;
                            float send_6_3 = ((hi_bit_3 != 0) ? tr_vals_3[3] : tr_vals_3[7]);
                            float keep_7_3 = ((hi_bit_3 != 0) ? tr_vals_3[7] : tr_vals_3[3]);
                            float _shfl_222 = __shfl_sync(0xFFFFFFFF, send_6_3, lane ^ 16);
                            float recv_8_3 = _shfl_222;
                            float _max_119 = max_noftz(keep_7_3, recv_8_3);
                            tr_vals_3[3] = _max_119;
                            int hi_bit_9_1 = lane & 8;
                            float send_10_1 = ((hi_bit_9_1 != 0) ? tr_vals_3[0] : tr_vals_3[2]);
                            float keep_11_1 = ((hi_bit_9_1 != 0) ? tr_vals_3[2] : tr_vals_3[0]);
                            float _shfl_223 = __shfl_sync(0xFFFFFFFF, send_10_1, lane ^ 8);
                            float recv_12_1 = _shfl_223;
                            float _max_120 = max_noftz(keep_11_1, recv_12_1);
                            tr_vals_3[0] = _max_120;
                            float send_13_1 = ((hi_bit_9_1 != 0) ? tr_vals_3[1] : tr_vals_3[3]);
                            float keep_14_1 = ((hi_bit_9_1 != 0) ? tr_vals_3[3] : tr_vals_3[1]);
                            float _shfl_224 = __shfl_sync(0xFFFFFFFF, send_13_1, lane ^ 8);
                            float recv_15_1 = _shfl_224;
                            float _max_121 = max_noftz(keep_14_1, recv_15_1);
                            tr_vals_3[1] = _max_121;
                            int hi_bit_16_1 = lane & 4;
                            float send_17_1 = ((hi_bit_16_1 != 0) ? tr_vals_3[0] : tr_vals_3[1]);
                            float keep_18_1 = ((hi_bit_16_1 != 0) ? tr_vals_3[1] : tr_vals_3[0]);
                            float _shfl_225 = __shfl_sync(0xFFFFFFFF, send_17_1, lane ^ 4);
                            float recv_19_1 = _shfl_225;
                            float _max_122 = max_noftz(keep_18_1, recv_19_1);
                            tr_vals_3[0] = _max_122;
                            float _shfl_226 = __shfl_sync(0xFFFFFFFF, tr_vals_3[0], lane ^ 2);
                            float other_3 = _shfl_226;
                            float _max_123 = max_noftz(tr_vals_3[0], other_3);
                            tr_vals_3[0] = _max_123;
                            float _shfl_227 = __shfl_sync(0xFFFFFFFF, tr_vals_3[0], lane ^ 1);
                            float other_20_1 = _shfl_227;
                            float _max_124 = max_noftz(tr_vals_3[0], other_20_1);
                            tr_vals_3[0] = _max_124;
                        }
                        if ((lane & ((-1 * TILE_N) / 4 + 11)) == 0) {
                            smem_pmax[(4 + local_warp_1) * (TILE_N / 2) + (lane >> ((-1 * TILE_N) / 16 + 4))] = tr_vals_3[0];
                        }
                        asm volatile("barrier.sync 12, 128;" ::: "memory");
                        float m_lane_3 = -CAKE_INF;
                        float alpha_3 = 1.0f;
                        int grow_3 = 0;
                        if (lane < (TILE_N / 4)) {
                            if constexpr (TILE_N == 16) {
                                float _max_105 = max_noftz(smem_pmax[32 + lane], smem_pmax[40 + lane]);
                                float _max_106 = max_noftz(smem_pmax[48 + lane], smem_pmax[56 + lane]);
                                float _max_107 = max_noftz(_max_105, _max_106);
                                m_lane_3 = _max_107;
                            } else {
                                float _max_125 = max_noftz(smem_pmax[64 + lane], smem_pmax[80 + lane]);
                                float _max_126 = max_noftz(smem_pmax[96 + lane], smem_pmax[112 + lane]);
                                float _max_127 = max_noftz(_max_125, _max_126);
                                m_lane_3 = _max_127;
                            }
                            float cand_3 = m_lane_3 * softmax_scale_log2_1;
                            if (it_0_1 == 0) {
                                if constexpr (TILE_N == 16) {
                                    float _max_108 = max_noftz(cand_3, sink_lane_1);
                                    cand_3 = _max_108;
                                } else {
                                    float _max_128 = max_noftz(cand_3, sink_lane_1);
                                    cand_3 = _max_128;
                                }
                            }
                            if constexpr (TILE_N == 16) {
                                float _max_109 = max_noftz(cand_3, m_run_1);
                                cand_3 = _max_109;
                            } else {
                                float _max_129 = max_noftz(cand_3, m_run_1);
                                cand_3 = _max_129;
                            }
                            if (it_0_1 == 0) {
                                grow_3 = 1;
                            }
                            if (cand_3 - m_run_1 > 8.0f) {
                                grow_3 = 1;
                            }
                            if (grow_3 != 0) {
                                if constexpr (TILE_N == 16) {
                                    float _exp2_26 = approx_exp2(m_run_1 - cand_3);
                                    alpha_3 = ((m_run_1 > -CAKE_INF) ? _exp2_26 : 0.0f);
                                } else {
                                    float _exp2_46 = approx_exp2(m_run_1 - cand_3);
                                    alpha_3 = ((m_run_1 > -CAKE_INF) ? _exp2_46 : 0.0f);
                                }
                                l_run_1 = l_run_1 * alpha_3;
                                r_run_1 = r_run_1 * alpha_3;
                                m_run_1 = cand_3;
                            }
                        }
                        float m_scaled_3 = ((m_run_1 > -CAKE_INF) ? m_run_1 : 0.0f);
                        unsigned int _vote_3 = __ballot_sync(0xFFFFFFFF, grow_3 != 0);
                        unsigned int grow_bits_3 = _vote_3;
                        if (it_0_1 > 0) {
                            if (grow_bits_3 != 0) {
                                float alpha_c_3[(TILE_N / 4)];
                                if constexpr (TILE_N == 16) {
                                    float _shfl_126 = __shfl_sync(0xFFFFFFFF, alpha_3, 0);
                                    alpha_c_3[0] = _shfl_126;
                                    float _shfl_127 = __shfl_sync(0xFFFFFFFF, alpha_3, 1);
                                    alpha_c_3[1] = _shfl_127;
                                    float _shfl_128 = __shfl_sync(0xFFFFFFFF, alpha_3, 2);
                                    alpha_c_3[2] = _shfl_128;
                                    float _shfl_129 = __shfl_sync(0xFFFFFFFF, alpha_3, 3);
                                    alpha_c_3[3] = _shfl_129;
                                } else {
                                    float _shfl_228 = __shfl_sync(0xFFFFFFFF, alpha_3, 0);
                                    alpha_c_3[0] = _shfl_228;
                                    float _shfl_229 = __shfl_sync(0xFFFFFFFF, alpha_3, 1);
                                    alpha_c_3[1] = _shfl_229;
                                    float _shfl_230 = __shfl_sync(0xFFFFFFFF, alpha_3, 2);
                                    alpha_c_3[2] = _shfl_230;
                                    float _shfl_231 = __shfl_sync(0xFFFFFFFF, alpha_3, 3);
                                    alpha_c_3[3] = _shfl_231;
                                    float _shfl_232 = __shfl_sync(0xFFFFFFFF, alpha_3, 4);
                                    alpha_c_3[4] = _shfl_232;
                                    float _shfl_233 = __shfl_sync(0xFFFFFFFF, alpha_3, 5);
                                    alpha_c_3[5] = _shfl_233;
                                    float _shfl_234 = __shfl_sync(0xFFFFFFFF, alpha_3, 6);
                                    alpha_c_3[6] = _shfl_234;
                                    float _shfl_235 = __shfl_sync(0xFFFFFFFF, alpha_3, 7);
                                    alpha_c_3[7] = _shfl_235;
                                }
                                float ov_3[(TILE_N / 4)];
                                if constexpr (TILE_N == 16) {
                                    tmem_ld_x4(&ov_3[0], taddr + 16 + 8 + (unsigned int)(tmem_row_origin_1 << 16));
                                } else {
                                    tmem_ld_x8(&ov_3[0], taddr + 32 + 16 + (unsigned int)(tmem_row_origin_1 << 16));
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                ov_3[0] = ov_3[0] * alpha_c_3[0];
                                ov_3[1] = ov_3[1] * alpha_c_3[1];
                                ov_3[2] = ov_3[2] * alpha_c_3[2];
                                ov_3[3] = ov_3[3] * alpha_c_3[3];
                                if constexpr (TILE_N == 16) {
                                    tmem_st_x4_f32(taddr + 16 + 8 + (unsigned int)(tmem_row_origin_1 << 16), ov_3);
                                    tmem_ld_x4(&ov_3[0], taddr + 32 + 8 + (unsigned int)(tmem_row_origin_1 << 16));
                                } else {
                                    ov_3[4] = ov_3[4] * alpha_c_3[4];
                                    ov_3[5] = ov_3[5] * alpha_c_3[5];
                                    ov_3[6] = ov_3[6] * alpha_c_3[6];
                                    ov_3[7] = ov_3[7] * alpha_c_3[7];
                                    tmem_st_x8_f32(taddr + 32 + 16 + (unsigned int)(tmem_row_origin_1 << 16), ov_3);
                                    tmem_ld_x8(&ov_3[0], taddr + 64 + 16 + (unsigned int)(tmem_row_origin_1 << 16));
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                ov_3[0] = ov_3[0] * alpha_c_3[0];
                                ov_3[1] = ov_3[1] * alpha_c_3[1];
                                ov_3[2] = ov_3[2] * alpha_c_3[2];
                                ov_3[3] = ov_3[3] * alpha_c_3[3];
                                if constexpr (TILE_N == 16) {
                                    tmem_st_x4_f32(taddr + 32 + 8 + (unsigned int)(tmem_row_origin_1 << 16), ov_3);
                                    tmem_ld_x4(&ov_3[0], taddr + 48 + 8 + (unsigned int)(tmem_row_origin_1 << 16));
                                } else {
                                    ov_3[4] = ov_3[4] * alpha_c_3[4];
                                    ov_3[5] = ov_3[5] * alpha_c_3[5];
                                    ov_3[6] = ov_3[6] * alpha_c_3[6];
                                    ov_3[7] = ov_3[7] * alpha_c_3[7];
                                    tmem_st_x8_f32(taddr + 64 + 16 + (unsigned int)(tmem_row_origin_1 << 16), ov_3);
                                    tmem_ld_x8(&ov_3[0], taddr + 96 + 16 + (unsigned int)(tmem_row_origin_1 << 16));
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                ov_3[0] = ov_3[0] * alpha_c_3[0];
                                ov_3[1] = ov_3[1] * alpha_c_3[1];
                                ov_3[2] = ov_3[2] * alpha_c_3[2];
                                ov_3[3] = ov_3[3] * alpha_c_3[3];
                                if constexpr (TILE_N == 16) {
                                    tmem_st_x4_f32(taddr + 48 + 8 + (unsigned int)(tmem_row_origin_1 << 16), ov_3);
                                    tmem_ld_x4(&ov_3[0], taddr + 64 + 8 + (unsigned int)(tmem_row_origin_1 << 16));
                                } else {
                                    ov_3[4] = ov_3[4] * alpha_c_3[4];
                                    ov_3[5] = ov_3[5] * alpha_c_3[5];
                                    ov_3[6] = ov_3[6] * alpha_c_3[6];
                                    ov_3[7] = ov_3[7] * alpha_c_3[7];
                                    tmem_st_x8_f32(taddr + 96 + 16 + (unsigned int)(tmem_row_origin_1 << 16), ov_3);
                                    tmem_ld_x8(&ov_3[0], taddr + 128 + 16 + (unsigned int)(tmem_row_origin_1 << 16));
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                ov_3[0] = ov_3[0] * alpha_c_3[0];
                                ov_3[1] = ov_3[1] * alpha_c_3[1];
                                ov_3[2] = ov_3[2] * alpha_c_3[2];
                                ov_3[3] = ov_3[3] * alpha_c_3[3];
                                if constexpr (TILE_N == 16) {
                                    tmem_st_x4_f32(taddr + 64 + 8 + (unsigned int)(tmem_row_origin_1 << 16), ov_3);
                                } else {
                                    ov_3[4] = ov_3[4] * alpha_c_3[4];
                                    ov_3[5] = ov_3[5] * alpha_c_3[5];
                                    ov_3[6] = ov_3[6] * alpha_c_3[6];
                                    ov_3[7] = ov_3[7] * alpha_c_3[7];
                                    tmem_st_x8_f32(taddr + 128 + 16 + (unsigned int)(tmem_row_origin_1 << 16), ov_3);
                                }
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                        }
                        float col_max_3[(TILE_N / 4)];
                        if constexpr (TILE_N == 16) {
                            float _shfl_130 = __shfl_sync(0xFFFFFFFF, m_scaled_3, 0);
                            col_max_3[0] = _shfl_130;
                            float _shfl_131 = __shfl_sync(0xFFFFFFFF, m_scaled_3, 1);
                            col_max_3[1] = _shfl_131;
                            float _shfl_132 = __shfl_sync(0xFFFFFFFF, m_scaled_3, 2);
                            col_max_3[2] = _shfl_132;
                            float _shfl_133 = __shfl_sync(0xFFFFFFFF, m_scaled_3, 3);
                            col_max_3[3] = _shfl_133;
                            float _exp2_27 = approx_exp2(score_values_3[0] * softmax_scale_log2_1 - col_max_3[0]);
                            score_values_3[0] = _exp2_27;
                            float _exp2_28 = approx_exp2(score_values_3[1] * softmax_scale_log2_1 - col_max_3[1]);
                            score_values_3[1] = _exp2_28;
                            float _exp2_29 = approx_exp2(score_values_3[2] * softmax_scale_log2_1 - col_max_3[2]);
                            score_values_3[2] = _exp2_29;
                            float _exp2_30 = approx_exp2(score_values_3[3] * softmax_scale_log2_1 - col_max_3[3]);
                            score_values_3[3] = _exp2_30;
                        } else {
                            float _shfl_236 = __shfl_sync(0xFFFFFFFF, m_scaled_3, 0);
                            col_max_3[0] = _shfl_236;
                            float _shfl_237 = __shfl_sync(0xFFFFFFFF, m_scaled_3, 1);
                            col_max_3[1] = _shfl_237;
                            float _shfl_238 = __shfl_sync(0xFFFFFFFF, m_scaled_3, 2);
                            col_max_3[2] = _shfl_238;
                            float _shfl_239 = __shfl_sync(0xFFFFFFFF, m_scaled_3, 3);
                            col_max_3[3] = _shfl_239;
                            float _shfl_240 = __shfl_sync(0xFFFFFFFF, m_scaled_3, 4);
                            col_max_3[4] = _shfl_240;
                            float _shfl_241 = __shfl_sync(0xFFFFFFFF, m_scaled_3, 5);
                            col_max_3[5] = _shfl_241;
                            float _shfl_242 = __shfl_sync(0xFFFFFFFF, m_scaled_3, 6);
                            col_max_3[6] = _shfl_242;
                            float _shfl_243 = __shfl_sync(0xFFFFFFFF, m_scaled_3, 7);
                            col_max_3[7] = _shfl_243;
                            float _exp2_47 = approx_exp2(score_values_3[0] * softmax_scale_log2_1 - col_max_3[0]);
                            score_values_3[0] = _exp2_47;
                            float _exp2_48 = approx_exp2(score_values_3[1] * softmax_scale_log2_1 - col_max_3[1]);
                            score_values_3[1] = _exp2_48;
                            float _exp2_49 = approx_exp2(score_values_3[2] * softmax_scale_log2_1 - col_max_3[2]);
                            score_values_3[2] = _exp2_49;
                            float _exp2_50 = approx_exp2(score_values_3[3] * softmax_scale_log2_1 - col_max_3[3]);
                            score_values_3[3] = _exp2_50;
                            float _exp2_51 = approx_exp2(score_values_3[4] * softmax_scale_log2_1 - col_max_3[4]);
                            score_values_3[4] = _exp2_51;
                            float _exp2_52 = approx_exp2(score_values_3[5] * softmax_scale_log2_1 - col_max_3[5]);
                            score_values_3[5] = _exp2_52;
                            float _exp2_53 = approx_exp2(score_values_3[6] * softmax_scale_log2_1 - col_max_3[6]);
                            score_values_3[6] = _exp2_53;
                            float _exp2_54 = approx_exp2(score_values_3[7] * softmax_scale_log2_1 - col_max_3[7]);
                            score_values_3[7] = _exp2_54;
                        }
                        {
                            if constexpr (TILE_N == 16) {
                                uint16_t _fp8_pair_98;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_98) : "f"(0.0f), "f"(score_values_3[0]));
                                uint32_t _byte_98 = (uint32_t)(_fp8_pair_98 & 0xFF);
                                uint32_t _addr_98 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1024 + row_1 ^ (1024 + row_1 >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_98), "r"(_byte_98) : "memory");
                            } else {
                                uint16_t _fp8_pair_106;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_106) : "f"(0.0f), "f"(score_values_3[0]));
                                uint32_t _byte_106 = (uint32_t)(_fp8_pair_106 & 0xFF);
                                uint32_t _addr_106 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(2048 + row_1 ^ (2048 + row_1 >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_106), "r"(_byte_106) : "memory");
                            }
                        }
                        if constexpr (TILE_N == 16) {
                            float _fp8_rt_20;
                            uint16_t _e4m3x2_99;
                            uint32_t _f16x2_99;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_99) : "f"(0.0f), "f"(score_values_3[0]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_99) : "h"(_e4m3x2_99));
                            uint16_t _fp8_h0_99 = (uint16_t)(_f16x2_99 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_20) : "h"(_fp8_h0_99));
                            tr_vals_3[0] = _fp8_rt_20;
                        } else {
                            float _fp8_rt_40;
                            uint16_t _e4m3x2_107;
                            uint32_t _f16x2_107;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_107) : "f"(0.0f), "f"(score_values_3[0]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_107) : "h"(_e4m3x2_107));
                            uint16_t _fp8_h0_107 = (uint16_t)(_f16x2_107 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_40) : "h"(_fp8_h0_107));
                            tr_vals_3[0] = _fp8_rt_40;
                        }
                        {
                            if constexpr (TILE_N == 16) {
                                uint16_t _fp8_pair_100;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_100) : "f"(0.0f), "f"(score_values_3[1]));
                                uint32_t _byte_100 = (uint32_t)(_fp8_pair_100 & 0xFF);
                                uint32_t _addr_100 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1152 + row_1 ^ (1152 + row_1 >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_100), "r"(_byte_100) : "memory");
                            } else {
                                uint16_t _fp8_pair_108;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_108) : "f"(0.0f), "f"(score_values_3[1]));
                                uint32_t _byte_108 = (uint32_t)(_fp8_pair_108 & 0xFF);
                                uint32_t _addr_108 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(2176 + row_1 ^ (2176 + row_1 >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_108), "r"(_byte_108) : "memory");
                            }
                        }
                        if constexpr (TILE_N == 16) {
                            float _fp8_rt_21;
                            uint16_t _e4m3x2_101;
                            uint32_t _f16x2_101;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_101) : "f"(0.0f), "f"(score_values_3[1]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_101) : "h"(_e4m3x2_101));
                            uint16_t _fp8_h0_101 = (uint16_t)(_f16x2_101 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_21) : "h"(_fp8_h0_101));
                            tr_vals_3[1] = _fp8_rt_21;
                        } else {
                            float _fp8_rt_41;
                            uint16_t _e4m3x2_109;
                            uint32_t _f16x2_109;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_109) : "f"(0.0f), "f"(score_values_3[1]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_109) : "h"(_e4m3x2_109));
                            uint16_t _fp8_h0_109 = (uint16_t)(_f16x2_109 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_41) : "h"(_fp8_h0_109));
                            tr_vals_3[1] = _fp8_rt_41;
                        }
                        {
                            if constexpr (TILE_N == 16) {
                                uint16_t _fp8_pair_102;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_102) : "f"(0.0f), "f"(score_values_3[2]));
                                uint32_t _byte_102 = (uint32_t)(_fp8_pair_102 & 0xFF);
                                uint32_t _addr_102 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1280 + row_1 ^ (1280 + row_1 >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_102), "r"(_byte_102) : "memory");
                            } else {
                                uint16_t _fp8_pair_110;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_110) : "f"(0.0f), "f"(score_values_3[2]));
                                uint32_t _byte_110 = (uint32_t)(_fp8_pair_110 & 0xFF);
                                uint32_t _addr_110 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(2304 + row_1 ^ (2304 + row_1 >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_110), "r"(_byte_110) : "memory");
                            }
                        }
                        if constexpr (TILE_N == 16) {
                            float _fp8_rt_22;
                            uint16_t _e4m3x2_103;
                            uint32_t _f16x2_103;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_103) : "f"(0.0f), "f"(score_values_3[2]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_103) : "h"(_e4m3x2_103));
                            uint16_t _fp8_h0_103 = (uint16_t)(_f16x2_103 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_22) : "h"(_fp8_h0_103));
                            tr_vals_3[2] = _fp8_rt_22;
                        } else {
                            float _fp8_rt_42;
                            uint16_t _e4m3x2_111;
                            uint32_t _f16x2_111;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_111) : "f"(0.0f), "f"(score_values_3[2]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_111) : "h"(_e4m3x2_111));
                            uint16_t _fp8_h0_111 = (uint16_t)(_f16x2_111 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_42) : "h"(_fp8_h0_111));
                            tr_vals_3[2] = _fp8_rt_42;
                        }
                        {
                            if constexpr (TILE_N == 16) {
                                uint16_t _fp8_pair_104;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_104) : "f"(0.0f), "f"(score_values_3[3]));
                                uint32_t _byte_104 = (uint32_t)(_fp8_pair_104 & 0xFF);
                                uint32_t _addr_104 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1408 + row_1 ^ (1408 + row_1 >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_104), "r"(_byte_104) : "memory");
                            } else {
                                uint16_t _fp8_pair_112;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_112) : "f"(0.0f), "f"(score_values_3[3]));
                                uint32_t _byte_112 = (uint32_t)(_fp8_pair_112 & 0xFF);
                                uint32_t _addr_112 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(2432 + row_1 ^ (2432 + row_1 >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_112), "r"(_byte_112) : "memory");
                            }
                        }
                        if constexpr (TILE_N == 16) {
                            float _fp8_rt_23;
                            uint16_t _e4m3x2_105;
                            uint32_t _f16x2_105;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_105) : "f"(0.0f), "f"(score_values_3[3]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_105) : "h"(_e4m3x2_105));
                            uint16_t _fp8_h0_105 = (uint16_t)(_f16x2_105 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_23) : "h"(_fp8_h0_105));
                            tr_vals_3[3] = _fp8_rt_23;
                            int hi_bit_9_3 = lane & 16;
                            float send_10_3 = ((hi_bit_9_3 != 0) ? score_values_3[0] : score_values_3[2]);
                            float keep_11_3 = ((hi_bit_9_3 != 0) ? score_values_3[2] : score_values_3[0]);
                            float _shfl_134 = __shfl_sync(0xFFFFFFFF, send_10_3, lane ^ 16);
                            float recv_12_3 = _shfl_134;
                            score_values_3[0] = keep_11_3 + recv_12_3;
                            float send_13_3 = ((hi_bit_9_3 != 0) ? score_values_3[1] : score_values_3[3]);
                            float keep_14_3 = ((hi_bit_9_3 != 0) ? score_values_3[3] : score_values_3[1]);
                            float _shfl_135 = __shfl_sync(0xFFFFFFFF, send_13_3, lane ^ 16);
                            float recv_15_3 = _shfl_135;
                            score_values_3[1] = keep_14_3 + recv_15_3;
                            int hi_bit_16_3 = lane & 8;
                            float send_17_3 = ((hi_bit_16_3 != 0) ? score_values_3[0] : score_values_3[1]);
                            float keep_18_3 = ((hi_bit_16_3 != 0) ? score_values_3[1] : score_values_3[0]);
                            float _shfl_136 = __shfl_sync(0xFFFFFFFF, send_17_3, lane ^ 8);
                            float recv_19_3 = _shfl_136;
                            score_values_3[0] = keep_18_3 + recv_19_3;
                            float _shfl_137 = __shfl_sync(0xFFFFFFFF, score_values_3[0], lane ^ 4);
                            float other_20_3 = _shfl_137;
                            score_values_3[0] = score_values_3[0] + other_20_3;
                            float _shfl_138 = __shfl_sync(0xFFFFFFFF, score_values_3[0], lane ^ 2);
                            float other_21_1 = _shfl_138;
                            score_values_3[0] = score_values_3[0] + other_21_1;
                            float _shfl_139 = __shfl_sync(0xFFFFFFFF, score_values_3[0], lane ^ 1);
                            float other_22_1 = _shfl_139;
                            score_values_3[0] = score_values_3[0] + other_22_1;
                            int hi_bit_23_1 = lane & 16;
                            float send_24_1 = ((hi_bit_23_1 != 0) ? tr_vals_3[0] : tr_vals_3[2]);
                            float keep_25_1 = ((hi_bit_23_1 != 0) ? tr_vals_3[2] : tr_vals_3[0]);
                            float _shfl_140 = __shfl_sync(0xFFFFFFFF, send_24_1, lane ^ 16);
                            float recv_26_1 = _shfl_140;
                            tr_vals_3[0] = keep_25_1 + recv_26_1;
                            float send_27_1 = ((hi_bit_23_1 != 0) ? tr_vals_3[1] : tr_vals_3[3]);
                            float keep_28_1 = ((hi_bit_23_1 != 0) ? tr_vals_3[3] : tr_vals_3[1]);
                            float _shfl_141 = __shfl_sync(0xFFFFFFFF, send_27_1, lane ^ 16);
                            float recv_29_1 = _shfl_141;
                            tr_vals_3[1] = keep_28_1 + recv_29_1;
                            int hi_bit_30_1 = lane & 8;
                            float send_31_3 = ((hi_bit_30_1 != 0) ? tr_vals_3[0] : tr_vals_3[1]);
                            float keep_32_3 = ((hi_bit_30_1 != 0) ? tr_vals_3[1] : tr_vals_3[0]);
                            float _shfl_142 = __shfl_sync(0xFFFFFFFF, send_31_3, lane ^ 8);
                            float recv_33_3 = _shfl_142;
                            tr_vals_3[0] = keep_32_3 + recv_33_3;
                            float _shfl_143 = __shfl_sync(0xFFFFFFFF, tr_vals_3[0], lane ^ 4);
                            float other_34_1 = _shfl_143;
                            tr_vals_3[0] = tr_vals_3[0] + other_34_1;
                            float _shfl_144 = __shfl_sync(0xFFFFFFFF, tr_vals_3[0], lane ^ 2);
                            float other_35_1 = _shfl_144;
                            tr_vals_3[0] = tr_vals_3[0] + other_35_1;
                            float _shfl_145 = __shfl_sync(0xFFFFFFFF, tr_vals_3[0], lane ^ 1);
                            float other_36_1 = _shfl_145;
                            tr_vals_3[0] = tr_vals_3[0] + other_36_1;
                        } else {
                            float _fp8_rt_43;
                            uint16_t _e4m3x2_113;
                            uint32_t _f16x2_113;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_113) : "f"(0.0f), "f"(score_values_3[3]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_113) : "h"(_e4m3x2_113));
                            uint16_t _fp8_h0_113 = (uint16_t)(_f16x2_113 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_43) : "h"(_fp8_h0_113));
                            tr_vals_3[3] = _fp8_rt_43;
                            {
                                uint16_t _fp8_pair_114;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_114) : "f"(0.0f), "f"(score_values_3[4]));
                                uint32_t _byte_114 = (uint32_t)(_fp8_pair_114 & 0xFF);
                                uint32_t _addr_114 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(2560 + row_1 ^ (2560 + row_1 >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_114), "r"(_byte_114) : "memory");
                            }
                            float _fp8_rt_44;
                            uint16_t _e4m3x2_115;
                            uint32_t _f16x2_115;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_115) : "f"(0.0f), "f"(score_values_3[4]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_115) : "h"(_e4m3x2_115));
                            uint16_t _fp8_h0_115 = (uint16_t)(_f16x2_115 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_44) : "h"(_fp8_h0_115));
                            tr_vals_3[4] = _fp8_rt_44;
                            {
                                uint16_t _fp8_pair_116;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_116) : "f"(0.0f), "f"(score_values_3[5]));
                                uint32_t _byte_116 = (uint32_t)(_fp8_pair_116 & 0xFF);
                                uint32_t _addr_116 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(2688 + row_1 ^ (2688 + row_1 >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_116), "r"(_byte_116) : "memory");
                            }
                            float _fp8_rt_45;
                            uint16_t _e4m3x2_117;
                            uint32_t _f16x2_117;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_117) : "f"(0.0f), "f"(score_values_3[5]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_117) : "h"(_e4m3x2_117));
                            uint16_t _fp8_h0_117 = (uint16_t)(_f16x2_117 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_45) : "h"(_fp8_h0_117));
                            tr_vals_3[5] = _fp8_rt_45;
                            {
                                uint16_t _fp8_pair_118;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_118) : "f"(0.0f), "f"(score_values_3[6]));
                                uint32_t _byte_118 = (uint32_t)(_fp8_pair_118 & 0xFF);
                                uint32_t _addr_118 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(2816 + row_1 ^ (2816 + row_1 >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_118), "r"(_byte_118) : "memory");
                            }
                            float _fp8_rt_46;
                            uint16_t _e4m3x2_119;
                            uint32_t _f16x2_119;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_119) : "f"(0.0f), "f"(score_values_3[6]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_119) : "h"(_e4m3x2_119));
                            uint16_t _fp8_h0_119 = (uint16_t)(_f16x2_119 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_46) : "h"(_fp8_h0_119));
                            tr_vals_3[6] = _fp8_rt_46;
                            {
                                uint16_t _fp8_pair_120;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_120) : "f"(0.0f), "f"(score_values_3[7]));
                                uint32_t _byte_120 = (uint32_t)(_fp8_pair_120 & 0xFF);
                                uint32_t _addr_120 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(2944 + row_1 ^ (2944 + row_1 >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_120), "r"(_byte_120) : "memory");
                            }
                            float _fp8_rt_47;
                            uint16_t _e4m3x2_121;
                            uint32_t _f16x2_121;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_121) : "f"(0.0f), "f"(score_values_3[7]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_121) : "h"(_e4m3x2_121));
                            uint16_t _fp8_h0_121 = (uint16_t)(_f16x2_121 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_47) : "h"(_fp8_h0_121));
                            tr_vals_3[7] = _fp8_rt_47;
                            int hi_bit_21_3 = lane & 16;
                            float send_22_3 = ((hi_bit_21_3 != 0) ? score_values_3[0] : score_values_3[4]);
                            float keep_23_3 = ((hi_bit_21_3 != 0) ? score_values_3[4] : score_values_3[0]);
                            float _shfl_244 = __shfl_sync(0xFFFFFFFF, send_22_3, lane ^ 16);
                            float recv_24_3 = _shfl_244;
                            score_values_3[0] = keep_23_3 + recv_24_3;
                            float send_25_3 = ((hi_bit_21_3 != 0) ? score_values_3[1] : score_values_3[5]);
                            float keep_26_3 = ((hi_bit_21_3 != 0) ? score_values_3[5] : score_values_3[1]);
                            float _shfl_245 = __shfl_sync(0xFFFFFFFF, send_25_3, lane ^ 16);
                            float recv_27_3 = _shfl_245;
                            score_values_3[1] = keep_26_3 + recv_27_3;
                            float send_28_3 = ((hi_bit_21_3 != 0) ? score_values_3[2] : score_values_3[6]);
                            float keep_29_3 = ((hi_bit_21_3 != 0) ? score_values_3[6] : score_values_3[2]);
                            float _shfl_246 = __shfl_sync(0xFFFFFFFF, send_28_3, lane ^ 16);
                            float recv_30_3 = _shfl_246;
                            score_values_3[2] = keep_29_3 + recv_30_3;
                            float send_31_3 = ((hi_bit_21_3 != 0) ? score_values_3[3] : score_values_3[7]);
                            float keep_32_3 = ((hi_bit_21_3 != 0) ? score_values_3[7] : score_values_3[3]);
                            float _shfl_247 = __shfl_sync(0xFFFFFFFF, send_31_3, lane ^ 16);
                            float recv_33_3 = _shfl_247;
                            score_values_3[3] = keep_32_3 + recv_33_3;
                            int hi_bit_34_3 = lane & 8;
                            float send_35_3 = ((hi_bit_34_3 != 0) ? score_values_3[0] : score_values_3[2]);
                            float keep_36_3 = ((hi_bit_34_3 != 0) ? score_values_3[2] : score_values_3[0]);
                            float _shfl_248 = __shfl_sync(0xFFFFFFFF, send_35_3, lane ^ 8);
                            float recv_37_3 = _shfl_248;
                            score_values_3[0] = keep_36_3 + recv_37_3;
                            float send_38_3 = ((hi_bit_34_3 != 0) ? score_values_3[1] : score_values_3[3]);
                            float keep_39_3 = ((hi_bit_34_3 != 0) ? score_values_3[3] : score_values_3[1]);
                            float _shfl_249 = __shfl_sync(0xFFFFFFFF, send_38_3, lane ^ 8);
                            float recv_40_3 = _shfl_249;
                            score_values_3[1] = keep_39_3 + recv_40_3;
                            int hi_bit_41_3 = lane & 4;
                            float send_42_3 = ((hi_bit_41_3 != 0) ? score_values_3[0] : score_values_3[1]);
                            float keep_43_3 = ((hi_bit_41_3 != 0) ? score_values_3[1] : score_values_3[0]);
                            float _shfl_250 = __shfl_sync(0xFFFFFFFF, send_42_3, lane ^ 4);
                            float recv_44_3 = _shfl_250;
                            score_values_3[0] = keep_43_3 + recv_44_3;
                            float _shfl_251 = __shfl_sync(0xFFFFFFFF, score_values_3[0], lane ^ 2);
                            float other_45_1 = _shfl_251;
                            score_values_3[0] = score_values_3[0] + other_45_1;
                            float _shfl_252 = __shfl_sync(0xFFFFFFFF, score_values_3[0], lane ^ 1);
                            float other_46_1 = _shfl_252;
                            score_values_3[0] = score_values_3[0] + other_46_1;
                            int hi_bit_47_1 = lane & 16;
                            float send_48_1 = ((hi_bit_47_1 != 0) ? tr_vals_3[0] : tr_vals_3[4]);
                            float keep_49_1 = ((hi_bit_47_1 != 0) ? tr_vals_3[4] : tr_vals_3[0]);
                            float _shfl_253 = __shfl_sync(0xFFFFFFFF, send_48_1, lane ^ 16);
                            float recv_50_1 = _shfl_253;
                            tr_vals_3[0] = keep_49_1 + recv_50_1;
                            float send_51_1 = ((hi_bit_47_1 != 0) ? tr_vals_3[1] : tr_vals_3[5]);
                            float keep_52_1 = ((hi_bit_47_1 != 0) ? tr_vals_3[5] : tr_vals_3[1]);
                            float _shfl_254 = __shfl_sync(0xFFFFFFFF, send_51_1, lane ^ 16);
                            float recv_53_1 = _shfl_254;
                            tr_vals_3[1] = keep_52_1 + recv_53_1;
                            float send_54_1 = ((hi_bit_47_1 != 0) ? tr_vals_3[2] : tr_vals_3[6]);
                            float keep_55_1 = ((hi_bit_47_1 != 0) ? tr_vals_3[6] : tr_vals_3[2]);
                            float _shfl_255 = __shfl_sync(0xFFFFFFFF, send_54_1, lane ^ 16);
                            float recv_56_1 = _shfl_255;
                            tr_vals_3[2] = keep_55_1 + recv_56_1;
                            float send_57_1 = ((hi_bit_47_1 != 0) ? tr_vals_3[3] : tr_vals_3[7]);
                            float keep_58_1 = ((hi_bit_47_1 != 0) ? tr_vals_3[7] : tr_vals_3[3]);
                            float _shfl_256 = __shfl_sync(0xFFFFFFFF, send_57_1, lane ^ 16);
                            float recv_59_1 = _shfl_256;
                            tr_vals_3[3] = keep_58_1 + recv_59_1;
                            int hi_bit_60_1 = lane & 8;
                            float send_61_3 = ((hi_bit_60_1 != 0) ? tr_vals_3[0] : tr_vals_3[2]);
                            float keep_62_3 = ((hi_bit_60_1 != 0) ? tr_vals_3[2] : tr_vals_3[0]);
                            float _shfl_257 = __shfl_sync(0xFFFFFFFF, send_61_3, lane ^ 8);
                            float recv_63_3 = _shfl_257;
                            tr_vals_3[0] = keep_62_3 + recv_63_3;
                            float send_64_3 = ((hi_bit_60_1 != 0) ? tr_vals_3[1] : tr_vals_3[3]);
                            float keep_65_3 = ((hi_bit_60_1 != 0) ? tr_vals_3[3] : tr_vals_3[1]);
                            float _shfl_258 = __shfl_sync(0xFFFFFFFF, send_64_3, lane ^ 8);
                            float recv_66_3 = _shfl_258;
                            tr_vals_3[1] = keep_65_3 + recv_66_3;
                            int hi_bit_67_1 = lane & 4;
                            float send_68_1 = ((hi_bit_67_1 != 0) ? tr_vals_3[0] : tr_vals_3[1]);
                            float keep_69_1 = ((hi_bit_67_1 != 0) ? tr_vals_3[1] : tr_vals_3[0]);
                            float _shfl_259 = __shfl_sync(0xFFFFFFFF, send_68_1, lane ^ 4);
                            float recv_70_1 = _shfl_259;
                            tr_vals_3[0] = keep_69_1 + recv_70_1;
                            float _shfl_260 = __shfl_sync(0xFFFFFFFF, tr_vals_3[0], lane ^ 2);
                            float other_71_1 = _shfl_260;
                            tr_vals_3[0] = tr_vals_3[0] + other_71_1;
                            float _shfl_261 = __shfl_sync(0xFFFFFFFF, tr_vals_3[0], lane ^ 1);
                            float other_72_1 = _shfl_261;
                            tr_vals_3[0] = tr_vals_3[0] + other_72_1;
                        }
                        if ((lane & ((-1 * TILE_N) / 4 + 11)) == 0) {
                            smem_psum[(4 + local_warp_1) * (TILE_N / 2) + (lane >> ((-1 * TILE_N) / 16 + 4))] = score_values_3[0];
                            smem_rsum[(4 + local_warp_1) * (TILE_N / 2) + (lane >> ((-1 * TILE_N) / 16 + 4))] = tr_vals_3[0];
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(p_full_addr);
                        asm volatile("barrier.sync 12, 128;" ::: "memory");
                        if (lane < (TILE_N / 4)) {
                            float tile_sum_3 = smem_psum[2 * TILE_N + lane] + smem_psum[((5 * TILE_N) / 2) + lane] + smem_psum[3 * TILE_N + lane] + smem_psum[((7 * TILE_N) / 2) + lane];
                            float tile_rsum_3 = smem_rsum[2 * TILE_N + lane] + smem_rsum[((5 * TILE_N) / 2) + lane] + smem_rsum[3 * TILE_N + lane] + smem_rsum[((7 * TILE_N) / 2) + lane];
                            if (it_0_1 == 0) {
                                float sink_term_3;
                                if constexpr (TILE_N == 16) {
                                    float _exp2_31 = approx_exp2(sink_lane_1 - m_scaled_3);
                                    sink_term_3 = _exp2_31;
                                } else {
                                    float _exp2_55 = approx_exp2(sink_lane_1 - m_scaled_3);
                                    sink_term_3 = _exp2_55;
                                }
                                l_run_1 = sink_term_3;
                                r_run_1 = sink_term_3;
                            }
                            l_run_1 = l_run_1 + tile_sum_3;
                            r_run_1 = r_run_1 + tile_rsum_3;
                        }
                        asm volatile("barrier.sync 12, 128;" ::: "memory");
                    }
                }
            }
            int last_par_1 = tiles_per_split - 1 & 1;
            {
                float norm_lane_1 = 0.0f;
                if (lane < (TILE_N / 4)) {
                    if (r_run_1 > 0.0f) {
                        float _rcp_1 = approx_rcp(r_run_1);
                        norm_lane_1 = _rcp_1 * output_scale_1;
                    }
                    if (local_warp_1 == 0) {
                        if (o_chunk_1 == 0 && head_base_1 + (TILE_N / 2) + lane < num_heads) {
                            int lse_offset_1 = (query_idx_1 * num_heads + head_base_1 + (TILE_N / 2) + lane) * num_splits + split_idx_1;
                            float _log2_1;
                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_1) : "f"(l_run_1));
                            partial_lse[lse_offset_1] = ((l_run_1 > 0.0f) ? (m_run_1 + _log2_1) * lse_partial_scale : -CAKE_INF);
                        }
                    }
                }
                float norm_c_1[(TILE_N / 4)];
                if constexpr (TILE_N == 16) {
                    float _shfl_146 = __shfl_sync(0xFFFFFFFF, norm_lane_1, 0);
                    norm_c_1[0] = _shfl_146;
                    float _shfl_147 = __shfl_sync(0xFFFFFFFF, norm_lane_1, 1);
                    norm_c_1[1] = _shfl_147;
                    float _shfl_148 = __shfl_sync(0xFFFFFFFF, norm_lane_1, 2);
                    norm_c_1[2] = _shfl_148;
                    float _shfl_149 = __shfl_sync(0xFFFFFFFF, norm_lane_1, 3);
                    norm_c_1[3] = _shfl_149;
                } else {
                    float _shfl_262 = __shfl_sync(0xFFFFFFFF, norm_lane_1, 0);
                    norm_c_1[0] = _shfl_262;
                    float _shfl_263 = __shfl_sync(0xFFFFFFFF, norm_lane_1, 1);
                    norm_c_1[1] = _shfl_263;
                    float _shfl_264 = __shfl_sync(0xFFFFFFFF, norm_lane_1, 2);
                    norm_c_1[2] = _shfl_264;
                    float _shfl_265 = __shfl_sync(0xFFFFFFFF, norm_lane_1, 3);
                    norm_c_1[3] = _shfl_265;
                    float _shfl_266 = __shfl_sync(0xFFFFFFFF, norm_lane_1, 4);
                    norm_c_1[4] = _shfl_266;
                    float _shfl_267 = __shfl_sync(0xFFFFFFFF, norm_lane_1, 5);
                    norm_c_1[5] = _shfl_267;
                    float _shfl_268 = __shfl_sync(0xFFFFFFFF, norm_lane_1, 6);
                    norm_c_1[6] = _shfl_268;
                    float _shfl_269 = __shfl_sync(0xFFFFFFFF, norm_lane_1, 7);
                    norm_c_1[7] = _shfl_269;
                }
                float o_values_1[(TILE_N / 4)];
                int dim_1 = 0;
                long long out_off_1 = 0;
                mbarrier_wait_hint(o_full_addr, last_par_1, 10000000);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if constexpr (TILE_N == 16) {
                    tmem_ld_x4(&o_values_1[0], taddr + 16 + 8 + (unsigned int)(tmem_row_origin_1 << 16));
                } else {
                    tmem_ld_x8(&o_values_1[0], taddr + 32 + 16 + (unsigned int)(tmem_row_origin_1 << 16));
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                dim_1 = o_chunk_1 * 4 * 128 + row_1;
                if (head_base_1 + (TILE_N / 2) < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + (TILE_N / 2)) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[0] * norm_c_1[0];
                }
                if (head_base_1 + (TILE_N / 2) + 1 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + (TILE_N / 2) + 1) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[1] * norm_c_1[1];
                }
                if (head_base_1 + (TILE_N / 2) + 2 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + (TILE_N / 2) + 2) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[2] * norm_c_1[2];
                }
                if (head_base_1 + (TILE_N / 2) + 3 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + (TILE_N / 2) + 3) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[3] * norm_c_1[3];
                }
                if constexpr (TILE_N == 32) {
                    if (head_base_1 + 16 + 4 < num_heads) {
                        out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 16 + 4) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                        partial_O[out_off_1] = o_values_1[4] * norm_c_1[4];
                    }
                    if (head_base_1 + 16 + 5 < num_heads) {
                        out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 16 + 5) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                        partial_O[out_off_1] = o_values_1[5] * norm_c_1[5];
                    }
                    if (head_base_1 + 16 + 6 < num_heads) {
                        out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 16 + 6) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                        partial_O[out_off_1] = o_values_1[6] * norm_c_1[6];
                    }
                    if (head_base_1 + 16 + 7 < num_heads) {
                        out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 16 + 7) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                        partial_O[out_off_1] = o_values_1[7] * norm_c_1[7];
                    }
                }
                mbarrier_wait_hint(o_full_addr + 8, last_par_1, 10000000);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if constexpr (TILE_N == 16) {
                    tmem_ld_x4(&o_values_1[0], taddr + 32 + 8 + (unsigned int)(tmem_row_origin_1 << 16));
                } else {
                    tmem_ld_x8(&o_values_1[0], taddr + 64 + 16 + (unsigned int)(tmem_row_origin_1 << 16));
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                dim_1 = (o_chunk_1 * 4 + 1) * 128 + row_1;
                if (head_base_1 + (TILE_N / 2) < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + (TILE_N / 2)) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[0] * norm_c_1[0];
                }
                if (head_base_1 + (TILE_N / 2) + 1 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + (TILE_N / 2) + 1) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[1] * norm_c_1[1];
                }
                if (head_base_1 + (TILE_N / 2) + 2 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + (TILE_N / 2) + 2) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[2] * norm_c_1[2];
                }
                if (head_base_1 + (TILE_N / 2) + 3 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + (TILE_N / 2) + 3) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[3] * norm_c_1[3];
                }
                if constexpr (TILE_N == 32) {
                    if (head_base_1 + 16 + 4 < num_heads) {
                        out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 16 + 4) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                        partial_O[out_off_1] = o_values_1[4] * norm_c_1[4];
                    }
                    if (head_base_1 + 16 + 5 < num_heads) {
                        out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 16 + 5) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                        partial_O[out_off_1] = o_values_1[5] * norm_c_1[5];
                    }
                    if (head_base_1 + 16 + 6 < num_heads) {
                        out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 16 + 6) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                        partial_O[out_off_1] = o_values_1[6] * norm_c_1[6];
                    }
                    if (head_base_1 + 16 + 7 < num_heads) {
                        out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 16 + 7) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                        partial_O[out_off_1] = o_values_1[7] * norm_c_1[7];
                    }
                }
                mbarrier_wait_hint(o_full_addr + 16, last_par_1, 10000000);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if constexpr (TILE_N == 16) {
                    tmem_ld_x4(&o_values_1[0], taddr + 48 + 8 + (unsigned int)(tmem_row_origin_1 << 16));
                } else {
                    tmem_ld_x8(&o_values_1[0], taddr + 96 + 16 + (unsigned int)(tmem_row_origin_1 << 16));
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                dim_1 = (o_chunk_1 * 4 + 2) * 128 + row_1;
                if (head_base_1 + (TILE_N / 2) < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + (TILE_N / 2)) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[0] * norm_c_1[0];
                }
                if (head_base_1 + (TILE_N / 2) + 1 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + (TILE_N / 2) + 1) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[1] * norm_c_1[1];
                }
                if (head_base_1 + (TILE_N / 2) + 2 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + (TILE_N / 2) + 2) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[2] * norm_c_1[2];
                }
                if (head_base_1 + (TILE_N / 2) + 3 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + (TILE_N / 2) + 3) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[3] * norm_c_1[3];
                }
                if constexpr (TILE_N == 32) {
                    if (head_base_1 + 16 + 4 < num_heads) {
                        out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 16 + 4) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                        partial_O[out_off_1] = o_values_1[4] * norm_c_1[4];
                    }
                    if (head_base_1 + 16 + 5 < num_heads) {
                        out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 16 + 5) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                        partial_O[out_off_1] = o_values_1[5] * norm_c_1[5];
                    }
                    if (head_base_1 + 16 + 6 < num_heads) {
                        out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 16 + 6) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                        partial_O[out_off_1] = o_values_1[6] * norm_c_1[6];
                    }
                    if (head_base_1 + 16 + 7 < num_heads) {
                        out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 16 + 7) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                        partial_O[out_off_1] = o_values_1[7] * norm_c_1[7];
                    }
                }
                mbarrier_wait_hint(o_full_addr + 24, last_par_1, 10000000);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if constexpr (TILE_N == 16) {
                    tmem_ld_x4(&o_values_1[0], taddr + 64 + 8 + (unsigned int)(tmem_row_origin_1 << 16));
                } else {
                    tmem_ld_x8(&o_values_1[0], taddr + 128 + 16 + (unsigned int)(tmem_row_origin_1 << 16));
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                dim_1 = (o_chunk_1 * 4 + 3) * 128 + row_1;
                if (head_base_1 + (TILE_N / 2) < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + (TILE_N / 2)) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[0] * norm_c_1[0];
                }
                if (head_base_1 + (TILE_N / 2) + 1 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + (TILE_N / 2) + 1) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[1] * norm_c_1[1];
                }
                if (head_base_1 + (TILE_N / 2) + 2 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + (TILE_N / 2) + 2) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[2] * norm_c_1[2];
                }
                if (head_base_1 + (TILE_N / 2) + 3 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + (TILE_N / 2) + 3) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[3] * norm_c_1[3];
                }
                if constexpr (TILE_N == 32) {
                    if (head_base_1 + 16 + 4 < num_heads) {
                        out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 16 + 4) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                        partial_O[out_off_1] = o_values_1[4] * norm_c_1[4];
                    }
                    if (head_base_1 + 16 + 5 < num_heads) {
                        out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 16 + 5) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                        partial_O[out_off_1] = o_values_1[5] * norm_c_1[5];
                    }
                    if (head_base_1 + 16 + 6 < num_heads) {
                        out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 16 + 6) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                        partial_O[out_off_1] = o_values_1[6] * norm_c_1[6];
                    }
                    if (head_base_1 + 16 + 7 < num_heads) {
                        out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 16 + 7) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                        partial_O[out_off_1] = o_values_1[7] * norm_c_1[7];
                    }
                }
                mbarrier_arrive(tmem_dealloc_addr);
            }
        }
    }
    // ---- Role: compute2 ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 144;");
        // compute2_main
        {
            const int local_warp_2 = warp - 8;
            int o_chunk_2 = 0;
            int work_idx_2 = blockIdx.x;
            int head_tile_2 = work_idx_2 % num_head_tiles;
            int split_work_2 = work_idx_2 / num_head_tiles;
            int split_idx_2 = split_work_2 % num_splits;
            int query_idx_2 = split_work_2 / num_splits;
            int head_base_2 = head_tile_2 * 128;
            const int row_2 = local_warp_2 * 32 + lane;
            const int tmem_row_origin_2 = local_warp_2 * 32;
            asm volatile("barrier.sync 9, 384;" ::: "memory");
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
            const int q_warp_2 = warp;
            for (int i_2 = 0; i_2 < 1; i_2++) {
                int unit_2 = q_warp_2 + 12 * i_2;
                if (unit_2 < 7) {
                    int q_block_2 = unit_2 / 7;
                    int kset_2 = unit_2 - q_block_2 * 7;
                    int q_row_2 = q_block_2 * 32 + lane;
                    if (head_base_2 + q_block_2 * 32 < num_heads && q_row_2 < TILE_N) {
                        int q_row_addr_2 = smem_qstage_addr + (unsigned int)(kset_2 * 128 * TILE_N) + (unsigned int)(q_row_2 * 128);
                        unsigned int sf_word_2 = 0;
                        for (int bp_2 = 0; bp_2 < 2; bp_2++) {
                            unsigned int words_2[4];
                            unsigned int qa_2[4];
                            unsigned int qb_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&qa_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_2[(0) + 3]))
                                : "r"(q_row_addr_2 + (4 * bp_2 ^ q_row_2 % 8) * 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&qb_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_3[(0) + 3]))
                                : "r"(q_row_addr_2 + (4 * bp_2 + 1 ^ q_row_2 % 8) * 16));
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
                            float _fabs_64 = fabsf(qv_3[0]);
                            float _fabs_65 = fabsf(qv_3[1]);
                            if constexpr (TILE_N == 16) {
                                float _max_110 = max_noftz(_fabs_64, _fabs_65);
                                m8_2[0] = _max_110;
                            } else {
                                float _max_130 = max_noftz(_fabs_64, _fabs_65);
                                m8_2[0] = _max_130;
                            }
                            float _fabs_66 = fabsf(qv_3[2]);
                            float _fabs_67 = fabsf(qv_3[3]);
                            if constexpr (TILE_N == 16) {
                                float _max_111 = max_noftz(_fabs_66, _fabs_67);
                                m8_2[1] = _max_111;
                            } else {
                                float _max_131 = max_noftz(_fabs_66, _fabs_67);
                                m8_2[1] = _max_131;
                            }
                            float _fabs_68 = fabsf(qv_3[4]);
                            float _fabs_69 = fabsf(qv_3[5]);
                            if constexpr (TILE_N == 16) {
                                float _max_112 = max_noftz(_fabs_68, _fabs_69);
                                m8_2[2] = _max_112;
                            } else {
                                float _max_132 = max_noftz(_fabs_68, _fabs_69);
                                m8_2[2] = _max_132;
                            }
                            float _fabs_70 = fabsf(qv_3[6]);
                            float _fabs_71 = fabsf(qv_3[7]);
                            if constexpr (TILE_N == 16) {
                                float _max_113 = max_noftz(_fabs_70, _fabs_71);
                                m8_2[3] = _max_113;
                            } else {
                                float _max_133 = max_noftz(_fabs_70, _fabs_71);
                                m8_2[3] = _max_133;
                            }
                            float _fabs_72 = fabsf(qv_3[8]);
                            float _fabs_73 = fabsf(qv_3[9]);
                            if constexpr (TILE_N == 16) {
                                float _max_114 = max_noftz(_fabs_72, _fabs_73);
                                m8_2[4] = _max_114;
                            } else {
                                float _max_134 = max_noftz(_fabs_72, _fabs_73);
                                m8_2[4] = _max_134;
                            }
                            float _fabs_74 = fabsf(qv_3[10]);
                            float _fabs_75 = fabsf(qv_3[11]);
                            if constexpr (TILE_N == 16) {
                                float _max_115 = max_noftz(_fabs_74, _fabs_75);
                                m8_2[5] = _max_115;
                            } else {
                                float _max_135 = max_noftz(_fabs_74, _fabs_75);
                                m8_2[5] = _max_135;
                            }
                            float _fabs_76 = fabsf(qv_3[12]);
                            float _fabs_77 = fabsf(qv_3[13]);
                            if constexpr (TILE_N == 16) {
                                float _max_116 = max_noftz(_fabs_76, _fabs_77);
                                m8_2[6] = _max_116;
                            } else {
                                float _max_136 = max_noftz(_fabs_76, _fabs_77);
                                m8_2[6] = _max_136;
                            }
                            float _fabs_78 = fabsf(qv_3[14]);
                            float _fabs_79 = fabsf(qv_3[15]);
                            if constexpr (TILE_N == 16) {
                                float _max_117 = max_noftz(_fabs_78, _fabs_79);
                                m8_2[7] = _max_117;
                            } else {
                                float _max_137 = max_noftz(_fabs_78, _fabs_79);
                                m8_2[7] = _max_137;
                            }
                            float m4_2[4];
                            float amax_2;
                            if constexpr (TILE_N == 16) {
                                float _max_118 = max_noftz(m8_2[0], m8_2[1]);
                                m4_2[0] = _max_118;
                                float _max_119 = max_noftz(m8_2[2], m8_2[3]);
                                m4_2[1] = _max_119;
                                float _max_120 = max_noftz(m8_2[4], m8_2[5]);
                                m4_2[2] = _max_120;
                                float _max_121 = max_noftz(m8_2[6], m8_2[7]);
                                m4_2[3] = _max_121;
                                float _max_122 = max_noftz(m4_2[0], m4_2[1]);
                                float _max_123 = max_noftz(m4_2[2], m4_2[3]);
                                float _max_124 = max_noftz(_max_122, _max_123);
                                amax_2 = _max_124;
                            } else {
                                float _max_138 = max_noftz(m8_2[0], m8_2[1]);
                                m4_2[0] = _max_138;
                                float _max_139 = max_noftz(m8_2[2], m8_2[3]);
                                m4_2[1] = _max_139;
                                float _max_140 = max_noftz(m8_2[4], m8_2[5]);
                                m4_2[2] = _max_140;
                                float _max_141 = max_noftz(m8_2[6], m8_2[7]);
                                m4_2[3] = _max_141;
                                float _max_142 = max_noftz(m4_2[0], m4_2[1]);
                                float _max_143 = max_noftz(m4_2[2], m4_2[3]);
                                float _max_144 = max_noftz(_max_142, _max_143);
                                amax_2 = _max_144;
                            }
                            float sc_2 = amax_2 * inv_six_2;
                            uint16_t _e4m3x2_f32_708;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_708) : "f"(0.0f), "f"(sc_2));
                            uint16_t sc_pair_2 = _e4m3x2_f32_708;
                            unsigned int sc_byte_2 = (unsigned int)sc_pair_2 & 255;
                            unsigned int sc_exp_2 = sc_byte_2 >> 3 & 15;
                            unsigned int sc_man_2 = sc_byte_2 & 7;
                            float inv_2 = 0.0f;
                            if (sc_exp_2 == 0) {
                                inv_2 = __uint_as_float(smem_rcptab[8 + sc_man_2]) * 512.0f;
                            } else {
                                inv_2 = __uint_as_float(smem_rcptab[sc_man_2]) * __uint_as_float(134 - sc_exp_2 << 23);
                            }
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
                            uint32_t _fp4_pair_32;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_32) : "f"(qv_3[0]), "f"(qv_3[1]));
                            uint32_t _fp4_pair_33;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_33) : "f"(qv_3[2]), "f"(qv_3[3]));
                            uint32_t _fp4_pair_34;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_34) : "f"(qv_3[4]), "f"(qv_3[5]));
                            uint32_t _fp4_pair_35;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_35) : "f"(qv_3[6]), "f"(qv_3[7]));
                            uint32_t _fp4_pair_36;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_36) : "f"(qv_3[8]), "f"(qv_3[9]));
                            uint32_t _fp4_pair_37;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_37) : "f"(qv_3[10]), "f"(qv_3[11]));
                            uint32_t _fp4_pair_38;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_38) : "f"(qv_3[12]), "f"(qv_3[13]));
                            uint32_t _fp4_pair_39;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_39) : "f"(qv_3[14]), "f"(qv_3[15]));
                            words_2[0] = _fp4_pair_32 | _fp4_pair_33 << 8 | _fp4_pair_34 << 16 | _fp4_pair_35 << 24;
                            words_2[1] = _fp4_pair_36 | _fp4_pair_37 << 8 | _fp4_pair_38 << 16 | _fp4_pair_39 << 24;
                            sf_word_2 = sf_word_2 | sc_byte_2 << (unsigned int)(8 * (2 * bp_2));
                            unsigned int qa_0_2[4];
                            unsigned int qb_1_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&qa_0_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0_2[(0) + 3]))
                                : "r"(q_row_addr_2 + (4 * bp_2 + 2 ^ q_row_2 % 8) * 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&qb_1_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1_2[(0) + 3]))
                                : "r"(q_row_addr_2 + (4 * bp_2 + 2 + 1 ^ q_row_2 % 8) * 16));
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
                            float _fabs_80 = fabsf(qv_2_2[0]);
                            float _fabs_81 = fabsf(qv_2_2[1]);
                            if constexpr (TILE_N == 16) {
                                float _max_125 = max_noftz(_fabs_80, _fabs_81);
                                m8_3_2[0] = _max_125;
                            } else {
                                float _max_145 = max_noftz(_fabs_80, _fabs_81);
                                m8_3_2[0] = _max_145;
                            }
                            float _fabs_82 = fabsf(qv_2_2[2]);
                            float _fabs_83 = fabsf(qv_2_2[3]);
                            if constexpr (TILE_N == 16) {
                                float _max_126 = max_noftz(_fabs_82, _fabs_83);
                                m8_3_2[1] = _max_126;
                            } else {
                                float _max_146 = max_noftz(_fabs_82, _fabs_83);
                                m8_3_2[1] = _max_146;
                            }
                            float _fabs_84 = fabsf(qv_2_2[4]);
                            float _fabs_85 = fabsf(qv_2_2[5]);
                            if constexpr (TILE_N == 16) {
                                float _max_127 = max_noftz(_fabs_84, _fabs_85);
                                m8_3_2[2] = _max_127;
                            } else {
                                float _max_147 = max_noftz(_fabs_84, _fabs_85);
                                m8_3_2[2] = _max_147;
                            }
                            float _fabs_86 = fabsf(qv_2_2[6]);
                            float _fabs_87 = fabsf(qv_2_2[7]);
                            if constexpr (TILE_N == 16) {
                                float _max_128 = max_noftz(_fabs_86, _fabs_87);
                                m8_3_2[3] = _max_128;
                            } else {
                                float _max_148 = max_noftz(_fabs_86, _fabs_87);
                                m8_3_2[3] = _max_148;
                            }
                            float _fabs_88 = fabsf(qv_2_2[8]);
                            float _fabs_89 = fabsf(qv_2_2[9]);
                            if constexpr (TILE_N == 16) {
                                float _max_129 = max_noftz(_fabs_88, _fabs_89);
                                m8_3_2[4] = _max_129;
                            } else {
                                float _max_149 = max_noftz(_fabs_88, _fabs_89);
                                m8_3_2[4] = _max_149;
                            }
                            float _fabs_90 = fabsf(qv_2_2[10]);
                            float _fabs_91 = fabsf(qv_2_2[11]);
                            if constexpr (TILE_N == 16) {
                                float _max_130 = max_noftz(_fabs_90, _fabs_91);
                                m8_3_2[5] = _max_130;
                            } else {
                                float _max_150 = max_noftz(_fabs_90, _fabs_91);
                                m8_3_2[5] = _max_150;
                            }
                            float _fabs_92 = fabsf(qv_2_2[12]);
                            float _fabs_93 = fabsf(qv_2_2[13]);
                            if constexpr (TILE_N == 16) {
                                float _max_131 = max_noftz(_fabs_92, _fabs_93);
                                m8_3_2[6] = _max_131;
                            } else {
                                float _max_151 = max_noftz(_fabs_92, _fabs_93);
                                m8_3_2[6] = _max_151;
                            }
                            float _fabs_94 = fabsf(qv_2_2[14]);
                            float _fabs_95 = fabsf(qv_2_2[15]);
                            if constexpr (TILE_N == 16) {
                                float _max_132 = max_noftz(_fabs_94, _fabs_95);
                                m8_3_2[7] = _max_132;
                            } else {
                                float _max_152 = max_noftz(_fabs_94, _fabs_95);
                                m8_3_2[7] = _max_152;
                            }
                            float m4_4_2[4];
                            float amax_5_2;
                            if constexpr (TILE_N == 16) {
                                float _max_133 = max_noftz(m8_3_2[0], m8_3_2[1]);
                                m4_4_2[0] = _max_133;
                                float _max_134 = max_noftz(m8_3_2[2], m8_3_2[3]);
                                m4_4_2[1] = _max_134;
                                float _max_135 = max_noftz(m8_3_2[4], m8_3_2[5]);
                                m4_4_2[2] = _max_135;
                                float _max_136 = max_noftz(m8_3_2[6], m8_3_2[7]);
                                m4_4_2[3] = _max_136;
                                float _max_137 = max_noftz(m4_4_2[0], m4_4_2[1]);
                                float _max_138 = max_noftz(m4_4_2[2], m4_4_2[3]);
                                float _max_139 = max_noftz(_max_137, _max_138);
                                amax_5_2 = _max_139;
                            } else {
                                float _max_153 = max_noftz(m8_3_2[0], m8_3_2[1]);
                                m4_4_2[0] = _max_153;
                                float _max_154 = max_noftz(m8_3_2[2], m8_3_2[3]);
                                m4_4_2[1] = _max_154;
                                float _max_155 = max_noftz(m8_3_2[4], m8_3_2[5]);
                                m4_4_2[2] = _max_155;
                                float _max_156 = max_noftz(m8_3_2[6], m8_3_2[7]);
                                m4_4_2[3] = _max_156;
                                float _max_157 = max_noftz(m4_4_2[0], m4_4_2[1]);
                                float _max_158 = max_noftz(m4_4_2[2], m4_4_2[3]);
                                float _max_159 = max_noftz(_max_157, _max_158);
                                amax_5_2 = _max_159;
                            }
                            float sc_6_2 = amax_5_2 * inv_six_2;
                            uint16_t _e4m3x2_f32_709;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_709) : "f"(0.0f), "f"(sc_6_2));
                            uint16_t sc_pair_7_2 = _e4m3x2_f32_709;
                            unsigned int sc_byte_8_2 = (unsigned int)sc_pair_7_2 & 255;
                            unsigned int sc_exp_9_2 = sc_byte_8_2 >> 3 & 15;
                            unsigned int sc_man_10_2 = sc_byte_8_2 & 7;
                            float inv_11_2 = 0.0f;
                            if (sc_exp_9_2 == 0) {
                                inv_11_2 = __uint_as_float(smem_rcptab[8 + sc_man_10_2]) * 512.0f;
                            } else {
                                inv_11_2 = __uint_as_float(smem_rcptab[sc_man_10_2]) * __uint_as_float(134 - sc_exp_9_2 << 23);
                            }
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_1 = {inv_11_2, inv_11_2};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_2_2)[_ls], _scale2_1);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++) {
                                qv_2_2[_ls] = qv_2_2[_ls] * inv_11_2;
                            }
                            #endif
                            uint32_t _fp4_pair_40;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_40) : "f"(qv_2_2[0]), "f"(qv_2_2[1]));
                            uint32_t _fp4_pair_41;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_41) : "f"(qv_2_2[2]), "f"(qv_2_2[3]));
                            uint32_t _fp4_pair_42;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_42) : "f"(qv_2_2[4]), "f"(qv_2_2[5]));
                            uint32_t _fp4_pair_43;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_43) : "f"(qv_2_2[6]), "f"(qv_2_2[7]));
                            uint32_t _fp4_pair_44;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_44) : "f"(qv_2_2[8]), "f"(qv_2_2[9]));
                            uint32_t _fp4_pair_45;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_45) : "f"(qv_2_2[10]), "f"(qv_2_2[11]));
                            uint32_t _fp4_pair_46;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_46) : "f"(qv_2_2[12]), "f"(qv_2_2[13]));
                            uint32_t _fp4_pair_47;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_47) : "f"(qv_2_2[14]), "f"(qv_2_2[15]));
                            words_2[2] = _fp4_pair_40 | _fp4_pair_41 << 8 | _fp4_pair_42 << 16 | _fp4_pair_43 << 24;
                            words_2[3] = _fp4_pair_44 | _fp4_pair_45 << 8 | _fp4_pair_46 << 16 | _fp4_pair_47 << 24;
                            sf_word_2 = sf_word_2 | sc_byte_8_2 << (unsigned int)(8 * (2 * bp_2 + 1));
                            int chunk_2 = 2 * kset_2 + bp_2;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(smem_qf4_addr + (unsigned int)(chunk_2 / 8 * 4096 + (q_row_2 * 128 + (chunk_2 % 8 * 16 ^ q_row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&words_2[0])), "r"(*reinterpret_cast<uint32_t*>(&words_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words_2[(0) + 3])));
                        }
                        smem_qsf32[kset_2 / 4 * 2048 + q_row_2 % 32 / 8 * 512 + kset_2 % 4 * 128 + q_row_2 % 8 * 16 + q_row_2 / 32 % 4 * 4 >> 2] = sf_word_2;
                    } else {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_qf4_addr + (unsigned int)(2 * kset_2 / 8 * 4096 + (q_row_2 * 128 + (2 * kset_2 % 8 * 16 ^ q_row_2 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_qf4_addr + (unsigned int)((2 * kset_2 + 1) / 8 * 4096 + (q_row_2 * 128 + ((2 * kset_2 + 1) % 8 * 16 ^ q_row_2 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                        smem_qsf32[kset_2 / 4 * 2048 + q_row_2 % 32 / 8 * 512 + kset_2 % 4 * 128 + q_row_2 % 8 * 16 + q_row_2 / 32 % 4 * 4 >> 2] = 0;
                    }
                }
            }
            if (local_warp_2 == 0) {
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_qf4_addr + (unsigned int)(4096 + (row_2 * 128 + (96 ^ row_2 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_qf4_addr + (unsigned int)(4096 + (row_2 * 128 + (112 ^ row_2 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                smem_qsf32[2048 + row_2 % 32 / 8 * 512 + 384 + row_2 % 8 * 16 + row_2 / 32 % 4 * 4 >> 2] = 0;
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(q_ready_addr);
            asm volatile("barrier.sync 9, 384;" ::: "memory");
            mbarrier_arrive(tok_free_addr + 8);
            float softmax_scale_log2_2 = bmm1_scale[0] * 1.4426950408889634f;
            float output_scale_2 = bmm2_scale[0];
            int strip_0_2 = smem_sfs_0_addr + (unsigned int)(row_2 * 32);
            int strip_1_2 = smem_sfs_1_addr + (unsigned int)(row_2 * 32);
            float m_run_2 = -CAKE_INF;
            float l_run_2 = 0.0f;
            float r_run_2 = 0.0f;
            float sink_lane_2 = -CAKE_INF;
            if (lane < (TILE_N / 4)) {
                if (has_sinks != 0 && split_idx_2 == 0 && head_base_2 + ((3 * TILE_N) / 4) + lane < num_heads) {
                    sink_lane_2 = sinks[head_base_2 + ((3 * TILE_N) / 4) + lane] * 1.4426950408889634f;
                }
            }
            for (int it2_2 = 0; it2_2 < (tiles_per_split + 1) / 2; it2_2++) {
                int par2_2 = it2_2 & 1;
                int it_2 = 2 * it2_2;
                if (it_2 < tiles_per_split) {
                    mbarrier_wait_hint(tok_full_addr, it2_2 & 1, 10000000);
                    int raw_index_4 = smem_tok_0[row_2];
                    int valid_4 = 1;
                    if (raw_index_4 < 0) {
                        valid_4 = 0;
                    }
                    if (valid_4 != 0) {
                    } else if (0) {
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(kv_full_addr);
                    if (it_2 > 0) {
                        mbarrier_wait_hint(o_full_addr, 1, 10000000);
                        mbarrier_wait_hint(o_full_addr + 8, 1, 10000000);
                        mbarrier_wait_hint(o_full_addr + 16, 1, 10000000);
                        mbarrier_wait_hint(o_full_addr + 24, 1, 10000000);
                    }
                    if (valid_4 != 0) {
                        unsigned int kraw4_4[4];
                        unsigned int sfw32_4 = 0;
                        int vblock_7 = 32 * o_chunk_2 + 22;
                        unsigned int v8_8[4];
                        {
                            {
                                int vchunk_24 = 16 * o_chunk_2 + 11;
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&kraw4_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_4[(0) + 3]))
                                    : "r"(smem_kf4_0_addr + (unsigned int)(vchunk_24 / 8 * 16384) + (unsigned int)(row_2 * 128 + (vchunk_24 % 8 * 16 ^ row_2 % 8 * 16))));
                            }
                            {
                                sfw32_4 = smem_sfs32_0[row_2 * 8 + 8 * o_chunk_2 + 5];
                            }
                            unsigned int scale_44 = sfw32_4 >> 16 & 255;
                            {
                                v8_8[0] = cake_dsv4_qmul4_portable<5>(kraw4_4[0], scale_44);
                            }
                            {
                                v8_8[1] = cake_dsv4_qmul4_portable<6>(kraw4_4[0], scale_44);
                            }
                            {
                                v8_8[2] = cake_dsv4_qmul4_portable<5>(kraw4_4[1], scale_44);
                            }
                            {
                                v8_8[3] = cake_dsv4_qmul4_portable<6>(kraw4_4[1], scale_44);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(32768 + (row_2 * 128 + (96 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_8[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_8[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_8[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_8[(0) + 3])));
                        int vblock_0_4 = 32 * o_chunk_2 + 23;
                        unsigned int v8_1_4[4];
                        {
                            unsigned int scale_45 = sfw32_4 >> 24 & 255;
                            {
                                v8_1_4[0] = cake_dsv4_qmul4_portable<5>(kraw4_4[2], scale_45);
                            }
                            {
                                v8_1_4[1] = cake_dsv4_qmul4_portable<6>(kraw4_4[2], scale_45);
                            }
                            {
                                v8_1_4[2] = cake_dsv4_qmul4_portable<5>(kraw4_4[3], scale_45);
                            }
                            {
                                v8_1_4[3] = cake_dsv4_qmul4_portable<6>(kraw4_4[3], scale_45);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(32768 + (row_2 * 128 + (112 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_1_4[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_1_4[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_1_4[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_1_4[(0) + 3])));
                        int vblock_2_4 = 32 * o_chunk_2 + 24;
                        unsigned int v8_3_4[4];
                        {
                            {
                                int vchunk_25 = 16 * o_chunk_2 + 12;
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&kraw4_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_4[(0) + 3]))
                                    : "r"(smem_kf4_0_addr + (unsigned int)(vchunk_25 / 8 * 16384) + (unsigned int)(row_2 * 128 + (vchunk_25 % 8 * 16 ^ row_2 % 8 * 16))));
                            }
                            {
                                sfw32_4 = smem_sfs32_0[row_2 * 8 + 8 * o_chunk_2 + 6];
                            }
                            unsigned int scale_46 = sfw32_4 & 255;
                            {
                                v8_3_4[0] = cake_dsv4_qmul4_portable<5>(kraw4_4[0], scale_46);
                            }
                            {
                                v8_3_4[1] = cake_dsv4_qmul4_portable<6>(kraw4_4[0], scale_46);
                            }
                            {
                                v8_3_4[2] = cake_dsv4_qmul4_portable<5>(kraw4_4[1], scale_46);
                            }
                            {
                                v8_3_4[3] = cake_dsv4_qmul4_portable<6>(kraw4_4[1], scale_46);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (0 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_3_4[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_3_4[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_3_4[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_3_4[(0) + 3])));
                        int vblock_4_4 = 32 * o_chunk_2 + 25;
                        unsigned int v8_5_4[4];
                        {
                            unsigned int scale_47 = sfw32_4 >> 8 & 255;
                            {
                                v8_5_4[0] = cake_dsv4_qmul4_portable<5>(kraw4_4[2], scale_47);
                            }
                            {
                                v8_5_4[1] = cake_dsv4_qmul4_portable<6>(kraw4_4[2], scale_47);
                            }
                            {
                                v8_5_4[2] = cake_dsv4_qmul4_portable<5>(kraw4_4[3], scale_47);
                            }
                            {
                                v8_5_4[3] = cake_dsv4_qmul4_portable<6>(kraw4_4[3], scale_47);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (16 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_5_4[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_5_4[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_5_4[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_5_4[(0) + 3])));
                        int vblock_6_4 = 32 * o_chunk_2 + 26;
                        unsigned int v8_7_4[4];
                        {
                            {
                                int vchunk_26 = 16 * o_chunk_2 + 13;
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&kraw4_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_4[(0) + 3]))
                                    : "r"(smem_kf4_0_addr + (unsigned int)(vchunk_26 / 8 * 16384) + (unsigned int)(row_2 * 128 + (vchunk_26 % 8 * 16 ^ row_2 % 8 * 16))));
                            }
                            unsigned int scale_48 = sfw32_4 >> 16 & 255;
                            {
                                v8_7_4[0] = cake_dsv4_qmul4_portable<5>(kraw4_4[0], scale_48);
                            }
                            {
                                v8_7_4[1] = cake_dsv4_qmul4_portable<6>(kraw4_4[0], scale_48);
                            }
                            {
                                v8_7_4[2] = cake_dsv4_qmul4_portable<5>(kraw4_4[1], scale_48);
                            }
                            {
                                v8_7_4[3] = cake_dsv4_qmul4_portable<6>(kraw4_4[1], scale_48);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (32 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_7_4[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_7_4[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_7_4[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_7_4[(0) + 3])));
                        int vblock_8_4 = 32 * o_chunk_2 + 27;
                        unsigned int v8_9_4[4];
                        {
                            unsigned int scale_49 = sfw32_4 >> 24 & 255;
                            {
                                v8_9_4[0] = cake_dsv4_qmul4_portable<5>(kraw4_4[2], scale_49);
                            }
                            {
                                v8_9_4[1] = cake_dsv4_qmul4_portable<6>(kraw4_4[2], scale_49);
                            }
                            {
                                v8_9_4[2] = cake_dsv4_qmul4_portable<5>(kraw4_4[3], scale_49);
                            }
                            {
                                v8_9_4[3] = cake_dsv4_qmul4_portable<6>(kraw4_4[3], scale_49);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (48 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_9_4[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_9_4[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_9_4[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_9_4[(0) + 3])));
                        int vblock_10_4 = 32 * o_chunk_2 + 28;
                        unsigned int v8_11_4[4];
                        {
                            int rblock = vblock_10_4 - 28;
                            unsigned int rope[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&rope[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope[(0) + 3]))
                                : "r"(smem_krope_0_addr + (unsigned int)(row_2 * 128 + (2 * rblock * 16 ^ row_2 % 8 * 16))));
                            float lo = __uint_as_float(rope[0] << 16);
                            float hi = __uint_as_float(rope[0] & 4294901760u);
                            uint16_t _e4m3x2_f32_806;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_806) : "f"(hi), "f"(lo));
                            uint16_t pair = _e4m3x2_f32_806;
                            {
                                v8_11_4[0] = (unsigned int)pair;
                            }
                            float lo_0 = __uint_as_float(rope[1] << 16);
                            float hi_1 = __uint_as_float(rope[1] & 4294901760u);
                            uint16_t _e4m3x2_f32_807;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_807) : "f"(hi_1), "f"(lo_0));
                            uint16_t pair_2 = _e4m3x2_f32_807;
                            {
                                v8_11_4[0] = v8_11_4[0] | (unsigned int)pair_2 << 16;
                            }
                            float lo_3 = __uint_as_float(rope[2] << 16);
                            float hi_4 = __uint_as_float(rope[2] & 4294901760u);
                            uint16_t _e4m3x2_f32_808;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_808) : "f"(hi_4), "f"(lo_3));
                            uint16_t pair_5 = _e4m3x2_f32_808;
                            {
                                v8_11_4[1] = (unsigned int)pair_5;
                            }
                            float lo_6 = __uint_as_float(rope[3] << 16);
                            float hi_7 = __uint_as_float(rope[3] & 4294901760u);
                            uint16_t _e4m3x2_f32_809;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_809) : "f"(hi_7), "f"(lo_6));
                            uint16_t pair_8 = _e4m3x2_f32_809;
                            {
                                v8_11_4[1] = v8_11_4[1] | (unsigned int)pair_8 << 16;
                            }
                            unsigned int rope_9[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&rope_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9[(0) + 3]))
                                : "r"(smem_krope_0_addr + (unsigned int)(row_2 * 128 + ((2 * rblock + 1) * 16 ^ row_2 % 8 * 16))));
                            float lo_10 = __uint_as_float(rope_9[0] << 16);
                            float hi_11 = __uint_as_float(rope_9[0] & 4294901760u);
                            uint16_t _e4m3x2_f32_810;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_810) : "f"(hi_11), "f"(lo_10));
                            uint16_t pair_12 = _e4m3x2_f32_810;
                            {
                                v8_11_4[2] = (unsigned int)pair_12;
                            }
                            float lo_13 = __uint_as_float(rope_9[1] << 16);
                            float hi_14 = __uint_as_float(rope_9[1] & 4294901760u);
                            uint16_t _e4m3x2_f32_811;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_811) : "f"(hi_14), "f"(lo_13));
                            uint16_t pair_15 = _e4m3x2_f32_811;
                            {
                                v8_11_4[2] = v8_11_4[2] | (unsigned int)pair_15 << 16;
                            }
                            float lo_16 = __uint_as_float(rope_9[2] << 16);
                            float hi_17 = __uint_as_float(rope_9[2] & 4294901760u);
                            uint16_t _e4m3x2_f32_812;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_812) : "f"(hi_17), "f"(lo_16));
                            uint16_t pair_18 = _e4m3x2_f32_812;
                            {
                                v8_11_4[3] = (unsigned int)pair_18;
                            }
                            float lo_19 = __uint_as_float(rope_9[3] << 16);
                            float hi_20 = __uint_as_float(rope_9[3] & 4294901760u);
                            uint16_t _e4m3x2_f32_813;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_813) : "f"(hi_20), "f"(lo_19));
                            uint16_t pair_21 = _e4m3x2_f32_813;
                            {
                                v8_11_4[3] = v8_11_4[3] | (unsigned int)pair_21 << 16;
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (64 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_11_4[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_11_4[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_11_4[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_11_4[(0) + 3])));
                        int vblock_12_4 = 32 * o_chunk_2 + 29;
                        unsigned int v8_13_4[4];
                        {
                            int rblock_1 = vblock_12_4 - 28;
                            unsigned int rope_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&rope_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_1[(0) + 3]))
                                : "r"(smem_krope_0_addr + (unsigned int)(row_2 * 128 + (2 * rblock_1 * 16 ^ row_2 % 8 * 16))));
                            float lo_1 = __uint_as_float(rope_1[0] << 16);
                            float hi_2 = __uint_as_float(rope_1[0] & 4294901760u);
                            uint16_t _e4m3x2_f32_822;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_822) : "f"(hi_2), "f"(lo_1));
                            uint16_t pair_1 = _e4m3x2_f32_822;
                            {
                                v8_13_4[0] = (unsigned int)pair_1;
                            }
                            float lo_0_1 = __uint_as_float(rope_1[1] << 16);
                            float hi_1_1 = __uint_as_float(rope_1[1] & 4294901760u);
                            uint16_t _e4m3x2_f32_823;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_823) : "f"(hi_1_1), "f"(lo_0_1));
                            uint16_t pair_2_1 = _e4m3x2_f32_823;
                            {
                                v8_13_4[0] = v8_13_4[0] | (unsigned int)pair_2_1 << 16;
                            }
                            float lo_3_1 = __uint_as_float(rope_1[2] << 16);
                            float hi_4_1 = __uint_as_float(rope_1[2] & 4294901760u);
                            uint16_t _e4m3x2_f32_824;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_824) : "f"(hi_4_1), "f"(lo_3_1));
                            uint16_t pair_5_1 = _e4m3x2_f32_824;
                            {
                                v8_13_4[1] = (unsigned int)pair_5_1;
                            }
                            float lo_6_1 = __uint_as_float(rope_1[3] << 16);
                            float hi_7_1 = __uint_as_float(rope_1[3] & 4294901760u);
                            uint16_t _e4m3x2_f32_825;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_825) : "f"(hi_7_1), "f"(lo_6_1));
                            uint16_t pair_8_1 = _e4m3x2_f32_825;
                            {
                                v8_13_4[1] = v8_13_4[1] | (unsigned int)pair_8_1 << 16;
                            }
                            unsigned int rope_9_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&rope_9_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_1[(0) + 3]))
                                : "r"(smem_krope_0_addr + (unsigned int)(row_2 * 128 + ((2 * rblock_1 + 1) * 16 ^ row_2 % 8 * 16))));
                            float lo_10_1 = __uint_as_float(rope_9_1[0] << 16);
                            float hi_11_1 = __uint_as_float(rope_9_1[0] & 4294901760u);
                            uint16_t _e4m3x2_f32_826;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_826) : "f"(hi_11_1), "f"(lo_10_1));
                            uint16_t pair_12_1 = _e4m3x2_f32_826;
                            {
                                v8_13_4[2] = (unsigned int)pair_12_1;
                            }
                            float lo_13_1 = __uint_as_float(rope_9_1[1] << 16);
                            float hi_14_1 = __uint_as_float(rope_9_1[1] & 4294901760u);
                            uint16_t _e4m3x2_f32_827;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_827) : "f"(hi_14_1), "f"(lo_13_1));
                            uint16_t pair_15_1 = _e4m3x2_f32_827;
                            {
                                v8_13_4[2] = v8_13_4[2] | (unsigned int)pair_15_1 << 16;
                            }
                            float lo_16_1 = __uint_as_float(rope_9_1[2] << 16);
                            float hi_17_1 = __uint_as_float(rope_9_1[2] & 4294901760u);
                            uint16_t _e4m3x2_f32_828;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_828) : "f"(hi_17_1), "f"(lo_16_1));
                            uint16_t pair_18_1 = _e4m3x2_f32_828;
                            {
                                v8_13_4[3] = (unsigned int)pair_18_1;
                            }
                            float lo_19_1 = __uint_as_float(rope_9_1[3] << 16);
                            float hi_20_1 = __uint_as_float(rope_9_1[3] & 4294901760u);
                            uint16_t _e4m3x2_f32_829;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_829) : "f"(hi_20_1), "f"(lo_19_1));
                            uint16_t pair_21_1 = _e4m3x2_f32_829;
                            {
                                v8_13_4[3] = v8_13_4[3] | (unsigned int)pair_21_1 << 16;
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (80 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_13_4[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_13_4[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_13_4[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_13_4[(0) + 3])));
                        int vblock_14_4 = 32 * o_chunk_2 + 30;
                        unsigned int v8_15_4[4];
                        {
                            int rblock_2 = vblock_14_4 - 28;
                            unsigned int rope_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&rope_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_2[(0) + 3]))
                                : "r"(smem_krope_0_addr + (unsigned int)(row_2 * 128 + (2 * rblock_2 * 16 ^ row_2 % 8 * 16))));
                            float lo_2 = __uint_as_float(rope_2[0] << 16);
                            float hi_3 = __uint_as_float(rope_2[0] & 4294901760u);
                            uint16_t _e4m3x2_f32_838;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_838) : "f"(hi_3), "f"(lo_2));
                            uint16_t pair_3 = _e4m3x2_f32_838;
                            {
                                v8_15_4[0] = (unsigned int)pair_3;
                            }
                            float lo_0_2 = __uint_as_float(rope_2[1] << 16);
                            float hi_1_2 = __uint_as_float(rope_2[1] & 4294901760u);
                            uint16_t _e4m3x2_f32_839;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_839) : "f"(hi_1_2), "f"(lo_0_2));
                            uint16_t pair_2_2 = _e4m3x2_f32_839;
                            {
                                v8_15_4[0] = v8_15_4[0] | (unsigned int)pair_2_2 << 16;
                            }
                            float lo_3_2 = __uint_as_float(rope_2[2] << 16);
                            float hi_4_2 = __uint_as_float(rope_2[2] & 4294901760u);
                            uint16_t _e4m3x2_f32_840;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_840) : "f"(hi_4_2), "f"(lo_3_2));
                            uint16_t pair_5_2 = _e4m3x2_f32_840;
                            {
                                v8_15_4[1] = (unsigned int)pair_5_2;
                            }
                            float lo_6_2 = __uint_as_float(rope_2[3] << 16);
                            float hi_7_2 = __uint_as_float(rope_2[3] & 4294901760u);
                            uint16_t _e4m3x2_f32_841;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_841) : "f"(hi_7_2), "f"(lo_6_2));
                            uint16_t pair_8_2 = _e4m3x2_f32_841;
                            {
                                v8_15_4[1] = v8_15_4[1] | (unsigned int)pair_8_2 << 16;
                            }
                            unsigned int rope_9_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&rope_9_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_2[(0) + 3]))
                                : "r"(smem_krope_0_addr + (unsigned int)(row_2 * 128 + ((2 * rblock_2 + 1) * 16 ^ row_2 % 8 * 16))));
                            float lo_10_2 = __uint_as_float(rope_9_2[0] << 16);
                            float hi_11_2 = __uint_as_float(rope_9_2[0] & 4294901760u);
                            uint16_t _e4m3x2_f32_842;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_842) : "f"(hi_11_2), "f"(lo_10_2));
                            uint16_t pair_12_2 = _e4m3x2_f32_842;
                            {
                                v8_15_4[2] = (unsigned int)pair_12_2;
                            }
                            float lo_13_2 = __uint_as_float(rope_9_2[1] << 16);
                            float hi_14_2 = __uint_as_float(rope_9_2[1] & 4294901760u);
                            uint16_t _e4m3x2_f32_843;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_843) : "f"(hi_14_2), "f"(lo_13_2));
                            uint16_t pair_15_2 = _e4m3x2_f32_843;
                            {
                                v8_15_4[2] = v8_15_4[2] | (unsigned int)pair_15_2 << 16;
                            }
                            float lo_16_2 = __uint_as_float(rope_9_2[2] << 16);
                            float hi_17_2 = __uint_as_float(rope_9_2[2] & 4294901760u);
                            uint16_t _e4m3x2_f32_844;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_844) : "f"(hi_17_2), "f"(lo_16_2));
                            uint16_t pair_18_2 = _e4m3x2_f32_844;
                            {
                                v8_15_4[3] = (unsigned int)pair_18_2;
                            }
                            float lo_19_2 = __uint_as_float(rope_9_2[3] << 16);
                            float hi_20_2 = __uint_as_float(rope_9_2[3] & 4294901760u);
                            uint16_t _e4m3x2_f32_845;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_845) : "f"(hi_20_2), "f"(lo_19_2));
                            uint16_t pair_21_2 = _e4m3x2_f32_845;
                            {
                                v8_15_4[3] = v8_15_4[3] | (unsigned int)pair_21_2 << 16;
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (96 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_15_4[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_15_4[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_15_4[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_15_4[(0) + 3])));
                        int vblock_16_4 = 32 * o_chunk_2 + 31;
                        unsigned int v8_17_4[4];
                        {
                            int rblock_3 = vblock_16_4 - 28;
                            unsigned int rope_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&rope_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_3[(0) + 3]))
                                : "r"(smem_krope_0_addr + (unsigned int)(row_2 * 128 + (2 * rblock_3 * 16 ^ row_2 % 8 * 16))));
                            float lo_4 = __uint_as_float(rope_3[0] << 16);
                            float hi_5 = __uint_as_float(rope_3[0] & 4294901760u);
                            uint16_t _e4m3x2_f32_854;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_854) : "f"(hi_5), "f"(lo_4));
                            uint16_t pair_4 = _e4m3x2_f32_854;
                            {
                                v8_17_4[0] = (unsigned int)pair_4;
                            }
                            float lo_0_3 = __uint_as_float(rope_3[1] << 16);
                            float hi_1_3 = __uint_as_float(rope_3[1] & 4294901760u);
                            uint16_t _e4m3x2_f32_855;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_855) : "f"(hi_1_3), "f"(lo_0_3));
                            uint16_t pair_2_3 = _e4m3x2_f32_855;
                            {
                                v8_17_4[0] = v8_17_4[0] | (unsigned int)pair_2_3 << 16;
                            }
                            float lo_3_3 = __uint_as_float(rope_3[2] << 16);
                            float hi_4_3 = __uint_as_float(rope_3[2] & 4294901760u);
                            uint16_t _e4m3x2_f32_856;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_856) : "f"(hi_4_3), "f"(lo_3_3));
                            uint16_t pair_5_3 = _e4m3x2_f32_856;
                            {
                                v8_17_4[1] = (unsigned int)pair_5_3;
                            }
                            float lo_6_3 = __uint_as_float(rope_3[3] << 16);
                            float hi_7_3 = __uint_as_float(rope_3[3] & 4294901760u);
                            uint16_t _e4m3x2_f32_857;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_857) : "f"(hi_7_3), "f"(lo_6_3));
                            uint16_t pair_8_3 = _e4m3x2_f32_857;
                            {
                                v8_17_4[1] = v8_17_4[1] | (unsigned int)pair_8_3 << 16;
                            }
                            unsigned int rope_9_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&rope_9_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_3[(0) + 3]))
                                : "r"(smem_krope_0_addr + (unsigned int)(row_2 * 128 + ((2 * rblock_3 + 1) * 16 ^ row_2 % 8 * 16))));
                            float lo_10_3 = __uint_as_float(rope_9_3[0] << 16);
                            float hi_11_3 = __uint_as_float(rope_9_3[0] & 4294901760u);
                            uint16_t _e4m3x2_f32_858;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_858) : "f"(hi_11_3), "f"(lo_10_3));
                            uint16_t pair_12_3 = _e4m3x2_f32_858;
                            {
                                v8_17_4[2] = (unsigned int)pair_12_3;
                            }
                            float lo_13_3 = __uint_as_float(rope_9_3[1] << 16);
                            float hi_14_3 = __uint_as_float(rope_9_3[1] & 4294901760u);
                            uint16_t _e4m3x2_f32_859;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_859) : "f"(hi_14_3), "f"(lo_13_3));
                            uint16_t pair_15_3 = _e4m3x2_f32_859;
                            {
                                v8_17_4[2] = v8_17_4[2] | (unsigned int)pair_15_3 << 16;
                            }
                            float lo_16_3 = __uint_as_float(rope_9_3[2] << 16);
                            float hi_17_3 = __uint_as_float(rope_9_3[2] & 4294901760u);
                            uint16_t _e4m3x2_f32_860;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_860) : "f"(hi_17_3), "f"(lo_16_3));
                            uint16_t pair_18_3 = _e4m3x2_f32_860;
                            {
                                v8_17_4[3] = (unsigned int)pair_18_3;
                            }
                            float lo_19_3 = __uint_as_float(rope_9_3[3] << 16);
                            float hi_20_3 = __uint_as_float(rope_9_3[3] & 4294901760u);
                            uint16_t _e4m3x2_f32_861;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_861) : "f"(hi_20_3), "f"(lo_19_3));
                            uint16_t pair_21_3 = _e4m3x2_f32_861;
                            {
                                v8_17_4[3] = v8_17_4[3] | (unsigned int)pair_21_3 << 16;
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (112 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_17_4[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_17_4[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_17_4[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_17_4[(0) + 3])));
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
                    mbarrier_wait_hint(s_full_addr, 0, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (it_2 + 2 < tiles_per_split) {
                        mbarrier_arrive(tok_free_addr);
                    }
                    {
                        float score_values_4[(TILE_N / 4)];
                        if constexpr (TILE_N == 16) {
                            tmem_ld_x4(&score_values_4[0], taddr + 12 + (unsigned int)(tmem_row_origin_2 << 16));
                        } else {
                            tmem_ld_x8(&score_values_4[0], taddr + 24 + (unsigned int)(tmem_row_origin_2 << 16));
                        }
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        if (valid_4 == 0) {
                            score_values_4[0] = -CAKE_INF;
                            score_values_4[1] = -CAKE_INF;
                            score_values_4[2] = -CAKE_INF;
                            score_values_4[3] = -CAKE_INF;
                            if constexpr (TILE_N == 32) {
                                score_values_4[4] = -CAKE_INF;
                                score_values_4[5] = -CAKE_INF;
                                score_values_4[6] = -CAKE_INF;
                                score_values_4[7] = -CAKE_INF;
                            }
                        }
                        float tr_vals_4[(TILE_N / 4)];
                        tr_vals_4[0] = score_values_4[0];
                        tr_vals_4[1] = score_values_4[1];
                        tr_vals_4[2] = score_values_4[2];
                        tr_vals_4[3] = score_values_4[3];
                        if constexpr (TILE_N == 16) {
                            int hi_bit_5 = lane & 16;
                            float send_7 = ((hi_bit_5 != 0) ? tr_vals_4[0] : tr_vals_4[2]);
                            float keep_8 = ((hi_bit_5 != 0) ? tr_vals_4[2] : tr_vals_4[0]);
                            float _shfl_150 = __shfl_sync(0xFFFFFFFF, send_7, lane ^ 16);
                            float recv_7 = _shfl_150;
                            float _max_140 = max_noftz(keep_8, recv_7);
                            tr_vals_4[0] = _max_140;
                            float send_0_4 = ((hi_bit_5 != 0) ? tr_vals_4[1] : tr_vals_4[3]);
                            float keep_1_4 = ((hi_bit_5 != 0) ? tr_vals_4[3] : tr_vals_4[1]);
                            float _shfl_151 = __shfl_sync(0xFFFFFFFF, send_0_4, lane ^ 16);
                            float recv_2_4 = _shfl_151;
                            float _max_141 = max_noftz(keep_1_4, recv_2_4);
                            tr_vals_4[1] = _max_141;
                            int hi_bit_3_2 = lane & 8;
                            float send_4_2 = ((hi_bit_3_2 != 0) ? tr_vals_4[0] : tr_vals_4[1]);
                            float keep_5_2 = ((hi_bit_3_2 != 0) ? tr_vals_4[1] : tr_vals_4[0]);
                            float _shfl_152 = __shfl_sync(0xFFFFFFFF, send_4_2, lane ^ 8);
                            float recv_6_2 = _shfl_152;
                            float _max_142 = max_noftz(keep_5_2, recv_6_2);
                            tr_vals_4[0] = _max_142;
                            float _shfl_153 = __shfl_sync(0xFFFFFFFF, tr_vals_4[0], lane ^ 4);
                            float other_4 = _shfl_153;
                            float _max_143 = max_noftz(tr_vals_4[0], other_4);
                            tr_vals_4[0] = _max_143;
                            float _shfl_154 = __shfl_sync(0xFFFFFFFF, tr_vals_4[0], lane ^ 2);
                            float other_7_2 = _shfl_154;
                            float _max_144 = max_noftz(tr_vals_4[0], other_7_2);
                            tr_vals_4[0] = _max_144;
                            float _shfl_155 = __shfl_sync(0xFFFFFFFF, tr_vals_4[0], lane ^ 1);
                            float other_8_2 = _shfl_155;
                            float _max_145 = max_noftz(tr_vals_4[0], other_8_2);
                            tr_vals_4[0] = _max_145;
                        } else {
                            tr_vals_4[4] = score_values_4[4];
                            tr_vals_4[5] = score_values_4[5];
                            tr_vals_4[6] = score_values_4[6];
                            tr_vals_4[7] = score_values_4[7];
                            int hi_bit_4 = lane & 16;
                            float send_5 = ((hi_bit_4 != 0) ? tr_vals_4[0] : tr_vals_4[4]);
                            float keep_6 = ((hi_bit_4 != 0) ? tr_vals_4[4] : tr_vals_4[0]);
                            float _shfl_270 = __shfl_sync(0xFFFFFFFF, send_5, lane ^ 16);
                            float recv_6 = _shfl_270;
                            float _max_160 = max_noftz(keep_6, recv_6);
                            tr_vals_4[0] = _max_160;
                            float send_0_4 = ((hi_bit_4 != 0) ? tr_vals_4[1] : tr_vals_4[5]);
                            float keep_1_4 = ((hi_bit_4 != 0) ? tr_vals_4[5] : tr_vals_4[1]);
                            float _shfl_271 = __shfl_sync(0xFFFFFFFF, send_0_4, lane ^ 16);
                            float recv_2_4 = _shfl_271;
                            float _max_161 = max_noftz(keep_1_4, recv_2_4);
                            tr_vals_4[1] = _max_161;
                            float send_3_4 = ((hi_bit_4 != 0) ? tr_vals_4[2] : tr_vals_4[6]);
                            float keep_4_4 = ((hi_bit_4 != 0) ? tr_vals_4[6] : tr_vals_4[2]);
                            float _shfl_272 = __shfl_sync(0xFFFFFFFF, send_3_4, lane ^ 16);
                            float recv_5_4 = _shfl_272;
                            float _max_162 = max_noftz(keep_4_4, recv_5_4);
                            tr_vals_4[2] = _max_162;
                            float send_6_4 = ((hi_bit_4 != 0) ? tr_vals_4[3] : tr_vals_4[7]);
                            float keep_7_4 = ((hi_bit_4 != 0) ? tr_vals_4[7] : tr_vals_4[3]);
                            float _shfl_273 = __shfl_sync(0xFFFFFFFF, send_6_4, lane ^ 16);
                            float recv_8_4 = _shfl_273;
                            float _max_163 = max_noftz(keep_7_4, recv_8_4);
                            tr_vals_4[3] = _max_163;
                            int hi_bit_9_2 = lane & 8;
                            float send_10_2 = ((hi_bit_9_2 != 0) ? tr_vals_4[0] : tr_vals_4[2]);
                            float keep_11_2 = ((hi_bit_9_2 != 0) ? tr_vals_4[2] : tr_vals_4[0]);
                            float _shfl_274 = __shfl_sync(0xFFFFFFFF, send_10_2, lane ^ 8);
                            float recv_12_2 = _shfl_274;
                            float _max_164 = max_noftz(keep_11_2, recv_12_2);
                            tr_vals_4[0] = _max_164;
                            float send_13_2 = ((hi_bit_9_2 != 0) ? tr_vals_4[1] : tr_vals_4[3]);
                            float keep_14_2 = ((hi_bit_9_2 != 0) ? tr_vals_4[3] : tr_vals_4[1]);
                            float _shfl_275 = __shfl_sync(0xFFFFFFFF, send_13_2, lane ^ 8);
                            float recv_15_2 = _shfl_275;
                            float _max_165 = max_noftz(keep_14_2, recv_15_2);
                            tr_vals_4[1] = _max_165;
                            int hi_bit_16_2 = lane & 4;
                            float send_17_2 = ((hi_bit_16_2 != 0) ? tr_vals_4[0] : tr_vals_4[1]);
                            float keep_18_2 = ((hi_bit_16_2 != 0) ? tr_vals_4[1] : tr_vals_4[0]);
                            float _shfl_276 = __shfl_sync(0xFFFFFFFF, send_17_2, lane ^ 4);
                            float recv_19_2 = _shfl_276;
                            float _max_166 = max_noftz(keep_18_2, recv_19_2);
                            tr_vals_4[0] = _max_166;
                            float _shfl_277 = __shfl_sync(0xFFFFFFFF, tr_vals_4[0], lane ^ 2);
                            float other_4 = _shfl_277;
                            float _max_167 = max_noftz(tr_vals_4[0], other_4);
                            tr_vals_4[0] = _max_167;
                            float _shfl_278 = __shfl_sync(0xFFFFFFFF, tr_vals_4[0], lane ^ 1);
                            float other_20_2 = _shfl_278;
                            float _max_168 = max_noftz(tr_vals_4[0], other_20_2);
                            tr_vals_4[0] = _max_168;
                        }
                        if ((lane & ((-1 * TILE_N) / 4 + 11)) == 0) {
                            smem_pmax[(8 + local_warp_2) * (TILE_N / 2) + (lane >> ((-1 * TILE_N) / 16 + 4))] = tr_vals_4[0];
                        }
                        asm volatile("barrier.sync 13, 128;" ::: "memory");
                        float m_lane_4 = -CAKE_INF;
                        float alpha_4 = 1.0f;
                        int grow_4 = 0;
                        if (lane < (TILE_N / 4)) {
                            if constexpr (TILE_N == 16) {
                                float _max_146 = max_noftz(smem_pmax[64 + lane], smem_pmax[72 + lane]);
                                float _max_147 = max_noftz(smem_pmax[80 + lane], smem_pmax[88 + lane]);
                                float _max_148 = max_noftz(_max_146, _max_147);
                                m_lane_4 = _max_148;
                            } else {
                                float _max_169 = max_noftz(smem_pmax[128 + lane], smem_pmax[144 + lane]);
                                float _max_170 = max_noftz(smem_pmax[160 + lane], smem_pmax[176 + lane]);
                                float _max_171 = max_noftz(_max_169, _max_170);
                                m_lane_4 = _max_171;
                            }
                            float cand_4 = m_lane_4 * softmax_scale_log2_2;
                            if (it_2 == 0) {
                                if constexpr (TILE_N == 16) {
                                    float _max_149 = max_noftz(cand_4, sink_lane_2);
                                    cand_4 = _max_149;
                                } else {
                                    float _max_172 = max_noftz(cand_4, sink_lane_2);
                                    cand_4 = _max_172;
                                }
                            }
                            if constexpr (TILE_N == 16) {
                                float _max_150 = max_noftz(cand_4, m_run_2);
                                cand_4 = _max_150;
                            } else {
                                float _max_173 = max_noftz(cand_4, m_run_2);
                                cand_4 = _max_173;
                            }
                            if (it_2 == 0) {
                                grow_4 = 1;
                            }
                            if (cand_4 - m_run_2 > 8.0f) {
                                grow_4 = 1;
                            }
                            if (grow_4 != 0) {
                                if constexpr (TILE_N == 16) {
                                    float _exp2_32 = approx_exp2(m_run_2 - cand_4);
                                    alpha_4 = ((m_run_2 > -CAKE_INF) ? _exp2_32 : 0.0f);
                                } else {
                                    float _exp2_56 = approx_exp2(m_run_2 - cand_4);
                                    alpha_4 = ((m_run_2 > -CAKE_INF) ? _exp2_56 : 0.0f);
                                }
                                l_run_2 = l_run_2 * alpha_4;
                                r_run_2 = r_run_2 * alpha_4;
                                m_run_2 = cand_4;
                            }
                        }
                        float m_scaled_4 = ((m_run_2 > -CAKE_INF) ? m_run_2 : 0.0f);
                        unsigned int _vote_4 = __ballot_sync(0xFFFFFFFF, grow_4 != 0);
                        unsigned int grow_bits_4 = _vote_4;
                        if (it_2 > 0) {
                            if (grow_bits_4 != 0) {
                                float alpha_c_4[(TILE_N / 4)];
                                if constexpr (TILE_N == 16) {
                                    float _shfl_156 = __shfl_sync(0xFFFFFFFF, alpha_4, 0);
                                    alpha_c_4[0] = _shfl_156;
                                    float _shfl_157 = __shfl_sync(0xFFFFFFFF, alpha_4, 1);
                                    alpha_c_4[1] = _shfl_157;
                                    float _shfl_158 = __shfl_sync(0xFFFFFFFF, alpha_4, 2);
                                    alpha_c_4[2] = _shfl_158;
                                    float _shfl_159 = __shfl_sync(0xFFFFFFFF, alpha_4, 3);
                                    alpha_c_4[3] = _shfl_159;
                                } else {
                                    float _shfl_279 = __shfl_sync(0xFFFFFFFF, alpha_4, 0);
                                    alpha_c_4[0] = _shfl_279;
                                    float _shfl_280 = __shfl_sync(0xFFFFFFFF, alpha_4, 1);
                                    alpha_c_4[1] = _shfl_280;
                                    float _shfl_281 = __shfl_sync(0xFFFFFFFF, alpha_4, 2);
                                    alpha_c_4[2] = _shfl_281;
                                    float _shfl_282 = __shfl_sync(0xFFFFFFFF, alpha_4, 3);
                                    alpha_c_4[3] = _shfl_282;
                                    float _shfl_283 = __shfl_sync(0xFFFFFFFF, alpha_4, 4);
                                    alpha_c_4[4] = _shfl_283;
                                    float _shfl_284 = __shfl_sync(0xFFFFFFFF, alpha_4, 5);
                                    alpha_c_4[5] = _shfl_284;
                                    float _shfl_285 = __shfl_sync(0xFFFFFFFF, alpha_4, 6);
                                    alpha_c_4[6] = _shfl_285;
                                    float _shfl_286 = __shfl_sync(0xFFFFFFFF, alpha_4, 7);
                                    alpha_c_4[7] = _shfl_286;
                                }
                                float ov_4[(TILE_N / 4)];
                                if constexpr (TILE_N == 16) {
                                    tmem_ld_x4(&ov_4[0], taddr + 16 + 12 + (unsigned int)(tmem_row_origin_2 << 16));
                                } else {
                                    tmem_ld_x8(&ov_4[0], taddr + 32 + 24 + (unsigned int)(tmem_row_origin_2 << 16));
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                ov_4[0] = ov_4[0] * alpha_c_4[0];
                                ov_4[1] = ov_4[1] * alpha_c_4[1];
                                ov_4[2] = ov_4[2] * alpha_c_4[2];
                                ov_4[3] = ov_4[3] * alpha_c_4[3];
                                if constexpr (TILE_N == 16) {
                                    tmem_st_x4_f32(taddr + 16 + 12 + (unsigned int)(tmem_row_origin_2 << 16), ov_4);
                                    tmem_ld_x4(&ov_4[0], taddr + 32 + 12 + (unsigned int)(tmem_row_origin_2 << 16));
                                } else {
                                    ov_4[4] = ov_4[4] * alpha_c_4[4];
                                    ov_4[5] = ov_4[5] * alpha_c_4[5];
                                    ov_4[6] = ov_4[6] * alpha_c_4[6];
                                    ov_4[7] = ov_4[7] * alpha_c_4[7];
                                    tmem_st_x8_f32(taddr + 32 + 24 + (unsigned int)(tmem_row_origin_2 << 16), ov_4);
                                    tmem_ld_x8(&ov_4[0], taddr + 64 + 24 + (unsigned int)(tmem_row_origin_2 << 16));
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                ov_4[0] = ov_4[0] * alpha_c_4[0];
                                ov_4[1] = ov_4[1] * alpha_c_4[1];
                                ov_4[2] = ov_4[2] * alpha_c_4[2];
                                ov_4[3] = ov_4[3] * alpha_c_4[3];
                                if constexpr (TILE_N == 16) {
                                    tmem_st_x4_f32(taddr + 32 + 12 + (unsigned int)(tmem_row_origin_2 << 16), ov_4);
                                    tmem_ld_x4(&ov_4[0], taddr + 48 + 12 + (unsigned int)(tmem_row_origin_2 << 16));
                                } else {
                                    ov_4[4] = ov_4[4] * alpha_c_4[4];
                                    ov_4[5] = ov_4[5] * alpha_c_4[5];
                                    ov_4[6] = ov_4[6] * alpha_c_4[6];
                                    ov_4[7] = ov_4[7] * alpha_c_4[7];
                                    tmem_st_x8_f32(taddr + 64 + 24 + (unsigned int)(tmem_row_origin_2 << 16), ov_4);
                                    tmem_ld_x8(&ov_4[0], taddr + 96 + 24 + (unsigned int)(tmem_row_origin_2 << 16));
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                ov_4[0] = ov_4[0] * alpha_c_4[0];
                                ov_4[1] = ov_4[1] * alpha_c_4[1];
                                ov_4[2] = ov_4[2] * alpha_c_4[2];
                                ov_4[3] = ov_4[3] * alpha_c_4[3];
                                if constexpr (TILE_N == 16) {
                                    tmem_st_x4_f32(taddr + 48 + 12 + (unsigned int)(tmem_row_origin_2 << 16), ov_4);
                                    tmem_ld_x4(&ov_4[0], taddr + 64 + 12 + (unsigned int)(tmem_row_origin_2 << 16));
                                } else {
                                    ov_4[4] = ov_4[4] * alpha_c_4[4];
                                    ov_4[5] = ov_4[5] * alpha_c_4[5];
                                    ov_4[6] = ov_4[6] * alpha_c_4[6];
                                    ov_4[7] = ov_4[7] * alpha_c_4[7];
                                    tmem_st_x8_f32(taddr + 96 + 24 + (unsigned int)(tmem_row_origin_2 << 16), ov_4);
                                    tmem_ld_x8(&ov_4[0], taddr + 128 + 24 + (unsigned int)(tmem_row_origin_2 << 16));
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                ov_4[0] = ov_4[0] * alpha_c_4[0];
                                ov_4[1] = ov_4[1] * alpha_c_4[1];
                                ov_4[2] = ov_4[2] * alpha_c_4[2];
                                ov_4[3] = ov_4[3] * alpha_c_4[3];
                                if constexpr (TILE_N == 16) {
                                    tmem_st_x4_f32(taddr + 64 + 12 + (unsigned int)(tmem_row_origin_2 << 16), ov_4);
                                } else {
                                    ov_4[4] = ov_4[4] * alpha_c_4[4];
                                    ov_4[5] = ov_4[5] * alpha_c_4[5];
                                    ov_4[6] = ov_4[6] * alpha_c_4[6];
                                    ov_4[7] = ov_4[7] * alpha_c_4[7];
                                    tmem_st_x8_f32(taddr + 128 + 24 + (unsigned int)(tmem_row_origin_2 << 16), ov_4);
                                }
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                        }
                        float col_max_4[(TILE_N / 4)];
                        if constexpr (TILE_N == 16) {
                            float _shfl_160 = __shfl_sync(0xFFFFFFFF, m_scaled_4, 0);
                            col_max_4[0] = _shfl_160;
                            float _shfl_161 = __shfl_sync(0xFFFFFFFF, m_scaled_4, 1);
                            col_max_4[1] = _shfl_161;
                            float _shfl_162 = __shfl_sync(0xFFFFFFFF, m_scaled_4, 2);
                            col_max_4[2] = _shfl_162;
                            float _shfl_163 = __shfl_sync(0xFFFFFFFF, m_scaled_4, 3);
                            col_max_4[3] = _shfl_163;
                            float _exp2_33 = approx_exp2(score_values_4[0] * softmax_scale_log2_2 - col_max_4[0]);
                            score_values_4[0] = _exp2_33;
                            float _exp2_34 = approx_exp2(score_values_4[1] * softmax_scale_log2_2 - col_max_4[1]);
                            score_values_4[1] = _exp2_34;
                            float _exp2_35 = approx_exp2(score_values_4[2] * softmax_scale_log2_2 - col_max_4[2]);
                            score_values_4[2] = _exp2_35;
                            float _exp2_36 = approx_exp2(score_values_4[3] * softmax_scale_log2_2 - col_max_4[3]);
                            score_values_4[3] = _exp2_36;
                        } else {
                            float _shfl_287 = __shfl_sync(0xFFFFFFFF, m_scaled_4, 0);
                            col_max_4[0] = _shfl_287;
                            float _shfl_288 = __shfl_sync(0xFFFFFFFF, m_scaled_4, 1);
                            col_max_4[1] = _shfl_288;
                            float _shfl_289 = __shfl_sync(0xFFFFFFFF, m_scaled_4, 2);
                            col_max_4[2] = _shfl_289;
                            float _shfl_290 = __shfl_sync(0xFFFFFFFF, m_scaled_4, 3);
                            col_max_4[3] = _shfl_290;
                            float _shfl_291 = __shfl_sync(0xFFFFFFFF, m_scaled_4, 4);
                            col_max_4[4] = _shfl_291;
                            float _shfl_292 = __shfl_sync(0xFFFFFFFF, m_scaled_4, 5);
                            col_max_4[5] = _shfl_292;
                            float _shfl_293 = __shfl_sync(0xFFFFFFFF, m_scaled_4, 6);
                            col_max_4[6] = _shfl_293;
                            float _shfl_294 = __shfl_sync(0xFFFFFFFF, m_scaled_4, 7);
                            col_max_4[7] = _shfl_294;
                            float _exp2_57 = approx_exp2(score_values_4[0] * softmax_scale_log2_2 - col_max_4[0]);
                            score_values_4[0] = _exp2_57;
                            float _exp2_58 = approx_exp2(score_values_4[1] * softmax_scale_log2_2 - col_max_4[1]);
                            score_values_4[1] = _exp2_58;
                            float _exp2_59 = approx_exp2(score_values_4[2] * softmax_scale_log2_2 - col_max_4[2]);
                            score_values_4[2] = _exp2_59;
                            float _exp2_60 = approx_exp2(score_values_4[3] * softmax_scale_log2_2 - col_max_4[3]);
                            score_values_4[3] = _exp2_60;
                            float _exp2_61 = approx_exp2(score_values_4[4] * softmax_scale_log2_2 - col_max_4[4]);
                            score_values_4[4] = _exp2_61;
                            float _exp2_62 = approx_exp2(score_values_4[5] * softmax_scale_log2_2 - col_max_4[5]);
                            score_values_4[5] = _exp2_62;
                            float _exp2_63 = approx_exp2(score_values_4[6] * softmax_scale_log2_2 - col_max_4[6]);
                            score_values_4[6] = _exp2_63;
                            float _exp2_64 = approx_exp2(score_values_4[7] * softmax_scale_log2_2 - col_max_4[7]);
                            score_values_4[7] = _exp2_64;
                        }
                        {
                            uint16_t _fp8_pair_26;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                : "=h"(_fp8_pair_26) : "f"(0.0f), "f"(score_values_4[0]));
                            uint32_t _byte_26 = (uint32_t)(_fp8_pair_26 & 0xFF);
                            uint32_t _addr_26 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(96 * TILE_N + row_2 ^ (96 * TILE_N + row_2 >> 7 & 7) << 4)));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_26), "r"(_byte_26) : "memory");
                        }
                        float _fp8_rt_24;
                        float _fp8_rt_48;
                        uint16_t _e4m3x2_27;
                        uint32_t _f16x2_27;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_27) : "f"(0.0f), "f"(score_values_4[0]));
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_27) : "h"(_e4m3x2_27));
                        uint16_t _fp8_h0_27 = (uint16_t)(_f16x2_27 & 0xFFFFu);
                        if constexpr (TILE_N == 16) {
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_24) : "h"(_fp8_h0_27));
                            tr_vals_4[0] = _fp8_rt_24;
                        } else {
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_48) : "h"(_fp8_h0_27));
                            tr_vals_4[0] = _fp8_rt_48;
                        }
                        {
                            uint16_t _fp8_pair_28;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                : "=h"(_fp8_pair_28) : "f"(0.0f), "f"(score_values_4[1]));
                            uint32_t _byte_28 = (uint32_t)(_fp8_pair_28 & 0xFF);
                            uint32_t _addr_28 = static_cast<uint32_t>((smem_p_addr + (unsigned int)((96 * TILE_N + 128) + row_2 ^ ((96 * TILE_N + 128) + row_2 >> 7 & 7) << 4)));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_28), "r"(_byte_28) : "memory");
                        }
                        float _fp8_rt_25;
                        float _fp8_rt_49;
                        uint16_t _e4m3x2_29;
                        uint32_t _f16x2_29;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_29) : "f"(0.0f), "f"(score_values_4[1]));
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_29) : "h"(_e4m3x2_29));
                        uint16_t _fp8_h0_29 = (uint16_t)(_f16x2_29 & 0xFFFFu);
                        if constexpr (TILE_N == 16) {
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_25) : "h"(_fp8_h0_29));
                            tr_vals_4[1] = _fp8_rt_25;
                        } else {
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_49) : "h"(_fp8_h0_29));
                            tr_vals_4[1] = _fp8_rt_49;
                        }
                        {
                            uint16_t _fp8_pair_30;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                : "=h"(_fp8_pair_30) : "f"(0.0f), "f"(score_values_4[2]));
                            uint32_t _byte_30 = (uint32_t)(_fp8_pair_30 & 0xFF);
                            uint32_t _addr_30 = static_cast<uint32_t>((smem_p_addr + (unsigned int)((96 * TILE_N + 256) + row_2 ^ ((96 * TILE_N + 256) + row_2 >> 7 & 7) << 4)));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_30), "r"(_byte_30) : "memory");
                        }
                        float _fp8_rt_26;
                        float _fp8_rt_50;
                        uint16_t _e4m3x2_31;
                        uint32_t _f16x2_31;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_31) : "f"(0.0f), "f"(score_values_4[2]));
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_31) : "h"(_e4m3x2_31));
                        uint16_t _fp8_h0_31 = (uint16_t)(_f16x2_31 & 0xFFFFu);
                        if constexpr (TILE_N == 16) {
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_26) : "h"(_fp8_h0_31));
                            tr_vals_4[2] = _fp8_rt_26;
                        } else {
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_50) : "h"(_fp8_h0_31));
                            tr_vals_4[2] = _fp8_rt_50;
                        }
                        {
                            uint16_t _fp8_pair_32;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                : "=h"(_fp8_pair_32) : "f"(0.0f), "f"(score_values_4[3]));
                            uint32_t _byte_32 = (uint32_t)(_fp8_pair_32 & 0xFF);
                            uint32_t _addr_32 = static_cast<uint32_t>((smem_p_addr + (unsigned int)((96 * TILE_N + 384) + row_2 ^ ((96 * TILE_N + 384) + row_2 >> 7 & 7) << 4)));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_32), "r"(_byte_32) : "memory");
                        }
                        float _fp8_rt_27;
                        float _fp8_rt_51;
                        uint16_t _e4m3x2_33;
                        uint32_t _f16x2_33;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_33) : "f"(0.0f), "f"(score_values_4[3]));
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_33) : "h"(_e4m3x2_33));
                        uint16_t _fp8_h0_33 = (uint16_t)(_f16x2_33 & 0xFFFFu);
                        if constexpr (TILE_N == 16) {
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_27) : "h"(_fp8_h0_33));
                            tr_vals_4[3] = _fp8_rt_27;
                            int hi_bit_9_4 = lane & 16;
                            float send_10_4 = ((hi_bit_9_4 != 0) ? score_values_4[0] : score_values_4[2]);
                            float keep_11_4 = ((hi_bit_9_4 != 0) ? score_values_4[2] : score_values_4[0]);
                            float _shfl_164 = __shfl_sync(0xFFFFFFFF, send_10_4, lane ^ 16);
                            float recv_12_4 = _shfl_164;
                            score_values_4[0] = keep_11_4 + recv_12_4;
                            float send_13_4 = ((hi_bit_9_4 != 0) ? score_values_4[1] : score_values_4[3]);
                            float keep_14_4 = ((hi_bit_9_4 != 0) ? score_values_4[3] : score_values_4[1]);
                            float _shfl_165 = __shfl_sync(0xFFFFFFFF, send_13_4, lane ^ 16);
                            float recv_15_4 = _shfl_165;
                            score_values_4[1] = keep_14_4 + recv_15_4;
                            int hi_bit_16_4 = lane & 8;
                            float send_17_4 = ((hi_bit_16_4 != 0) ? score_values_4[0] : score_values_4[1]);
                            float keep_18_4 = ((hi_bit_16_4 != 0) ? score_values_4[1] : score_values_4[0]);
                            float _shfl_166 = __shfl_sync(0xFFFFFFFF, send_17_4, lane ^ 8);
                            float recv_19_4 = _shfl_166;
                            score_values_4[0] = keep_18_4 + recv_19_4;
                            float _shfl_167 = __shfl_sync(0xFFFFFFFF, score_values_4[0], lane ^ 4);
                            float other_20_4 = _shfl_167;
                            score_values_4[0] = score_values_4[0] + other_20_4;
                            float _shfl_168 = __shfl_sync(0xFFFFFFFF, score_values_4[0], lane ^ 2);
                            float other_21_2 = _shfl_168;
                            score_values_4[0] = score_values_4[0] + other_21_2;
                            float _shfl_169 = __shfl_sync(0xFFFFFFFF, score_values_4[0], lane ^ 1);
                            float other_22_2 = _shfl_169;
                            score_values_4[0] = score_values_4[0] + other_22_2;
                            int hi_bit_23_2 = lane & 16;
                            float send_24_2 = ((hi_bit_23_2 != 0) ? tr_vals_4[0] : tr_vals_4[2]);
                            float keep_25_2 = ((hi_bit_23_2 != 0) ? tr_vals_4[2] : tr_vals_4[0]);
                            float _shfl_170 = __shfl_sync(0xFFFFFFFF, send_24_2, lane ^ 16);
                            float recv_26_2 = _shfl_170;
                            tr_vals_4[0] = keep_25_2 + recv_26_2;
                            float send_27_2 = ((hi_bit_23_2 != 0) ? tr_vals_4[1] : tr_vals_4[3]);
                            float keep_28_2 = ((hi_bit_23_2 != 0) ? tr_vals_4[3] : tr_vals_4[1]);
                            float _shfl_171 = __shfl_sync(0xFFFFFFFF, send_27_2, lane ^ 16);
                            float recv_29_2 = _shfl_171;
                            tr_vals_4[1] = keep_28_2 + recv_29_2;
                            int hi_bit_30_2 = lane & 8;
                            float send_31_4 = ((hi_bit_30_2 != 0) ? tr_vals_4[0] : tr_vals_4[1]);
                            float keep_32_4 = ((hi_bit_30_2 != 0) ? tr_vals_4[1] : tr_vals_4[0]);
                            float _shfl_172 = __shfl_sync(0xFFFFFFFF, send_31_4, lane ^ 8);
                            float recv_33_4 = _shfl_172;
                            tr_vals_4[0] = keep_32_4 + recv_33_4;
                            float _shfl_173 = __shfl_sync(0xFFFFFFFF, tr_vals_4[0], lane ^ 4);
                            float other_34_2 = _shfl_173;
                            tr_vals_4[0] = tr_vals_4[0] + other_34_2;
                            float _shfl_174 = __shfl_sync(0xFFFFFFFF, tr_vals_4[0], lane ^ 2);
                            float other_35_2 = _shfl_174;
                            tr_vals_4[0] = tr_vals_4[0] + other_35_2;
                            float _shfl_175 = __shfl_sync(0xFFFFFFFF, tr_vals_4[0], lane ^ 1);
                            float other_36_2 = _shfl_175;
                            tr_vals_4[0] = tr_vals_4[0] + other_36_2;
                        } else {
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_51) : "h"(_fp8_h0_33));
                            tr_vals_4[3] = _fp8_rt_51;
                            {
                                uint16_t _fp8_pair_34;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_34) : "f"(0.0f), "f"(score_values_4[4]));
                                uint32_t _byte_34 = (uint32_t)(_fp8_pair_34 & 0xFF);
                                uint32_t _addr_34 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(3584 + row_2 ^ (3584 + row_2 >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_34), "r"(_byte_34) : "memory");
                            }
                            float _fp8_rt_52;
                            uint16_t _e4m3x2_35;
                            uint32_t _f16x2_35;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_35) : "f"(0.0f), "f"(score_values_4[4]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_35) : "h"(_e4m3x2_35));
                            uint16_t _fp8_h0_35 = (uint16_t)(_f16x2_35 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_52) : "h"(_fp8_h0_35));
                            tr_vals_4[4] = _fp8_rt_52;
                            {
                                uint16_t _fp8_pair_36;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_36) : "f"(0.0f), "f"(score_values_4[5]));
                                uint32_t _byte_36 = (uint32_t)(_fp8_pair_36 & 0xFF);
                                uint32_t _addr_36 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(3712 + row_2 ^ (3712 + row_2 >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_36), "r"(_byte_36) : "memory");
                            }
                            float _fp8_rt_53;
                            uint16_t _e4m3x2_37;
                            uint32_t _f16x2_37;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_37) : "f"(0.0f), "f"(score_values_4[5]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_37) : "h"(_e4m3x2_37));
                            uint16_t _fp8_h0_37 = (uint16_t)(_f16x2_37 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_53) : "h"(_fp8_h0_37));
                            tr_vals_4[5] = _fp8_rt_53;
                            {
                                uint16_t _fp8_pair_38;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_38) : "f"(0.0f), "f"(score_values_4[6]));
                                uint32_t _byte_38 = (uint32_t)(_fp8_pair_38 & 0xFF);
                                uint32_t _addr_38 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(3840 + row_2 ^ (3840 + row_2 >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_38), "r"(_byte_38) : "memory");
                            }
                            float _fp8_rt_54;
                            uint16_t _e4m3x2_39;
                            uint32_t _f16x2_39;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_39) : "f"(0.0f), "f"(score_values_4[6]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_39) : "h"(_e4m3x2_39));
                            uint16_t _fp8_h0_39 = (uint16_t)(_f16x2_39 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_54) : "h"(_fp8_h0_39));
                            tr_vals_4[6] = _fp8_rt_54;
                            {
                                uint16_t _fp8_pair_40;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_40) : "f"(0.0f), "f"(score_values_4[7]));
                                uint32_t _byte_40 = (uint32_t)(_fp8_pair_40 & 0xFF);
                                uint32_t _addr_40 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(3968 + row_2 ^ (3968 + row_2 >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_40), "r"(_byte_40) : "memory");
                            }
                            float _fp8_rt_55;
                            uint16_t _e4m3x2_41;
                            uint32_t _f16x2_41;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_41) : "f"(0.0f), "f"(score_values_4[7]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_41) : "h"(_e4m3x2_41));
                            uint16_t _fp8_h0_41 = (uint16_t)(_f16x2_41 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_55) : "h"(_fp8_h0_41));
                            tr_vals_4[7] = _fp8_rt_55;
                            int hi_bit_21_4 = lane & 16;
                            float send_22_4 = ((hi_bit_21_4 != 0) ? score_values_4[0] : score_values_4[4]);
                            float keep_23_4 = ((hi_bit_21_4 != 0) ? score_values_4[4] : score_values_4[0]);
                            float _shfl_295 = __shfl_sync(0xFFFFFFFF, send_22_4, lane ^ 16);
                            float recv_24_4 = _shfl_295;
                            score_values_4[0] = keep_23_4 + recv_24_4;
                            float send_25_4 = ((hi_bit_21_4 != 0) ? score_values_4[1] : score_values_4[5]);
                            float keep_26_4 = ((hi_bit_21_4 != 0) ? score_values_4[5] : score_values_4[1]);
                            float _shfl_296 = __shfl_sync(0xFFFFFFFF, send_25_4, lane ^ 16);
                            float recv_27_4 = _shfl_296;
                            score_values_4[1] = keep_26_4 + recv_27_4;
                            float send_28_4 = ((hi_bit_21_4 != 0) ? score_values_4[2] : score_values_4[6]);
                            float keep_29_4 = ((hi_bit_21_4 != 0) ? score_values_4[6] : score_values_4[2]);
                            float _shfl_297 = __shfl_sync(0xFFFFFFFF, send_28_4, lane ^ 16);
                            float recv_30_4 = _shfl_297;
                            score_values_4[2] = keep_29_4 + recv_30_4;
                            float send_31_4 = ((hi_bit_21_4 != 0) ? score_values_4[3] : score_values_4[7]);
                            float keep_32_4 = ((hi_bit_21_4 != 0) ? score_values_4[7] : score_values_4[3]);
                            float _shfl_298 = __shfl_sync(0xFFFFFFFF, send_31_4, lane ^ 16);
                            float recv_33_4 = _shfl_298;
                            score_values_4[3] = keep_32_4 + recv_33_4;
                            int hi_bit_34_4 = lane & 8;
                            float send_35_4 = ((hi_bit_34_4 != 0) ? score_values_4[0] : score_values_4[2]);
                            float keep_36_4 = ((hi_bit_34_4 != 0) ? score_values_4[2] : score_values_4[0]);
                            float _shfl_299 = __shfl_sync(0xFFFFFFFF, send_35_4, lane ^ 8);
                            float recv_37_4 = _shfl_299;
                            score_values_4[0] = keep_36_4 + recv_37_4;
                            float send_38_4 = ((hi_bit_34_4 != 0) ? score_values_4[1] : score_values_4[3]);
                            float keep_39_4 = ((hi_bit_34_4 != 0) ? score_values_4[3] : score_values_4[1]);
                            float _shfl_300 = __shfl_sync(0xFFFFFFFF, send_38_4, lane ^ 8);
                            float recv_40_4 = _shfl_300;
                            score_values_4[1] = keep_39_4 + recv_40_4;
                            int hi_bit_41_4 = lane & 4;
                            float send_42_4 = ((hi_bit_41_4 != 0) ? score_values_4[0] : score_values_4[1]);
                            float keep_43_4 = ((hi_bit_41_4 != 0) ? score_values_4[1] : score_values_4[0]);
                            float _shfl_301 = __shfl_sync(0xFFFFFFFF, send_42_4, lane ^ 4);
                            float recv_44_4 = _shfl_301;
                            score_values_4[0] = keep_43_4 + recv_44_4;
                            float _shfl_302 = __shfl_sync(0xFFFFFFFF, score_values_4[0], lane ^ 2);
                            float other_45_2 = _shfl_302;
                            score_values_4[0] = score_values_4[0] + other_45_2;
                            float _shfl_303 = __shfl_sync(0xFFFFFFFF, score_values_4[0], lane ^ 1);
                            float other_46_2 = _shfl_303;
                            score_values_4[0] = score_values_4[0] + other_46_2;
                            int hi_bit_47_2 = lane & 16;
                            float send_48_2 = ((hi_bit_47_2 != 0) ? tr_vals_4[0] : tr_vals_4[4]);
                            float keep_49_2 = ((hi_bit_47_2 != 0) ? tr_vals_4[4] : tr_vals_4[0]);
                            float _shfl_304 = __shfl_sync(0xFFFFFFFF, send_48_2, lane ^ 16);
                            float recv_50_2 = _shfl_304;
                            tr_vals_4[0] = keep_49_2 + recv_50_2;
                            float send_51_2 = ((hi_bit_47_2 != 0) ? tr_vals_4[1] : tr_vals_4[5]);
                            float keep_52_2 = ((hi_bit_47_2 != 0) ? tr_vals_4[5] : tr_vals_4[1]);
                            float _shfl_305 = __shfl_sync(0xFFFFFFFF, send_51_2, lane ^ 16);
                            float recv_53_2 = _shfl_305;
                            tr_vals_4[1] = keep_52_2 + recv_53_2;
                            float send_54_2 = ((hi_bit_47_2 != 0) ? tr_vals_4[2] : tr_vals_4[6]);
                            float keep_55_2 = ((hi_bit_47_2 != 0) ? tr_vals_4[6] : tr_vals_4[2]);
                            float _shfl_306 = __shfl_sync(0xFFFFFFFF, send_54_2, lane ^ 16);
                            float recv_56_2 = _shfl_306;
                            tr_vals_4[2] = keep_55_2 + recv_56_2;
                            float send_57_2 = ((hi_bit_47_2 != 0) ? tr_vals_4[3] : tr_vals_4[7]);
                            float keep_58_2 = ((hi_bit_47_2 != 0) ? tr_vals_4[7] : tr_vals_4[3]);
                            float _shfl_307 = __shfl_sync(0xFFFFFFFF, send_57_2, lane ^ 16);
                            float recv_59_2 = _shfl_307;
                            tr_vals_4[3] = keep_58_2 + recv_59_2;
                            int hi_bit_60_2 = lane & 8;
                            float send_61_4 = ((hi_bit_60_2 != 0) ? tr_vals_4[0] : tr_vals_4[2]);
                            float keep_62_4 = ((hi_bit_60_2 != 0) ? tr_vals_4[2] : tr_vals_4[0]);
                            float _shfl_308 = __shfl_sync(0xFFFFFFFF, send_61_4, lane ^ 8);
                            float recv_63_4 = _shfl_308;
                            tr_vals_4[0] = keep_62_4 + recv_63_4;
                            float send_64_4 = ((hi_bit_60_2 != 0) ? tr_vals_4[1] : tr_vals_4[3]);
                            float keep_65_4 = ((hi_bit_60_2 != 0) ? tr_vals_4[3] : tr_vals_4[1]);
                            float _shfl_309 = __shfl_sync(0xFFFFFFFF, send_64_4, lane ^ 8);
                            float recv_66_4 = _shfl_309;
                            tr_vals_4[1] = keep_65_4 + recv_66_4;
                            int hi_bit_67_2 = lane & 4;
                            float send_68_2 = ((hi_bit_67_2 != 0) ? tr_vals_4[0] : tr_vals_4[1]);
                            float keep_69_2 = ((hi_bit_67_2 != 0) ? tr_vals_4[1] : tr_vals_4[0]);
                            float _shfl_310 = __shfl_sync(0xFFFFFFFF, send_68_2, lane ^ 4);
                            float recv_70_2 = _shfl_310;
                            tr_vals_4[0] = keep_69_2 + recv_70_2;
                            float _shfl_311 = __shfl_sync(0xFFFFFFFF, tr_vals_4[0], lane ^ 2);
                            float other_71_2 = _shfl_311;
                            tr_vals_4[0] = tr_vals_4[0] + other_71_2;
                            float _shfl_312 = __shfl_sync(0xFFFFFFFF, tr_vals_4[0], lane ^ 1);
                            float other_72_2 = _shfl_312;
                            tr_vals_4[0] = tr_vals_4[0] + other_72_2;
                        }
                        if ((lane & ((-1 * TILE_N) / 4 + 11)) == 0) {
                            smem_psum[(8 + local_warp_2) * (TILE_N / 2) + (lane >> ((-1 * TILE_N) / 16 + 4))] = score_values_4[0];
                            smem_rsum[(8 + local_warp_2) * (TILE_N / 2) + (lane >> ((-1 * TILE_N) / 16 + 4))] = tr_vals_4[0];
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(p_full_addr);
                        asm volatile("barrier.sync 13, 128;" ::: "memory");
                        if (lane < (TILE_N / 4)) {
                            float tile_sum_4 = smem_psum[4 * TILE_N + lane] + smem_psum[((9 * TILE_N) / 2) + lane] + smem_psum[5 * TILE_N + lane] + smem_psum[((11 * TILE_N) / 2) + lane];
                            float tile_rsum_4 = smem_rsum[4 * TILE_N + lane] + smem_rsum[((9 * TILE_N) / 2) + lane] + smem_rsum[5 * TILE_N + lane] + smem_rsum[((11 * TILE_N) / 2) + lane];
                            if (it_2 == 0) {
                                float sink_term_4;
                                if constexpr (TILE_N == 16) {
                                    float _exp2_37 = approx_exp2(sink_lane_2 - m_scaled_4);
                                    sink_term_4 = _exp2_37;
                                } else {
                                    float _exp2_65 = approx_exp2(sink_lane_2 - m_scaled_4);
                                    sink_term_4 = _exp2_65;
                                }
                                l_run_2 = sink_term_4;
                                r_run_2 = sink_term_4;
                            }
                            l_run_2 = l_run_2 + tile_sum_4;
                            r_run_2 = r_run_2 + tile_rsum_4;
                        }
                        asm volatile("barrier.sync 13, 128;" ::: "memory");
                    }
                }
                int it_0_2 = 2 * it2_2 + 1;
                if (it_0_2 < tiles_per_split) {
                    mbarrier_wait_hint(tok_full_addr + 8, it2_2 & 1, 10000000);
                    int raw_index_5 = smem_tok_1[row_2];
                    int valid_5 = 1;
                    if (raw_index_5 < 0) {
                        valid_5 = 0;
                    }
                    if (valid_5 != 0) {
                    } else if (0) {
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(kv_full_addr);
                    if (it_0_2 > 0) {
                        mbarrier_wait_hint(o_full_addr, 0, 10000000);
                        mbarrier_wait_hint(o_full_addr + 8, 0, 10000000);
                        mbarrier_wait_hint(o_full_addr + 16, 0, 10000000);
                        mbarrier_wait_hint(o_full_addr + 24, 0, 10000000);
                    }
                    if (valid_5 != 0) {
                        unsigned int kraw4_5[4];
                        unsigned int sfw32_5 = 0;
                        int vblock_9 = 32 * o_chunk_2 + 22;
                        unsigned int v8_10[4];
                        {
                            {
                                int vchunk_27 = 16 * o_chunk_2 + 11;
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&kraw4_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_5[(0) + 3]))
                                    : "r"(smem_kf4_1_addr + (unsigned int)(vchunk_27 / 8 * 16384) + (unsigned int)(row_2 * 128 + (vchunk_27 % 8 * 16 ^ row_2 % 8 * 16))));
                            }
                            {
                                sfw32_5 = smem_sfs32_1[row_2 * 8 + 8 * o_chunk_2 + 5];
                            }
                            unsigned int scale_50 = sfw32_5 >> 16 & 255;
                            {
                                v8_10[0] = cake_dsv4_qmul4_portable<5>(kraw4_5[0], scale_50);
                            }
                            {
                                v8_10[1] = cake_dsv4_qmul4_portable<6>(kraw4_5[0], scale_50);
                            }
                            {
                                v8_10[2] = cake_dsv4_qmul4_portable<5>(kraw4_5[1], scale_50);
                            }
                            {
                                v8_10[3] = cake_dsv4_qmul4_portable<6>(kraw4_5[1], scale_50);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(32768 + (row_2 * 128 + (96 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_10[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_10[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_10[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_10[(0) + 3])));
                        int vblock_0_5 = 32 * o_chunk_2 + 23;
                        unsigned int v8_1_5[4];
                        {
                            unsigned int scale_51 = sfw32_5 >> 24 & 255;
                            {
                                v8_1_5[0] = cake_dsv4_qmul4_portable<5>(kraw4_5[2], scale_51);
                            }
                            {
                                v8_1_5[1] = cake_dsv4_qmul4_portable<6>(kraw4_5[2], scale_51);
                            }
                            {
                                v8_1_5[2] = cake_dsv4_qmul4_portable<5>(kraw4_5[3], scale_51);
                            }
                            {
                                v8_1_5[3] = cake_dsv4_qmul4_portable<6>(kraw4_5[3], scale_51);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(32768 + (row_2 * 128 + (112 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_1_5[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_1_5[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_1_5[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_1_5[(0) + 3])));
                        int vblock_2_5 = 32 * o_chunk_2 + 24;
                        unsigned int v8_3_5[4];
                        {
                            {
                                int vchunk_28 = 16 * o_chunk_2 + 12;
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&kraw4_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_5[(0) + 3]))
                                    : "r"(smem_kf4_1_addr + (unsigned int)(vchunk_28 / 8 * 16384) + (unsigned int)(row_2 * 128 + (vchunk_28 % 8 * 16 ^ row_2 % 8 * 16))));
                            }
                            {
                                sfw32_5 = smem_sfs32_1[row_2 * 8 + 8 * o_chunk_2 + 6];
                            }
                            unsigned int scale_52 = sfw32_5 & 255;
                            {
                                v8_3_5[0] = cake_dsv4_qmul4_portable<5>(kraw4_5[0], scale_52);
                            }
                            {
                                v8_3_5[1] = cake_dsv4_qmul4_portable<6>(kraw4_5[0], scale_52);
                            }
                            {
                                v8_3_5[2] = cake_dsv4_qmul4_portable<5>(kraw4_5[1], scale_52);
                            }
                            {
                                v8_3_5[3] = cake_dsv4_qmul4_portable<6>(kraw4_5[1], scale_52);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (0 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_3_5[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_3_5[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_3_5[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_3_5[(0) + 3])));
                        int vblock_4_5 = 32 * o_chunk_2 + 25;
                        unsigned int v8_5_5[4];
                        {
                            unsigned int scale_53 = sfw32_5 >> 8 & 255;
                            {
                                v8_5_5[0] = cake_dsv4_qmul4_portable<5>(kraw4_5[2], scale_53);
                            }
                            {
                                v8_5_5[1] = cake_dsv4_qmul4_portable<6>(kraw4_5[2], scale_53);
                            }
                            {
                                v8_5_5[2] = cake_dsv4_qmul4_portable<5>(kraw4_5[3], scale_53);
                            }
                            {
                                v8_5_5[3] = cake_dsv4_qmul4_portable<6>(kraw4_5[3], scale_53);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (16 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_5_5[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_5_5[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_5_5[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_5_5[(0) + 3])));
                        int vblock_6_5 = 32 * o_chunk_2 + 26;
                        unsigned int v8_7_5[4];
                        {
                            {
                                int vchunk_29 = 16 * o_chunk_2 + 13;
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&kraw4_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw4_5[(0) + 3]))
                                    : "r"(smem_kf4_1_addr + (unsigned int)(vchunk_29 / 8 * 16384) + (unsigned int)(row_2 * 128 + (vchunk_29 % 8 * 16 ^ row_2 % 8 * 16))));
                            }
                            unsigned int scale_54 = sfw32_5 >> 16 & 255;
                            {
                                v8_7_5[0] = cake_dsv4_qmul4_portable<5>(kraw4_5[0], scale_54);
                            }
                            {
                                v8_7_5[1] = cake_dsv4_qmul4_portable<6>(kraw4_5[0], scale_54);
                            }
                            {
                                v8_7_5[2] = cake_dsv4_qmul4_portable<5>(kraw4_5[1], scale_54);
                            }
                            {
                                v8_7_5[3] = cake_dsv4_qmul4_portable<6>(kraw4_5[1], scale_54);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (32 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_7_5[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_7_5[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_7_5[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_7_5[(0) + 3])));
                        int vblock_8_5 = 32 * o_chunk_2 + 27;
                        unsigned int v8_9_5[4];
                        {
                            unsigned int scale_55 = sfw32_5 >> 24 & 255;
                            {
                                v8_9_5[0] = cake_dsv4_qmul4_portable<5>(kraw4_5[2], scale_55);
                            }
                            {
                                v8_9_5[1] = cake_dsv4_qmul4_portable<6>(kraw4_5[2], scale_55);
                            }
                            {
                                v8_9_5[2] = cake_dsv4_qmul4_portable<5>(kraw4_5[3], scale_55);
                            }
                            {
                                v8_9_5[3] = cake_dsv4_qmul4_portable<6>(kraw4_5[3], scale_55);
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (48 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_9_5[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_9_5[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_9_5[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_9_5[(0) + 3])));
                        int vblock_10_5 = 32 * o_chunk_2 + 28;
                        unsigned int v8_11_5[4];
                        {
                            int rblock_4 = vblock_10_5 - 28;
                            unsigned int rope_4[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&rope_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_4[(0) + 3]))
                                : "r"(smem_krope_1_addr + (unsigned int)(row_2 * 128 + (2 * rblock_4 * 16 ^ row_2 % 8 * 16))));
                            float lo_5 = __uint_as_float(rope_4[0] << 16);
                            float hi_6 = __uint_as_float(rope_4[0] & 4294901760u);
                            uint16_t _e4m3x2_f32_966;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_966) : "f"(hi_6), "f"(lo_5));
                            uint16_t pair_6 = _e4m3x2_f32_966;
                            {
                                v8_11_5[0] = (unsigned int)pair_6;
                            }
                            float lo_0_4 = __uint_as_float(rope_4[1] << 16);
                            float hi_1_4 = __uint_as_float(rope_4[1] & 4294901760u);
                            uint16_t _e4m3x2_f32_967;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_967) : "f"(hi_1_4), "f"(lo_0_4));
                            uint16_t pair_2_4 = _e4m3x2_f32_967;
                            {
                                v8_11_5[0] = v8_11_5[0] | (unsigned int)pair_2_4 << 16;
                            }
                            float lo_3_4 = __uint_as_float(rope_4[2] << 16);
                            float hi_4_4 = __uint_as_float(rope_4[2] & 4294901760u);
                            uint16_t _e4m3x2_f32_968;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_968) : "f"(hi_4_4), "f"(lo_3_4));
                            uint16_t pair_5_4 = _e4m3x2_f32_968;
                            {
                                v8_11_5[1] = (unsigned int)pair_5_4;
                            }
                            float lo_6_4 = __uint_as_float(rope_4[3] << 16);
                            float hi_7_4 = __uint_as_float(rope_4[3] & 4294901760u);
                            uint16_t _e4m3x2_f32_969;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_969) : "f"(hi_7_4), "f"(lo_6_4));
                            uint16_t pair_8_4 = _e4m3x2_f32_969;
                            {
                                v8_11_5[1] = v8_11_5[1] | (unsigned int)pair_8_4 << 16;
                            }
                            unsigned int rope_9_4[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&rope_9_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_4[(0) + 3]))
                                : "r"(smem_krope_1_addr + (unsigned int)(row_2 * 128 + ((2 * rblock_4 + 1) * 16 ^ row_2 % 8 * 16))));
                            float lo_10_4 = __uint_as_float(rope_9_4[0] << 16);
                            float hi_11_4 = __uint_as_float(rope_9_4[0] & 4294901760u);
                            uint16_t _e4m3x2_f32_970;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_970) : "f"(hi_11_4), "f"(lo_10_4));
                            uint16_t pair_12_4 = _e4m3x2_f32_970;
                            {
                                v8_11_5[2] = (unsigned int)pair_12_4;
                            }
                            float lo_13_4 = __uint_as_float(rope_9_4[1] << 16);
                            float hi_14_4 = __uint_as_float(rope_9_4[1] & 4294901760u);
                            uint16_t _e4m3x2_f32_971;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_971) : "f"(hi_14_4), "f"(lo_13_4));
                            uint16_t pair_15_4 = _e4m3x2_f32_971;
                            {
                                v8_11_5[2] = v8_11_5[2] | (unsigned int)pair_15_4 << 16;
                            }
                            float lo_16_4 = __uint_as_float(rope_9_4[2] << 16);
                            float hi_17_4 = __uint_as_float(rope_9_4[2] & 4294901760u);
                            uint16_t _e4m3x2_f32_972;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_972) : "f"(hi_17_4), "f"(lo_16_4));
                            uint16_t pair_18_4 = _e4m3x2_f32_972;
                            {
                                v8_11_5[3] = (unsigned int)pair_18_4;
                            }
                            float lo_19_4 = __uint_as_float(rope_9_4[3] << 16);
                            float hi_20_4 = __uint_as_float(rope_9_4[3] & 4294901760u);
                            uint16_t _e4m3x2_f32_973;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_973) : "f"(hi_20_4), "f"(lo_19_4));
                            uint16_t pair_21_4 = _e4m3x2_f32_973;
                            {
                                v8_11_5[3] = v8_11_5[3] | (unsigned int)pair_21_4 << 16;
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (64 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_11_5[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_11_5[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_11_5[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_11_5[(0) + 3])));
                        int vblock_12_5 = 32 * o_chunk_2 + 29;
                        unsigned int v8_13_5[4];
                        {
                            int rblock_5 = vblock_12_5 - 28;
                            unsigned int rope_5[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&rope_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_5[(0) + 3]))
                                : "r"(smem_krope_1_addr + (unsigned int)(row_2 * 128 + (2 * rblock_5 * 16 ^ row_2 % 8 * 16))));
                            float lo_7 = __uint_as_float(rope_5[0] << 16);
                            float hi_8 = __uint_as_float(rope_5[0] & 4294901760u);
                            uint16_t _e4m3x2_f32_982;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_982) : "f"(hi_8), "f"(lo_7));
                            uint16_t pair_7 = _e4m3x2_f32_982;
                            {
                                v8_13_5[0] = (unsigned int)pair_7;
                            }
                            float lo_0_5 = __uint_as_float(rope_5[1] << 16);
                            float hi_1_5 = __uint_as_float(rope_5[1] & 4294901760u);
                            uint16_t _e4m3x2_f32_983;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_983) : "f"(hi_1_5), "f"(lo_0_5));
                            uint16_t pair_2_5 = _e4m3x2_f32_983;
                            {
                                v8_13_5[0] = v8_13_5[0] | (unsigned int)pair_2_5 << 16;
                            }
                            float lo_3_5 = __uint_as_float(rope_5[2] << 16);
                            float hi_4_5 = __uint_as_float(rope_5[2] & 4294901760u);
                            uint16_t _e4m3x2_f32_984;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_984) : "f"(hi_4_5), "f"(lo_3_5));
                            uint16_t pair_5_5 = _e4m3x2_f32_984;
                            {
                                v8_13_5[1] = (unsigned int)pair_5_5;
                            }
                            float lo_6_5 = __uint_as_float(rope_5[3] << 16);
                            float hi_7_5 = __uint_as_float(rope_5[3] & 4294901760u);
                            uint16_t _e4m3x2_f32_985;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_985) : "f"(hi_7_5), "f"(lo_6_5));
                            uint16_t pair_8_5 = _e4m3x2_f32_985;
                            {
                                v8_13_5[1] = v8_13_5[1] | (unsigned int)pair_8_5 << 16;
                            }
                            unsigned int rope_9_5[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&rope_9_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_5[(0) + 3]))
                                : "r"(smem_krope_1_addr + (unsigned int)(row_2 * 128 + ((2 * rblock_5 + 1) * 16 ^ row_2 % 8 * 16))));
                            float lo_10_5 = __uint_as_float(rope_9_5[0] << 16);
                            float hi_11_5 = __uint_as_float(rope_9_5[0] & 4294901760u);
                            uint16_t _e4m3x2_f32_986;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_986) : "f"(hi_11_5), "f"(lo_10_5));
                            uint16_t pair_12_5 = _e4m3x2_f32_986;
                            {
                                v8_13_5[2] = (unsigned int)pair_12_5;
                            }
                            float lo_13_5 = __uint_as_float(rope_9_5[1] << 16);
                            float hi_14_5 = __uint_as_float(rope_9_5[1] & 4294901760u);
                            uint16_t _e4m3x2_f32_987;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_987) : "f"(hi_14_5), "f"(lo_13_5));
                            uint16_t pair_15_5 = _e4m3x2_f32_987;
                            {
                                v8_13_5[2] = v8_13_5[2] | (unsigned int)pair_15_5 << 16;
                            }
                            float lo_16_5 = __uint_as_float(rope_9_5[2] << 16);
                            float hi_17_5 = __uint_as_float(rope_9_5[2] & 4294901760u);
                            uint16_t _e4m3x2_f32_988;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_988) : "f"(hi_17_5), "f"(lo_16_5));
                            uint16_t pair_18_5 = _e4m3x2_f32_988;
                            {
                                v8_13_5[3] = (unsigned int)pair_18_5;
                            }
                            float lo_19_5 = __uint_as_float(rope_9_5[3] << 16);
                            float hi_20_5 = __uint_as_float(rope_9_5[3] & 4294901760u);
                            uint16_t _e4m3x2_f32_989;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_989) : "f"(hi_20_5), "f"(lo_19_5));
                            uint16_t pair_21_5 = _e4m3x2_f32_989;
                            {
                                v8_13_5[3] = v8_13_5[3] | (unsigned int)pair_21_5 << 16;
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (80 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_13_5[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_13_5[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_13_5[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_13_5[(0) + 3])));
                        int vblock_14_5 = 32 * o_chunk_2 + 30;
                        unsigned int v8_15_5[4];
                        {
                            int rblock_6 = vblock_14_5 - 28;
                            unsigned int rope_6[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&rope_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_6[(0) + 3]))
                                : "r"(smem_krope_1_addr + (unsigned int)(row_2 * 128 + (2 * rblock_6 * 16 ^ row_2 % 8 * 16))));
                            float lo_8 = __uint_as_float(rope_6[0] << 16);
                            float hi_9 = __uint_as_float(rope_6[0] & 4294901760u);
                            uint16_t _e4m3x2_f32_998;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_998) : "f"(hi_9), "f"(lo_8));
                            uint16_t pair_9 = _e4m3x2_f32_998;
                            {
                                v8_15_5[0] = (unsigned int)pair_9;
                            }
                            float lo_0_6 = __uint_as_float(rope_6[1] << 16);
                            float hi_1_6 = __uint_as_float(rope_6[1] & 4294901760u);
                            uint16_t _e4m3x2_f32_999;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_999) : "f"(hi_1_6), "f"(lo_0_6));
                            uint16_t pair_2_6 = _e4m3x2_f32_999;
                            {
                                v8_15_5[0] = v8_15_5[0] | (unsigned int)pair_2_6 << 16;
                            }
                            float lo_3_6 = __uint_as_float(rope_6[2] << 16);
                            float hi_4_6 = __uint_as_float(rope_6[2] & 4294901760u);
                            uint16_t _e4m3x2_f32_1000;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1000) : "f"(hi_4_6), "f"(lo_3_6));
                            uint16_t pair_5_6 = _e4m3x2_f32_1000;
                            {
                                v8_15_5[1] = (unsigned int)pair_5_6;
                            }
                            float lo_6_6 = __uint_as_float(rope_6[3] << 16);
                            float hi_7_6 = __uint_as_float(rope_6[3] & 4294901760u);
                            uint16_t _e4m3x2_f32_1001;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1001) : "f"(hi_7_6), "f"(lo_6_6));
                            uint16_t pair_8_6 = _e4m3x2_f32_1001;
                            {
                                v8_15_5[1] = v8_15_5[1] | (unsigned int)pair_8_6 << 16;
                            }
                            unsigned int rope_9_6[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&rope_9_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_6[(0) + 3]))
                                : "r"(smem_krope_1_addr + (unsigned int)(row_2 * 128 + ((2 * rblock_6 + 1) * 16 ^ row_2 % 8 * 16))));
                            float lo_10_6 = __uint_as_float(rope_9_6[0] << 16);
                            float hi_11_6 = __uint_as_float(rope_9_6[0] & 4294901760u);
                            uint16_t _e4m3x2_f32_1002;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1002) : "f"(hi_11_6), "f"(lo_10_6));
                            uint16_t pair_12_6 = _e4m3x2_f32_1002;
                            {
                                v8_15_5[2] = (unsigned int)pair_12_6;
                            }
                            float lo_13_6 = __uint_as_float(rope_9_6[1] << 16);
                            float hi_14_6 = __uint_as_float(rope_9_6[1] & 4294901760u);
                            uint16_t _e4m3x2_f32_1003;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1003) : "f"(hi_14_6), "f"(lo_13_6));
                            uint16_t pair_15_6 = _e4m3x2_f32_1003;
                            {
                                v8_15_5[2] = v8_15_5[2] | (unsigned int)pair_15_6 << 16;
                            }
                            float lo_16_6 = __uint_as_float(rope_9_6[2] << 16);
                            float hi_17_6 = __uint_as_float(rope_9_6[2] & 4294901760u);
                            uint16_t _e4m3x2_f32_1004;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1004) : "f"(hi_17_6), "f"(lo_16_6));
                            uint16_t pair_18_6 = _e4m3x2_f32_1004;
                            {
                                v8_15_5[3] = (unsigned int)pair_18_6;
                            }
                            float lo_19_6 = __uint_as_float(rope_9_6[3] << 16);
                            float hi_20_6 = __uint_as_float(rope_9_6[3] & 4294901760u);
                            uint16_t _e4m3x2_f32_1005;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1005) : "f"(hi_20_6), "f"(lo_19_6));
                            uint16_t pair_21_6 = _e4m3x2_f32_1005;
                            {
                                v8_15_5[3] = v8_15_5[3] | (unsigned int)pair_21_6 << 16;
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (96 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_15_5[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_15_5[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_15_5[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_15_5[(0) + 3])));
                        int vblock_16_5 = 32 * o_chunk_2 + 31;
                        unsigned int v8_17_5[4];
                        {
                            int rblock_7 = vblock_16_5 - 28;
                            unsigned int rope_7[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&rope_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_7[(0) + 3]))
                                : "r"(smem_krope_1_addr + (unsigned int)(row_2 * 128 + (2 * rblock_7 * 16 ^ row_2 % 8 * 16))));
                            float lo_9 = __uint_as_float(rope_7[0] << 16);
                            float hi_10 = __uint_as_float(rope_7[0] & 4294901760u);
                            uint16_t _e4m3x2_f32_1014;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1014) : "f"(hi_10), "f"(lo_9));
                            uint16_t pair_10 = _e4m3x2_f32_1014;
                            {
                                v8_17_5[0] = (unsigned int)pair_10;
                            }
                            float lo_0_7 = __uint_as_float(rope_7[1] << 16);
                            float hi_1_7 = __uint_as_float(rope_7[1] & 4294901760u);
                            uint16_t _e4m3x2_f32_1015;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1015) : "f"(hi_1_7), "f"(lo_0_7));
                            uint16_t pair_2_7 = _e4m3x2_f32_1015;
                            {
                                v8_17_5[0] = v8_17_5[0] | (unsigned int)pair_2_7 << 16;
                            }
                            float lo_3_7 = __uint_as_float(rope_7[2] << 16);
                            float hi_4_7 = __uint_as_float(rope_7[2] & 4294901760u);
                            uint16_t _e4m3x2_f32_1016;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1016) : "f"(hi_4_7), "f"(lo_3_7));
                            uint16_t pair_5_7 = _e4m3x2_f32_1016;
                            {
                                v8_17_5[1] = (unsigned int)pair_5_7;
                            }
                            float lo_6_7 = __uint_as_float(rope_7[3] << 16);
                            float hi_7_7 = __uint_as_float(rope_7[3] & 4294901760u);
                            uint16_t _e4m3x2_f32_1017;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1017) : "f"(hi_7_7), "f"(lo_6_7));
                            uint16_t pair_8_7 = _e4m3x2_f32_1017;
                            {
                                v8_17_5[1] = v8_17_5[1] | (unsigned int)pair_8_7 << 16;
                            }
                            unsigned int rope_9_7[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&rope_9_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_7[(0) + 3]))
                                : "r"(smem_krope_1_addr + (unsigned int)(row_2 * 128 + ((2 * rblock_7 + 1) * 16 ^ row_2 % 8 * 16))));
                            float lo_10_7 = __uint_as_float(rope_9_7[0] << 16);
                            float hi_11_7 = __uint_as_float(rope_9_7[0] & 4294901760u);
                            uint16_t _e4m3x2_f32_1018;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1018) : "f"(hi_11_7), "f"(lo_10_7));
                            uint16_t pair_12_7 = _e4m3x2_f32_1018;
                            {
                                v8_17_5[2] = (unsigned int)pair_12_7;
                            }
                            float lo_13_7 = __uint_as_float(rope_9_7[1] << 16);
                            float hi_14_7 = __uint_as_float(rope_9_7[1] & 4294901760u);
                            uint16_t _e4m3x2_f32_1019;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1019) : "f"(hi_14_7), "f"(lo_13_7));
                            uint16_t pair_15_7 = _e4m3x2_f32_1019;
                            {
                                v8_17_5[2] = v8_17_5[2] | (unsigned int)pair_15_7 << 16;
                            }
                            float lo_16_7 = __uint_as_float(rope_9_7[2] << 16);
                            float hi_17_7 = __uint_as_float(rope_9_7[2] & 4294901760u);
                            uint16_t _e4m3x2_f32_1020;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1020) : "f"(hi_17_7), "f"(lo_16_7));
                            uint16_t pair_18_7 = _e4m3x2_f32_1020;
                            {
                                v8_17_5[3] = (unsigned int)pair_18_7;
                            }
                            float lo_19_7 = __uint_as_float(rope_9_7[3] << 16);
                            float hi_20_7 = __uint_as_float(rope_9_7[3] & 4294901760u);
                            uint16_t _e4m3x2_f32_1021;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1021) : "f"(hi_20_7), "f"(lo_19_7));
                            uint16_t pair_21_7 = _e4m3x2_f32_1021;
                            {
                                v8_17_5[3] = v8_17_5[3] | (unsigned int)pair_21_7 << 16;
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (112 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_17_5[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_17_5[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_17_5[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_17_5[(0) + 3])));
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
                    mbarrier_wait_hint(s_full_addr, 1, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (it_0_2 + 2 < tiles_per_split) {
                        mbarrier_arrive(tok_free_addr + 8);
                    }
                    {
                        float score_values_5[(TILE_N / 4)];
                        if constexpr (TILE_N == 16) {
                            tmem_ld_x4(&score_values_5[0], taddr + 12 + (unsigned int)(tmem_row_origin_2 << 16));
                        } else {
                            tmem_ld_x8(&score_values_5[0], taddr + 24 + (unsigned int)(tmem_row_origin_2 << 16));
                        }
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        if (valid_5 == 0) {
                            score_values_5[0] = -CAKE_INF;
                            score_values_5[1] = -CAKE_INF;
                            score_values_5[2] = -CAKE_INF;
                            score_values_5[3] = -CAKE_INF;
                            if constexpr (TILE_N == 32) {
                                score_values_5[4] = -CAKE_INF;
                                score_values_5[5] = -CAKE_INF;
                                score_values_5[6] = -CAKE_INF;
                                score_values_5[7] = -CAKE_INF;
                            }
                        }
                        float tr_vals_5[(TILE_N / 4)];
                        tr_vals_5[0] = score_values_5[0];
                        tr_vals_5[1] = score_values_5[1];
                        tr_vals_5[2] = score_values_5[2];
                        tr_vals_5[3] = score_values_5[3];
                        if constexpr (TILE_N == 16) {
                            int hi_bit_6 = lane & 16;
                            float send_8 = ((hi_bit_6 != 0) ? tr_vals_5[0] : tr_vals_5[2]);
                            float keep_9 = ((hi_bit_6 != 0) ? tr_vals_5[2] : tr_vals_5[0]);
                            float _shfl_176 = __shfl_sync(0xFFFFFFFF, send_8, lane ^ 16);
                            float recv_9 = _shfl_176;
                            float _max_151 = max_noftz(keep_9, recv_9);
                            tr_vals_5[0] = _max_151;
                            float send_0_5 = ((hi_bit_6 != 0) ? tr_vals_5[1] : tr_vals_5[3]);
                            float keep_1_5 = ((hi_bit_6 != 0) ? tr_vals_5[3] : tr_vals_5[1]);
                            float _shfl_177 = __shfl_sync(0xFFFFFFFF, send_0_5, lane ^ 16);
                            float recv_2_5 = _shfl_177;
                            float _max_152 = max_noftz(keep_1_5, recv_2_5);
                            tr_vals_5[1] = _max_152;
                            int hi_bit_3_3 = lane & 8;
                            float send_4_3 = ((hi_bit_3_3 != 0) ? tr_vals_5[0] : tr_vals_5[1]);
                            float keep_5_3 = ((hi_bit_3_3 != 0) ? tr_vals_5[1] : tr_vals_5[0]);
                            float _shfl_178 = __shfl_sync(0xFFFFFFFF, send_4_3, lane ^ 8);
                            float recv_6_3 = _shfl_178;
                            float _max_153 = max_noftz(keep_5_3, recv_6_3);
                            tr_vals_5[0] = _max_153;
                            float _shfl_179 = __shfl_sync(0xFFFFFFFF, tr_vals_5[0], lane ^ 4);
                            float other_5 = _shfl_179;
                            float _max_154 = max_noftz(tr_vals_5[0], other_5);
                            tr_vals_5[0] = _max_154;
                            float _shfl_180 = __shfl_sync(0xFFFFFFFF, tr_vals_5[0], lane ^ 2);
                            float other_7_3 = _shfl_180;
                            float _max_155 = max_noftz(tr_vals_5[0], other_7_3);
                            tr_vals_5[0] = _max_155;
                            float _shfl_181 = __shfl_sync(0xFFFFFFFF, tr_vals_5[0], lane ^ 1);
                            float other_8_3 = _shfl_181;
                            float _max_156 = max_noftz(tr_vals_5[0], other_8_3);
                            tr_vals_5[0] = _max_156;
                        } else {
                            tr_vals_5[4] = score_values_5[4];
                            tr_vals_5[5] = score_values_5[5];
                            tr_vals_5[6] = score_values_5[6];
                            tr_vals_5[7] = score_values_5[7];
                            int hi_bit_5 = lane & 16;
                            float send_7 = ((hi_bit_5 != 0) ? tr_vals_5[0] : tr_vals_5[4]);
                            float keep_8 = ((hi_bit_5 != 0) ? tr_vals_5[4] : tr_vals_5[0]);
                            float _shfl_313 = __shfl_sync(0xFFFFFFFF, send_7, lane ^ 16);
                            float recv_7 = _shfl_313;
                            float _max_174 = max_noftz(keep_8, recv_7);
                            tr_vals_5[0] = _max_174;
                            float send_0_5 = ((hi_bit_5 != 0) ? tr_vals_5[1] : tr_vals_5[5]);
                            float keep_1_5 = ((hi_bit_5 != 0) ? tr_vals_5[5] : tr_vals_5[1]);
                            float _shfl_314 = __shfl_sync(0xFFFFFFFF, send_0_5, lane ^ 16);
                            float recv_2_5 = _shfl_314;
                            float _max_175 = max_noftz(keep_1_5, recv_2_5);
                            tr_vals_5[1] = _max_175;
                            float send_3_5 = ((hi_bit_5 != 0) ? tr_vals_5[2] : tr_vals_5[6]);
                            float keep_4_5 = ((hi_bit_5 != 0) ? tr_vals_5[6] : tr_vals_5[2]);
                            float _shfl_315 = __shfl_sync(0xFFFFFFFF, send_3_5, lane ^ 16);
                            float recv_5_5 = _shfl_315;
                            float _max_176 = max_noftz(keep_4_5, recv_5_5);
                            tr_vals_5[2] = _max_176;
                            float send_6_5 = ((hi_bit_5 != 0) ? tr_vals_5[3] : tr_vals_5[7]);
                            float keep_7_5 = ((hi_bit_5 != 0) ? tr_vals_5[7] : tr_vals_5[3]);
                            float _shfl_316 = __shfl_sync(0xFFFFFFFF, send_6_5, lane ^ 16);
                            float recv_8_5 = _shfl_316;
                            float _max_177 = max_noftz(keep_7_5, recv_8_5);
                            tr_vals_5[3] = _max_177;
                            int hi_bit_9_3 = lane & 8;
                            float send_10_3 = ((hi_bit_9_3 != 0) ? tr_vals_5[0] : tr_vals_5[2]);
                            float keep_11_3 = ((hi_bit_9_3 != 0) ? tr_vals_5[2] : tr_vals_5[0]);
                            float _shfl_317 = __shfl_sync(0xFFFFFFFF, send_10_3, lane ^ 8);
                            float recv_12_3 = _shfl_317;
                            float _max_178 = max_noftz(keep_11_3, recv_12_3);
                            tr_vals_5[0] = _max_178;
                            float send_13_3 = ((hi_bit_9_3 != 0) ? tr_vals_5[1] : tr_vals_5[3]);
                            float keep_14_3 = ((hi_bit_9_3 != 0) ? tr_vals_5[3] : tr_vals_5[1]);
                            float _shfl_318 = __shfl_sync(0xFFFFFFFF, send_13_3, lane ^ 8);
                            float recv_15_3 = _shfl_318;
                            float _max_179 = max_noftz(keep_14_3, recv_15_3);
                            tr_vals_5[1] = _max_179;
                            int hi_bit_16_3 = lane & 4;
                            float send_17_3 = ((hi_bit_16_3 != 0) ? tr_vals_5[0] : tr_vals_5[1]);
                            float keep_18_3 = ((hi_bit_16_3 != 0) ? tr_vals_5[1] : tr_vals_5[0]);
                            float _shfl_319 = __shfl_sync(0xFFFFFFFF, send_17_3, lane ^ 4);
                            float recv_19_3 = _shfl_319;
                            float _max_180 = max_noftz(keep_18_3, recv_19_3);
                            tr_vals_5[0] = _max_180;
                            float _shfl_320 = __shfl_sync(0xFFFFFFFF, tr_vals_5[0], lane ^ 2);
                            float other_5 = _shfl_320;
                            float _max_181 = max_noftz(tr_vals_5[0], other_5);
                            tr_vals_5[0] = _max_181;
                            float _shfl_321 = __shfl_sync(0xFFFFFFFF, tr_vals_5[0], lane ^ 1);
                            float other_20_3 = _shfl_321;
                            float _max_182 = max_noftz(tr_vals_5[0], other_20_3);
                            tr_vals_5[0] = _max_182;
                        }
                        if ((lane & ((-1 * TILE_N) / 4 + 11)) == 0) {
                            smem_pmax[(8 + local_warp_2) * (TILE_N / 2) + (lane >> ((-1 * TILE_N) / 16 + 4))] = tr_vals_5[0];
                        }
                        asm volatile("barrier.sync 13, 128;" ::: "memory");
                        float m_lane_5 = -CAKE_INF;
                        float alpha_5 = 1.0f;
                        int grow_5 = 0;
                        if (lane < (TILE_N / 4)) {
                            if constexpr (TILE_N == 16) {
                                float _max_157 = max_noftz(smem_pmax[64 + lane], smem_pmax[72 + lane]);
                                float _max_158 = max_noftz(smem_pmax[80 + lane], smem_pmax[88 + lane]);
                                float _max_159 = max_noftz(_max_157, _max_158);
                                m_lane_5 = _max_159;
                            } else {
                                float _max_183 = max_noftz(smem_pmax[128 + lane], smem_pmax[144 + lane]);
                                float _max_184 = max_noftz(smem_pmax[160 + lane], smem_pmax[176 + lane]);
                                float _max_185 = max_noftz(_max_183, _max_184);
                                m_lane_5 = _max_185;
                            }
                            float cand_5 = m_lane_5 * softmax_scale_log2_2;
                            if (it_0_2 == 0) {
                                if constexpr (TILE_N == 16) {
                                    float _max_160 = max_noftz(cand_5, sink_lane_2);
                                    cand_5 = _max_160;
                                } else {
                                    float _max_186 = max_noftz(cand_5, sink_lane_2);
                                    cand_5 = _max_186;
                                }
                            }
                            if constexpr (TILE_N == 16) {
                                float _max_161 = max_noftz(cand_5, m_run_2);
                                cand_5 = _max_161;
                            } else {
                                float _max_187 = max_noftz(cand_5, m_run_2);
                                cand_5 = _max_187;
                            }
                            if (it_0_2 == 0) {
                                grow_5 = 1;
                            }
                            if (cand_5 - m_run_2 > 8.0f) {
                                grow_5 = 1;
                            }
                            if (grow_5 != 0) {
                                if constexpr (TILE_N == 16) {
                                    float _exp2_38 = approx_exp2(m_run_2 - cand_5);
                                    alpha_5 = ((m_run_2 > -CAKE_INF) ? _exp2_38 : 0.0f);
                                } else {
                                    float _exp2_66 = approx_exp2(m_run_2 - cand_5);
                                    alpha_5 = ((m_run_2 > -CAKE_INF) ? _exp2_66 : 0.0f);
                                }
                                l_run_2 = l_run_2 * alpha_5;
                                r_run_2 = r_run_2 * alpha_5;
                                m_run_2 = cand_5;
                            }
                        }
                        float m_scaled_5 = ((m_run_2 > -CAKE_INF) ? m_run_2 : 0.0f);
                        unsigned int _vote_5 = __ballot_sync(0xFFFFFFFF, grow_5 != 0);
                        unsigned int grow_bits_5 = _vote_5;
                        if (it_0_2 > 0) {
                            if (grow_bits_5 != 0) {
                                float alpha_c_5[(TILE_N / 4)];
                                if constexpr (TILE_N == 16) {
                                    float _shfl_182 = __shfl_sync(0xFFFFFFFF, alpha_5, 0);
                                    alpha_c_5[0] = _shfl_182;
                                    float _shfl_183 = __shfl_sync(0xFFFFFFFF, alpha_5, 1);
                                    alpha_c_5[1] = _shfl_183;
                                    float _shfl_184 = __shfl_sync(0xFFFFFFFF, alpha_5, 2);
                                    alpha_c_5[2] = _shfl_184;
                                    float _shfl_185 = __shfl_sync(0xFFFFFFFF, alpha_5, 3);
                                    alpha_c_5[3] = _shfl_185;
                                } else {
                                    float _shfl_322 = __shfl_sync(0xFFFFFFFF, alpha_5, 0);
                                    alpha_c_5[0] = _shfl_322;
                                    float _shfl_323 = __shfl_sync(0xFFFFFFFF, alpha_5, 1);
                                    alpha_c_5[1] = _shfl_323;
                                    float _shfl_324 = __shfl_sync(0xFFFFFFFF, alpha_5, 2);
                                    alpha_c_5[2] = _shfl_324;
                                    float _shfl_325 = __shfl_sync(0xFFFFFFFF, alpha_5, 3);
                                    alpha_c_5[3] = _shfl_325;
                                    float _shfl_326 = __shfl_sync(0xFFFFFFFF, alpha_5, 4);
                                    alpha_c_5[4] = _shfl_326;
                                    float _shfl_327 = __shfl_sync(0xFFFFFFFF, alpha_5, 5);
                                    alpha_c_5[5] = _shfl_327;
                                    float _shfl_328 = __shfl_sync(0xFFFFFFFF, alpha_5, 6);
                                    alpha_c_5[6] = _shfl_328;
                                    float _shfl_329 = __shfl_sync(0xFFFFFFFF, alpha_5, 7);
                                    alpha_c_5[7] = _shfl_329;
                                }
                                float ov_5[(TILE_N / 4)];
                                if constexpr (TILE_N == 16) {
                                    tmem_ld_x4(&ov_5[0], taddr + 16 + 12 + (unsigned int)(tmem_row_origin_2 << 16));
                                } else {
                                    tmem_ld_x8(&ov_5[0], taddr + 32 + 24 + (unsigned int)(tmem_row_origin_2 << 16));
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                ov_5[0] = ov_5[0] * alpha_c_5[0];
                                ov_5[1] = ov_5[1] * alpha_c_5[1];
                                ov_5[2] = ov_5[2] * alpha_c_5[2];
                                ov_5[3] = ov_5[3] * alpha_c_5[3];
                                if constexpr (TILE_N == 16) {
                                    tmem_st_x4_f32(taddr + 16 + 12 + (unsigned int)(tmem_row_origin_2 << 16), ov_5);
                                    tmem_ld_x4(&ov_5[0], taddr + 32 + 12 + (unsigned int)(tmem_row_origin_2 << 16));
                                } else {
                                    ov_5[4] = ov_5[4] * alpha_c_5[4];
                                    ov_5[5] = ov_5[5] * alpha_c_5[5];
                                    ov_5[6] = ov_5[6] * alpha_c_5[6];
                                    ov_5[7] = ov_5[7] * alpha_c_5[7];
                                    tmem_st_x8_f32(taddr + 32 + 24 + (unsigned int)(tmem_row_origin_2 << 16), ov_5);
                                    tmem_ld_x8(&ov_5[0], taddr + 64 + 24 + (unsigned int)(tmem_row_origin_2 << 16));
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                ov_5[0] = ov_5[0] * alpha_c_5[0];
                                ov_5[1] = ov_5[1] * alpha_c_5[1];
                                ov_5[2] = ov_5[2] * alpha_c_5[2];
                                ov_5[3] = ov_5[3] * alpha_c_5[3];
                                if constexpr (TILE_N == 16) {
                                    tmem_st_x4_f32(taddr + 32 + 12 + (unsigned int)(tmem_row_origin_2 << 16), ov_5);
                                    tmem_ld_x4(&ov_5[0], taddr + 48 + 12 + (unsigned int)(tmem_row_origin_2 << 16));
                                } else {
                                    ov_5[4] = ov_5[4] * alpha_c_5[4];
                                    ov_5[5] = ov_5[5] * alpha_c_5[5];
                                    ov_5[6] = ov_5[6] * alpha_c_5[6];
                                    ov_5[7] = ov_5[7] * alpha_c_5[7];
                                    tmem_st_x8_f32(taddr + 64 + 24 + (unsigned int)(tmem_row_origin_2 << 16), ov_5);
                                    tmem_ld_x8(&ov_5[0], taddr + 96 + 24 + (unsigned int)(tmem_row_origin_2 << 16));
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                ov_5[0] = ov_5[0] * alpha_c_5[0];
                                ov_5[1] = ov_5[1] * alpha_c_5[1];
                                ov_5[2] = ov_5[2] * alpha_c_5[2];
                                ov_5[3] = ov_5[3] * alpha_c_5[3];
                                if constexpr (TILE_N == 16) {
                                    tmem_st_x4_f32(taddr + 48 + 12 + (unsigned int)(tmem_row_origin_2 << 16), ov_5);
                                    tmem_ld_x4(&ov_5[0], taddr + 64 + 12 + (unsigned int)(tmem_row_origin_2 << 16));
                                } else {
                                    ov_5[4] = ov_5[4] * alpha_c_5[4];
                                    ov_5[5] = ov_5[5] * alpha_c_5[5];
                                    ov_5[6] = ov_5[6] * alpha_c_5[6];
                                    ov_5[7] = ov_5[7] * alpha_c_5[7];
                                    tmem_st_x8_f32(taddr + 96 + 24 + (unsigned int)(tmem_row_origin_2 << 16), ov_5);
                                    tmem_ld_x8(&ov_5[0], taddr + 128 + 24 + (unsigned int)(tmem_row_origin_2 << 16));
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                ov_5[0] = ov_5[0] * alpha_c_5[0];
                                ov_5[1] = ov_5[1] * alpha_c_5[1];
                                ov_5[2] = ov_5[2] * alpha_c_5[2];
                                ov_5[3] = ov_5[3] * alpha_c_5[3];
                                if constexpr (TILE_N == 16) {
                                    tmem_st_x4_f32(taddr + 64 + 12 + (unsigned int)(tmem_row_origin_2 << 16), ov_5);
                                } else {
                                    ov_5[4] = ov_5[4] * alpha_c_5[4];
                                    ov_5[5] = ov_5[5] * alpha_c_5[5];
                                    ov_5[6] = ov_5[6] * alpha_c_5[6];
                                    ov_5[7] = ov_5[7] * alpha_c_5[7];
                                    tmem_st_x8_f32(taddr + 128 + 24 + (unsigned int)(tmem_row_origin_2 << 16), ov_5);
                                }
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                        }
                        float col_max_5[(TILE_N / 4)];
                        if constexpr (TILE_N == 16) {
                            float _shfl_186 = __shfl_sync(0xFFFFFFFF, m_scaled_5, 0);
                            col_max_5[0] = _shfl_186;
                            float _shfl_187 = __shfl_sync(0xFFFFFFFF, m_scaled_5, 1);
                            col_max_5[1] = _shfl_187;
                            float _shfl_188 = __shfl_sync(0xFFFFFFFF, m_scaled_5, 2);
                            col_max_5[2] = _shfl_188;
                            float _shfl_189 = __shfl_sync(0xFFFFFFFF, m_scaled_5, 3);
                            col_max_5[3] = _shfl_189;
                            float _exp2_39 = approx_exp2(score_values_5[0] * softmax_scale_log2_2 - col_max_5[0]);
                            score_values_5[0] = _exp2_39;
                            float _exp2_40 = approx_exp2(score_values_5[1] * softmax_scale_log2_2 - col_max_5[1]);
                            score_values_5[1] = _exp2_40;
                            float _exp2_41 = approx_exp2(score_values_5[2] * softmax_scale_log2_2 - col_max_5[2]);
                            score_values_5[2] = _exp2_41;
                            float _exp2_42 = approx_exp2(score_values_5[3] * softmax_scale_log2_2 - col_max_5[3]);
                            score_values_5[3] = _exp2_42;
                        } else {
                            float _shfl_330 = __shfl_sync(0xFFFFFFFF, m_scaled_5, 0);
                            col_max_5[0] = _shfl_330;
                            float _shfl_331 = __shfl_sync(0xFFFFFFFF, m_scaled_5, 1);
                            col_max_5[1] = _shfl_331;
                            float _shfl_332 = __shfl_sync(0xFFFFFFFF, m_scaled_5, 2);
                            col_max_5[2] = _shfl_332;
                            float _shfl_333 = __shfl_sync(0xFFFFFFFF, m_scaled_5, 3);
                            col_max_5[3] = _shfl_333;
                            float _shfl_334 = __shfl_sync(0xFFFFFFFF, m_scaled_5, 4);
                            col_max_5[4] = _shfl_334;
                            float _shfl_335 = __shfl_sync(0xFFFFFFFF, m_scaled_5, 5);
                            col_max_5[5] = _shfl_335;
                            float _shfl_336 = __shfl_sync(0xFFFFFFFF, m_scaled_5, 6);
                            col_max_5[6] = _shfl_336;
                            float _shfl_337 = __shfl_sync(0xFFFFFFFF, m_scaled_5, 7);
                            col_max_5[7] = _shfl_337;
                            float _exp2_67 = approx_exp2(score_values_5[0] * softmax_scale_log2_2 - col_max_5[0]);
                            score_values_5[0] = _exp2_67;
                            float _exp2_68 = approx_exp2(score_values_5[1] * softmax_scale_log2_2 - col_max_5[1]);
                            score_values_5[1] = _exp2_68;
                            float _exp2_69 = approx_exp2(score_values_5[2] * softmax_scale_log2_2 - col_max_5[2]);
                            score_values_5[2] = _exp2_69;
                            float _exp2_70 = approx_exp2(score_values_5[3] * softmax_scale_log2_2 - col_max_5[3]);
                            score_values_5[3] = _exp2_70;
                            float _exp2_71 = approx_exp2(score_values_5[4] * softmax_scale_log2_2 - col_max_5[4]);
                            score_values_5[4] = _exp2_71;
                            float _exp2_72 = approx_exp2(score_values_5[5] * softmax_scale_log2_2 - col_max_5[5]);
                            score_values_5[5] = _exp2_72;
                            float _exp2_73 = approx_exp2(score_values_5[6] * softmax_scale_log2_2 - col_max_5[6]);
                            score_values_5[6] = _exp2_73;
                            float _exp2_74 = approx_exp2(score_values_5[7] * softmax_scale_log2_2 - col_max_5[7]);
                            score_values_5[7] = _exp2_74;
                        }
                        {
                            if constexpr (TILE_N == 16) {
                                uint16_t _fp8_pair_58;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_58) : "f"(0.0f), "f"(score_values_5[0]));
                                uint32_t _byte_58 = (uint32_t)(_fp8_pair_58 & 0xFF);
                                uint32_t _addr_58 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1536 + row_2 ^ (1536 + row_2 >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_58), "r"(_byte_58) : "memory");
                            } else {
                                uint16_t _fp8_pair_66;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_66) : "f"(0.0f), "f"(score_values_5[0]));
                                uint32_t _byte_66 = (uint32_t)(_fp8_pair_66 & 0xFF);
                                uint32_t _addr_66 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(3072 + row_2 ^ (3072 + row_2 >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_66), "r"(_byte_66) : "memory");
                            }
                        }
                        if constexpr (TILE_N == 16) {
                            float _fp8_rt_28;
                            uint16_t _e4m3x2_59;
                            uint32_t _f16x2_59;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_59) : "f"(0.0f), "f"(score_values_5[0]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_59) : "h"(_e4m3x2_59));
                            uint16_t _fp8_h0_59 = (uint16_t)(_f16x2_59 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_28) : "h"(_fp8_h0_59));
                            tr_vals_5[0] = _fp8_rt_28;
                        } else {
                            float _fp8_rt_56;
                            uint16_t _e4m3x2_67;
                            uint32_t _f16x2_67;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_67) : "f"(0.0f), "f"(score_values_5[0]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_67) : "h"(_e4m3x2_67));
                            uint16_t _fp8_h0_67 = (uint16_t)(_f16x2_67 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_56) : "h"(_fp8_h0_67));
                            tr_vals_5[0] = _fp8_rt_56;
                        }
                        {
                            if constexpr (TILE_N == 16) {
                                uint16_t _fp8_pair_60;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_60) : "f"(0.0f), "f"(score_values_5[1]));
                                uint32_t _byte_60 = (uint32_t)(_fp8_pair_60 & 0xFF);
                                uint32_t _addr_60 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1664 + row_2 ^ (1664 + row_2 >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_60), "r"(_byte_60) : "memory");
                            } else {
                                uint16_t _fp8_pair_68;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_68) : "f"(0.0f), "f"(score_values_5[1]));
                                uint32_t _byte_68 = (uint32_t)(_fp8_pair_68 & 0xFF);
                                uint32_t _addr_68 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(3200 + row_2 ^ (3200 + row_2 >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_68), "r"(_byte_68) : "memory");
                            }
                        }
                        if constexpr (TILE_N == 16) {
                            float _fp8_rt_29;
                            uint16_t _e4m3x2_61;
                            uint32_t _f16x2_61;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_61) : "f"(0.0f), "f"(score_values_5[1]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_61) : "h"(_e4m3x2_61));
                            uint16_t _fp8_h0_61 = (uint16_t)(_f16x2_61 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_29) : "h"(_fp8_h0_61));
                            tr_vals_5[1] = _fp8_rt_29;
                        } else {
                            float _fp8_rt_57;
                            uint16_t _e4m3x2_69;
                            uint32_t _f16x2_69;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_69) : "f"(0.0f), "f"(score_values_5[1]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_69) : "h"(_e4m3x2_69));
                            uint16_t _fp8_h0_69 = (uint16_t)(_f16x2_69 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_57) : "h"(_fp8_h0_69));
                            tr_vals_5[1] = _fp8_rt_57;
                        }
                        {
                            if constexpr (TILE_N == 16) {
                                uint16_t _fp8_pair_62;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_62) : "f"(0.0f), "f"(score_values_5[2]));
                                uint32_t _byte_62 = (uint32_t)(_fp8_pair_62 & 0xFF);
                                uint32_t _addr_62 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1792 + row_2 ^ (1792 + row_2 >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_62), "r"(_byte_62) : "memory");
                            } else {
                                uint16_t _fp8_pair_70;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_70) : "f"(0.0f), "f"(score_values_5[2]));
                                uint32_t _byte_70 = (uint32_t)(_fp8_pair_70 & 0xFF);
                                uint32_t _addr_70 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(3328 + row_2 ^ (3328 + row_2 >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_70), "r"(_byte_70) : "memory");
                            }
                        }
                        if constexpr (TILE_N == 16) {
                            float _fp8_rt_30;
                            uint16_t _e4m3x2_63;
                            uint32_t _f16x2_63;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_63) : "f"(0.0f), "f"(score_values_5[2]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_63) : "h"(_e4m3x2_63));
                            uint16_t _fp8_h0_63 = (uint16_t)(_f16x2_63 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_30) : "h"(_fp8_h0_63));
                            tr_vals_5[2] = _fp8_rt_30;
                        } else {
                            float _fp8_rt_58;
                            uint16_t _e4m3x2_71;
                            uint32_t _f16x2_71;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_71) : "f"(0.0f), "f"(score_values_5[2]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_71) : "h"(_e4m3x2_71));
                            uint16_t _fp8_h0_71 = (uint16_t)(_f16x2_71 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_58) : "h"(_fp8_h0_71));
                            tr_vals_5[2] = _fp8_rt_58;
                        }
                        {
                            if constexpr (TILE_N == 16) {
                                uint16_t _fp8_pair_64;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_64) : "f"(0.0f), "f"(score_values_5[3]));
                                uint32_t _byte_64 = (uint32_t)(_fp8_pair_64 & 0xFF);
                                uint32_t _addr_64 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1920 + row_2 ^ (1920 + row_2 >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_64), "r"(_byte_64) : "memory");
                            } else {
                                uint16_t _fp8_pair_72;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_72) : "f"(0.0f), "f"(score_values_5[3]));
                                uint32_t _byte_72 = (uint32_t)(_fp8_pair_72 & 0xFF);
                                uint32_t _addr_72 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(3456 + row_2 ^ (3456 + row_2 >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_72), "r"(_byte_72) : "memory");
                            }
                        }
                        if constexpr (TILE_N == 16) {
                            float _fp8_rt_31;
                            uint16_t _e4m3x2_65;
                            uint32_t _f16x2_65;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_65) : "f"(0.0f), "f"(score_values_5[3]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_65) : "h"(_e4m3x2_65));
                            uint16_t _fp8_h0_65 = (uint16_t)(_f16x2_65 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_31) : "h"(_fp8_h0_65));
                            tr_vals_5[3] = _fp8_rt_31;
                            int hi_bit_9_5 = lane & 16;
                            float send_10_5 = ((hi_bit_9_5 != 0) ? score_values_5[0] : score_values_5[2]);
                            float keep_11_5 = ((hi_bit_9_5 != 0) ? score_values_5[2] : score_values_5[0]);
                            float _shfl_190 = __shfl_sync(0xFFFFFFFF, send_10_5, lane ^ 16);
                            float recv_12_5 = _shfl_190;
                            score_values_5[0] = keep_11_5 + recv_12_5;
                            float send_13_5 = ((hi_bit_9_5 != 0) ? score_values_5[1] : score_values_5[3]);
                            float keep_14_5 = ((hi_bit_9_5 != 0) ? score_values_5[3] : score_values_5[1]);
                            float _shfl_191 = __shfl_sync(0xFFFFFFFF, send_13_5, lane ^ 16);
                            float recv_15_5 = _shfl_191;
                            score_values_5[1] = keep_14_5 + recv_15_5;
                            int hi_bit_16_5 = lane & 8;
                            float send_17_5 = ((hi_bit_16_5 != 0) ? score_values_5[0] : score_values_5[1]);
                            float keep_18_5 = ((hi_bit_16_5 != 0) ? score_values_5[1] : score_values_5[0]);
                            float _shfl_192 = __shfl_sync(0xFFFFFFFF, send_17_5, lane ^ 8);
                            float recv_19_5 = _shfl_192;
                            score_values_5[0] = keep_18_5 + recv_19_5;
                            float _shfl_193 = __shfl_sync(0xFFFFFFFF, score_values_5[0], lane ^ 4);
                            float other_20_5 = _shfl_193;
                            score_values_5[0] = score_values_5[0] + other_20_5;
                            float _shfl_194 = __shfl_sync(0xFFFFFFFF, score_values_5[0], lane ^ 2);
                            float other_21_3 = _shfl_194;
                            score_values_5[0] = score_values_5[0] + other_21_3;
                            float _shfl_195 = __shfl_sync(0xFFFFFFFF, score_values_5[0], lane ^ 1);
                            float other_22_3 = _shfl_195;
                            score_values_5[0] = score_values_5[0] + other_22_3;
                            int hi_bit_23_3 = lane & 16;
                            float send_24_3 = ((hi_bit_23_3 != 0) ? tr_vals_5[0] : tr_vals_5[2]);
                            float keep_25_3 = ((hi_bit_23_3 != 0) ? tr_vals_5[2] : tr_vals_5[0]);
                            float _shfl_196 = __shfl_sync(0xFFFFFFFF, send_24_3, lane ^ 16);
                            float recv_26_3 = _shfl_196;
                            tr_vals_5[0] = keep_25_3 + recv_26_3;
                            float send_27_3 = ((hi_bit_23_3 != 0) ? tr_vals_5[1] : tr_vals_5[3]);
                            float keep_28_3 = ((hi_bit_23_3 != 0) ? tr_vals_5[3] : tr_vals_5[1]);
                            float _shfl_197 = __shfl_sync(0xFFFFFFFF, send_27_3, lane ^ 16);
                            float recv_29_3 = _shfl_197;
                            tr_vals_5[1] = keep_28_3 + recv_29_3;
                            int hi_bit_30_3 = lane & 8;
                            float send_31_5 = ((hi_bit_30_3 != 0) ? tr_vals_5[0] : tr_vals_5[1]);
                            float keep_32_5 = ((hi_bit_30_3 != 0) ? tr_vals_5[1] : tr_vals_5[0]);
                            float _shfl_198 = __shfl_sync(0xFFFFFFFF, send_31_5, lane ^ 8);
                            float recv_33_5 = _shfl_198;
                            tr_vals_5[0] = keep_32_5 + recv_33_5;
                            float _shfl_199 = __shfl_sync(0xFFFFFFFF, tr_vals_5[0], lane ^ 4);
                            float other_34_3 = _shfl_199;
                            tr_vals_5[0] = tr_vals_5[0] + other_34_3;
                            float _shfl_200 = __shfl_sync(0xFFFFFFFF, tr_vals_5[0], lane ^ 2);
                            float other_35_3 = _shfl_200;
                            tr_vals_5[0] = tr_vals_5[0] + other_35_3;
                            float _shfl_201 = __shfl_sync(0xFFFFFFFF, tr_vals_5[0], lane ^ 1);
                            float other_36_3 = _shfl_201;
                            tr_vals_5[0] = tr_vals_5[0] + other_36_3;
                        } else {
                            float _fp8_rt_59;
                            uint16_t _e4m3x2_73;
                            uint32_t _f16x2_73;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_73) : "f"(0.0f), "f"(score_values_5[3]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_73) : "h"(_e4m3x2_73));
                            uint16_t _fp8_h0_73 = (uint16_t)(_f16x2_73 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_59) : "h"(_fp8_h0_73));
                            tr_vals_5[3] = _fp8_rt_59;
                            {
                                uint16_t _fp8_pair_74;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_74) : "f"(0.0f), "f"(score_values_5[4]));
                                uint32_t _byte_74 = (uint32_t)(_fp8_pair_74 & 0xFF);
                                uint32_t _addr_74 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(3584 + row_2 ^ (3584 + row_2 >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_74), "r"(_byte_74) : "memory");
                            }
                            float _fp8_rt_60;
                            uint16_t _e4m3x2_75;
                            uint32_t _f16x2_75;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_75) : "f"(0.0f), "f"(score_values_5[4]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_75) : "h"(_e4m3x2_75));
                            uint16_t _fp8_h0_75 = (uint16_t)(_f16x2_75 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_60) : "h"(_fp8_h0_75));
                            tr_vals_5[4] = _fp8_rt_60;
                            {
                                uint16_t _fp8_pair_76;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_76) : "f"(0.0f), "f"(score_values_5[5]));
                                uint32_t _byte_76 = (uint32_t)(_fp8_pair_76 & 0xFF);
                                uint32_t _addr_76 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(3712 + row_2 ^ (3712 + row_2 >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_76), "r"(_byte_76) : "memory");
                            }
                            float _fp8_rt_61;
                            uint16_t _e4m3x2_77;
                            uint32_t _f16x2_77;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_77) : "f"(0.0f), "f"(score_values_5[5]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_77) : "h"(_e4m3x2_77));
                            uint16_t _fp8_h0_77 = (uint16_t)(_f16x2_77 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_61) : "h"(_fp8_h0_77));
                            tr_vals_5[5] = _fp8_rt_61;
                            {
                                uint16_t _fp8_pair_78;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_78) : "f"(0.0f), "f"(score_values_5[6]));
                                uint32_t _byte_78 = (uint32_t)(_fp8_pair_78 & 0xFF);
                                uint32_t _addr_78 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(3840 + row_2 ^ (3840 + row_2 >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_78), "r"(_byte_78) : "memory");
                            }
                            float _fp8_rt_62;
                            uint16_t _e4m3x2_79;
                            uint32_t _f16x2_79;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_79) : "f"(0.0f), "f"(score_values_5[6]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_79) : "h"(_e4m3x2_79));
                            uint16_t _fp8_h0_79 = (uint16_t)(_f16x2_79 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_62) : "h"(_fp8_h0_79));
                            tr_vals_5[6] = _fp8_rt_62;
                            {
                                uint16_t _fp8_pair_80;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_80) : "f"(0.0f), "f"(score_values_5[7]));
                                uint32_t _byte_80 = (uint32_t)(_fp8_pair_80 & 0xFF);
                                uint32_t _addr_80 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(3968 + row_2 ^ (3968 + row_2 >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_80), "r"(_byte_80) : "memory");
                            }
                            float _fp8_rt_63;
                            uint16_t _e4m3x2_81;
                            uint32_t _f16x2_81;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_81) : "f"(0.0f), "f"(score_values_5[7]));
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_81) : "h"(_e4m3x2_81));
                            uint16_t _fp8_h0_81 = (uint16_t)(_f16x2_81 & 0xFFFFu);
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_63) : "h"(_fp8_h0_81));
                            tr_vals_5[7] = _fp8_rt_63;
                            int hi_bit_21_5 = lane & 16;
                            float send_22_5 = ((hi_bit_21_5 != 0) ? score_values_5[0] : score_values_5[4]);
                            float keep_23_5 = ((hi_bit_21_5 != 0) ? score_values_5[4] : score_values_5[0]);
                            float _shfl_338 = __shfl_sync(0xFFFFFFFF, send_22_5, lane ^ 16);
                            float recv_24_5 = _shfl_338;
                            score_values_5[0] = keep_23_5 + recv_24_5;
                            float send_25_5 = ((hi_bit_21_5 != 0) ? score_values_5[1] : score_values_5[5]);
                            float keep_26_5 = ((hi_bit_21_5 != 0) ? score_values_5[5] : score_values_5[1]);
                            float _shfl_339 = __shfl_sync(0xFFFFFFFF, send_25_5, lane ^ 16);
                            float recv_27_5 = _shfl_339;
                            score_values_5[1] = keep_26_5 + recv_27_5;
                            float send_28_5 = ((hi_bit_21_5 != 0) ? score_values_5[2] : score_values_5[6]);
                            float keep_29_5 = ((hi_bit_21_5 != 0) ? score_values_5[6] : score_values_5[2]);
                            float _shfl_340 = __shfl_sync(0xFFFFFFFF, send_28_5, lane ^ 16);
                            float recv_30_5 = _shfl_340;
                            score_values_5[2] = keep_29_5 + recv_30_5;
                            float send_31_5 = ((hi_bit_21_5 != 0) ? score_values_5[3] : score_values_5[7]);
                            float keep_32_5 = ((hi_bit_21_5 != 0) ? score_values_5[7] : score_values_5[3]);
                            float _shfl_341 = __shfl_sync(0xFFFFFFFF, send_31_5, lane ^ 16);
                            float recv_33_5 = _shfl_341;
                            score_values_5[3] = keep_32_5 + recv_33_5;
                            int hi_bit_34_5 = lane & 8;
                            float send_35_5 = ((hi_bit_34_5 != 0) ? score_values_5[0] : score_values_5[2]);
                            float keep_36_5 = ((hi_bit_34_5 != 0) ? score_values_5[2] : score_values_5[0]);
                            float _shfl_342 = __shfl_sync(0xFFFFFFFF, send_35_5, lane ^ 8);
                            float recv_37_5 = _shfl_342;
                            score_values_5[0] = keep_36_5 + recv_37_5;
                            float send_38_5 = ((hi_bit_34_5 != 0) ? score_values_5[1] : score_values_5[3]);
                            float keep_39_5 = ((hi_bit_34_5 != 0) ? score_values_5[3] : score_values_5[1]);
                            float _shfl_343 = __shfl_sync(0xFFFFFFFF, send_38_5, lane ^ 8);
                            float recv_40_5 = _shfl_343;
                            score_values_5[1] = keep_39_5 + recv_40_5;
                            int hi_bit_41_5 = lane & 4;
                            float send_42_5 = ((hi_bit_41_5 != 0) ? score_values_5[0] : score_values_5[1]);
                            float keep_43_5 = ((hi_bit_41_5 != 0) ? score_values_5[1] : score_values_5[0]);
                            float _shfl_344 = __shfl_sync(0xFFFFFFFF, send_42_5, lane ^ 4);
                            float recv_44_5 = _shfl_344;
                            score_values_5[0] = keep_43_5 + recv_44_5;
                            float _shfl_345 = __shfl_sync(0xFFFFFFFF, score_values_5[0], lane ^ 2);
                            float other_45_3 = _shfl_345;
                            score_values_5[0] = score_values_5[0] + other_45_3;
                            float _shfl_346 = __shfl_sync(0xFFFFFFFF, score_values_5[0], lane ^ 1);
                            float other_46_3 = _shfl_346;
                            score_values_5[0] = score_values_5[0] + other_46_3;
                            int hi_bit_47_3 = lane & 16;
                            float send_48_3 = ((hi_bit_47_3 != 0) ? tr_vals_5[0] : tr_vals_5[4]);
                            float keep_49_3 = ((hi_bit_47_3 != 0) ? tr_vals_5[4] : tr_vals_5[0]);
                            float _shfl_347 = __shfl_sync(0xFFFFFFFF, send_48_3, lane ^ 16);
                            float recv_50_3 = _shfl_347;
                            tr_vals_5[0] = keep_49_3 + recv_50_3;
                            float send_51_3 = ((hi_bit_47_3 != 0) ? tr_vals_5[1] : tr_vals_5[5]);
                            float keep_52_3 = ((hi_bit_47_3 != 0) ? tr_vals_5[5] : tr_vals_5[1]);
                            float _shfl_348 = __shfl_sync(0xFFFFFFFF, send_51_3, lane ^ 16);
                            float recv_53_3 = _shfl_348;
                            tr_vals_5[1] = keep_52_3 + recv_53_3;
                            float send_54_3 = ((hi_bit_47_3 != 0) ? tr_vals_5[2] : tr_vals_5[6]);
                            float keep_55_3 = ((hi_bit_47_3 != 0) ? tr_vals_5[6] : tr_vals_5[2]);
                            float _shfl_349 = __shfl_sync(0xFFFFFFFF, send_54_3, lane ^ 16);
                            float recv_56_3 = _shfl_349;
                            tr_vals_5[2] = keep_55_3 + recv_56_3;
                            float send_57_3 = ((hi_bit_47_3 != 0) ? tr_vals_5[3] : tr_vals_5[7]);
                            float keep_58_3 = ((hi_bit_47_3 != 0) ? tr_vals_5[7] : tr_vals_5[3]);
                            float _shfl_350 = __shfl_sync(0xFFFFFFFF, send_57_3, lane ^ 16);
                            float recv_59_3 = _shfl_350;
                            tr_vals_5[3] = keep_58_3 + recv_59_3;
                            int hi_bit_60_3 = lane & 8;
                            float send_61_5 = ((hi_bit_60_3 != 0) ? tr_vals_5[0] : tr_vals_5[2]);
                            float keep_62_5 = ((hi_bit_60_3 != 0) ? tr_vals_5[2] : tr_vals_5[0]);
                            float _shfl_351 = __shfl_sync(0xFFFFFFFF, send_61_5, lane ^ 8);
                            float recv_63_5 = _shfl_351;
                            tr_vals_5[0] = keep_62_5 + recv_63_5;
                            float send_64_5 = ((hi_bit_60_3 != 0) ? tr_vals_5[1] : tr_vals_5[3]);
                            float keep_65_5 = ((hi_bit_60_3 != 0) ? tr_vals_5[3] : tr_vals_5[1]);
                            float _shfl_352 = __shfl_sync(0xFFFFFFFF, send_64_5, lane ^ 8);
                            float recv_66_5 = _shfl_352;
                            tr_vals_5[1] = keep_65_5 + recv_66_5;
                            int hi_bit_67_3 = lane & 4;
                            float send_68_3 = ((hi_bit_67_3 != 0) ? tr_vals_5[0] : tr_vals_5[1]);
                            float keep_69_3 = ((hi_bit_67_3 != 0) ? tr_vals_5[1] : tr_vals_5[0]);
                            float _shfl_353 = __shfl_sync(0xFFFFFFFF, send_68_3, lane ^ 4);
                            float recv_70_3 = _shfl_353;
                            tr_vals_5[0] = keep_69_3 + recv_70_3;
                            float _shfl_354 = __shfl_sync(0xFFFFFFFF, tr_vals_5[0], lane ^ 2);
                            float other_71_3 = _shfl_354;
                            tr_vals_5[0] = tr_vals_5[0] + other_71_3;
                            float _shfl_355 = __shfl_sync(0xFFFFFFFF, tr_vals_5[0], lane ^ 1);
                            float other_72_3 = _shfl_355;
                            tr_vals_5[0] = tr_vals_5[0] + other_72_3;
                        }
                        if ((lane & ((-1 * TILE_N) / 4 + 11)) == 0) {
                            smem_psum[(8 + local_warp_2) * (TILE_N / 2) + (lane >> ((-1 * TILE_N) / 16 + 4))] = score_values_5[0];
                            smem_rsum[(8 + local_warp_2) * (TILE_N / 2) + (lane >> ((-1 * TILE_N) / 16 + 4))] = tr_vals_5[0];
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(p_full_addr);
                        asm volatile("barrier.sync 13, 128;" ::: "memory");
                        if (lane < (TILE_N / 4)) {
                            float tile_sum_5 = smem_psum[4 * TILE_N + lane] + smem_psum[((9 * TILE_N) / 2) + lane] + smem_psum[5 * TILE_N + lane] + smem_psum[((11 * TILE_N) / 2) + lane];
                            float tile_rsum_5 = smem_rsum[4 * TILE_N + lane] + smem_rsum[((9 * TILE_N) / 2) + lane] + smem_rsum[5 * TILE_N + lane] + smem_rsum[((11 * TILE_N) / 2) + lane];
                            if (it_0_2 == 0) {
                                float sink_term_5;
                                if constexpr (TILE_N == 16) {
                                    float _exp2_43 = approx_exp2(sink_lane_2 - m_scaled_5);
                                    sink_term_5 = _exp2_43;
                                } else {
                                    float _exp2_75 = approx_exp2(sink_lane_2 - m_scaled_5);
                                    sink_term_5 = _exp2_75;
                                }
                                l_run_2 = sink_term_5;
                                r_run_2 = sink_term_5;
                            }
                            l_run_2 = l_run_2 + tile_sum_5;
                            r_run_2 = r_run_2 + tile_rsum_5;
                        }
                        asm volatile("barrier.sync 13, 128;" ::: "memory");
                    }
                }
            }
            int last_par_2 = tiles_per_split - 1 & 1;
            {
                float norm_lane_2 = 0.0f;
                if (lane < (TILE_N / 4)) {
                    if (r_run_2 > 0.0f) {
                        float _rcp_2 = approx_rcp(r_run_2);
                        norm_lane_2 = _rcp_2 * output_scale_2;
                    }
                    if (local_warp_2 == 0) {
                        if (o_chunk_2 == 0 && head_base_2 + ((3 * TILE_N) / 4) + lane < num_heads) {
                            int lse_offset_2 = (query_idx_2 * num_heads + head_base_2 + ((3 * TILE_N) / 4) + lane) * num_splits + split_idx_2;
                            float _log2_2;
                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_2) : "f"(l_run_2));
                            partial_lse[lse_offset_2] = ((l_run_2 > 0.0f) ? (m_run_2 + _log2_2) * lse_partial_scale : -CAKE_INF);
                        }
                    }
                }
                float norm_c_2[(TILE_N / 4)];
                if constexpr (TILE_N == 16) {
                    float _shfl_202 = __shfl_sync(0xFFFFFFFF, norm_lane_2, 0);
                    norm_c_2[0] = _shfl_202;
                    float _shfl_203 = __shfl_sync(0xFFFFFFFF, norm_lane_2, 1);
                    norm_c_2[1] = _shfl_203;
                    float _shfl_204 = __shfl_sync(0xFFFFFFFF, norm_lane_2, 2);
                    norm_c_2[2] = _shfl_204;
                    float _shfl_205 = __shfl_sync(0xFFFFFFFF, norm_lane_2, 3);
                    norm_c_2[3] = _shfl_205;
                } else {
                    float _shfl_356 = __shfl_sync(0xFFFFFFFF, norm_lane_2, 0);
                    norm_c_2[0] = _shfl_356;
                    float _shfl_357 = __shfl_sync(0xFFFFFFFF, norm_lane_2, 1);
                    norm_c_2[1] = _shfl_357;
                    float _shfl_358 = __shfl_sync(0xFFFFFFFF, norm_lane_2, 2);
                    norm_c_2[2] = _shfl_358;
                    float _shfl_359 = __shfl_sync(0xFFFFFFFF, norm_lane_2, 3);
                    norm_c_2[3] = _shfl_359;
                    float _shfl_360 = __shfl_sync(0xFFFFFFFF, norm_lane_2, 4);
                    norm_c_2[4] = _shfl_360;
                    float _shfl_361 = __shfl_sync(0xFFFFFFFF, norm_lane_2, 5);
                    norm_c_2[5] = _shfl_361;
                    float _shfl_362 = __shfl_sync(0xFFFFFFFF, norm_lane_2, 6);
                    norm_c_2[6] = _shfl_362;
                    float _shfl_363 = __shfl_sync(0xFFFFFFFF, norm_lane_2, 7);
                    norm_c_2[7] = _shfl_363;
                }
                float o_values_2[(TILE_N / 4)];
                int dim_2 = 0;
                long long out_off_2 = 0;
                mbarrier_wait_hint(o_full_addr, last_par_2, 10000000);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if constexpr (TILE_N == 16) {
                    tmem_ld_x4(&o_values_2[0], taddr + 16 + 12 + (unsigned int)(tmem_row_origin_2 << 16));
                } else {
                    tmem_ld_x8(&o_values_2[0], taddr + 32 + 24 + (unsigned int)(tmem_row_origin_2 << 16));
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                dim_2 = o_chunk_2 * 4 * 128 + row_2;
                if (head_base_2 + ((3 * TILE_N) / 4) < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + ((3 * TILE_N) / 4)) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[0] * norm_c_2[0];
                }
                if (head_base_2 + ((3 * TILE_N) / 4) + 1 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + ((3 * TILE_N) / 4) + 1) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[1] * norm_c_2[1];
                }
                if (head_base_2 + ((3 * TILE_N) / 4) + 2 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + ((3 * TILE_N) / 4) + 2) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[2] * norm_c_2[2];
                }
                if (head_base_2 + ((3 * TILE_N) / 4) + 3 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + ((3 * TILE_N) / 4) + 3) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[3] * norm_c_2[3];
                }
                if constexpr (TILE_N == 32) {
                    if (head_base_2 + 24 + 4 < num_heads) {
                        out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 24 + 4) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                        partial_O[out_off_2] = o_values_2[4] * norm_c_2[4];
                    }
                    if (head_base_2 + 24 + 5 < num_heads) {
                        out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 24 + 5) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                        partial_O[out_off_2] = o_values_2[5] * norm_c_2[5];
                    }
                    if (head_base_2 + 24 + 6 < num_heads) {
                        out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 24 + 6) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                        partial_O[out_off_2] = o_values_2[6] * norm_c_2[6];
                    }
                    if (head_base_2 + 24 + 7 < num_heads) {
                        out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 24 + 7) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                        partial_O[out_off_2] = o_values_2[7] * norm_c_2[7];
                    }
                }
                mbarrier_wait_hint(o_full_addr + 8, last_par_2, 10000000);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if constexpr (TILE_N == 16) {
                    tmem_ld_x4(&o_values_2[0], taddr + 32 + 12 + (unsigned int)(tmem_row_origin_2 << 16));
                } else {
                    tmem_ld_x8(&o_values_2[0], taddr + 64 + 24 + (unsigned int)(tmem_row_origin_2 << 16));
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                dim_2 = (o_chunk_2 * 4 + 1) * 128 + row_2;
                if (head_base_2 + ((3 * TILE_N) / 4) < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + ((3 * TILE_N) / 4)) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[0] * norm_c_2[0];
                }
                if (head_base_2 + ((3 * TILE_N) / 4) + 1 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + ((3 * TILE_N) / 4) + 1) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[1] * norm_c_2[1];
                }
                if (head_base_2 + ((3 * TILE_N) / 4) + 2 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + ((3 * TILE_N) / 4) + 2) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[2] * norm_c_2[2];
                }
                if (head_base_2 + ((3 * TILE_N) / 4) + 3 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + ((3 * TILE_N) / 4) + 3) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[3] * norm_c_2[3];
                }
                if constexpr (TILE_N == 32) {
                    if (head_base_2 + 24 + 4 < num_heads) {
                        out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 24 + 4) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                        partial_O[out_off_2] = o_values_2[4] * norm_c_2[4];
                    }
                    if (head_base_2 + 24 + 5 < num_heads) {
                        out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 24 + 5) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                        partial_O[out_off_2] = o_values_2[5] * norm_c_2[5];
                    }
                    if (head_base_2 + 24 + 6 < num_heads) {
                        out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 24 + 6) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                        partial_O[out_off_2] = o_values_2[6] * norm_c_2[6];
                    }
                    if (head_base_2 + 24 + 7 < num_heads) {
                        out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 24 + 7) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                        partial_O[out_off_2] = o_values_2[7] * norm_c_2[7];
                    }
                }
                mbarrier_wait_hint(o_full_addr + 16, last_par_2, 10000000);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if constexpr (TILE_N == 16) {
                    tmem_ld_x4(&o_values_2[0], taddr + 48 + 12 + (unsigned int)(tmem_row_origin_2 << 16));
                } else {
                    tmem_ld_x8(&o_values_2[0], taddr + 96 + 24 + (unsigned int)(tmem_row_origin_2 << 16));
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                dim_2 = (o_chunk_2 * 4 + 2) * 128 + row_2;
                if (head_base_2 + ((3 * TILE_N) / 4) < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + ((3 * TILE_N) / 4)) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[0] * norm_c_2[0];
                }
                if (head_base_2 + ((3 * TILE_N) / 4) + 1 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + ((3 * TILE_N) / 4) + 1) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[1] * norm_c_2[1];
                }
                if (head_base_2 + ((3 * TILE_N) / 4) + 2 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + ((3 * TILE_N) / 4) + 2) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[2] * norm_c_2[2];
                }
                if (head_base_2 + ((3 * TILE_N) / 4) + 3 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + ((3 * TILE_N) / 4) + 3) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[3] * norm_c_2[3];
                }
                if constexpr (TILE_N == 32) {
                    if (head_base_2 + 24 + 4 < num_heads) {
                        out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 24 + 4) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                        partial_O[out_off_2] = o_values_2[4] * norm_c_2[4];
                    }
                    if (head_base_2 + 24 + 5 < num_heads) {
                        out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 24 + 5) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                        partial_O[out_off_2] = o_values_2[5] * norm_c_2[5];
                    }
                    if (head_base_2 + 24 + 6 < num_heads) {
                        out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 24 + 6) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                        partial_O[out_off_2] = o_values_2[6] * norm_c_2[6];
                    }
                    if (head_base_2 + 24 + 7 < num_heads) {
                        out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 24 + 7) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                        partial_O[out_off_2] = o_values_2[7] * norm_c_2[7];
                    }
                }
                mbarrier_wait_hint(o_full_addr + 24, last_par_2, 10000000);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if constexpr (TILE_N == 16) {
                    tmem_ld_x4(&o_values_2[0], taddr + 64 + 12 + (unsigned int)(tmem_row_origin_2 << 16));
                } else {
                    tmem_ld_x8(&o_values_2[0], taddr + 128 + 24 + (unsigned int)(tmem_row_origin_2 << 16));
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                dim_2 = (o_chunk_2 * 4 + 3) * 128 + row_2;
                if (head_base_2 + ((3 * TILE_N) / 4) < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + ((3 * TILE_N) / 4)) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[0] * norm_c_2[0];
                }
                if (head_base_2 + ((3 * TILE_N) / 4) + 1 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + ((3 * TILE_N) / 4) + 1) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[1] * norm_c_2[1];
                }
                if (head_base_2 + ((3 * TILE_N) / 4) + 2 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + ((3 * TILE_N) / 4) + 2) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[2] * norm_c_2[2];
                }
                if (head_base_2 + ((3 * TILE_N) / 4) + 3 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + ((3 * TILE_N) / 4) + 3) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[3] * norm_c_2[3];
                }
                if constexpr (TILE_N == 32) {
                    if (head_base_2 + 24 + 4 < num_heads) {
                        out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 24 + 4) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                        partial_O[out_off_2] = o_values_2[4] * norm_c_2[4];
                    }
                    if (head_base_2 + 24 + 5 < num_heads) {
                        out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 24 + 5) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                        partial_O[out_off_2] = o_values_2[5] * norm_c_2[5];
                    }
                    if (head_base_2 + 24 + 6 < num_heads) {
                        out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 24 + 6) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                        partial_O[out_off_2] = o_values_2[6] * norm_c_2[6];
                    }
                    if (head_base_2 + 24 + 7 < num_heads) {
                        out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 24 + 7) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                        partial_O[out_off_2] = o_values_2[7] * norm_c_2[7];
                    }
                }
                mbarrier_arrive(tmem_dealloc_addr);
            }
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 12) {
        // mma_warp_main
        {
            unsigned int _phase_q_ready_0 = 0;
            mbarrier_wait_hint(q_ready_addr, _phase_q_ready_0, 10000000);
            _phase_q_ready_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (elect_sync()) {
                tcgen05_cp_32x128b_warpx4(tmem_tmem_sfb0, make_sf_cp_desc_lo_sbo512((((smem_qsf_addr) >> 4))));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb0 + 4), make_sf_cp_desc_lo_sbo512((((smem_qsf_addr) >> 4) + 8)));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb0 + 8), make_sf_cp_desc_lo_sbo512((((smem_qsf_addr) >> 4) + 16)));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb0 + 12), make_sf_cp_desc_lo_sbo512((((smem_qsf_addr) >> 4) + 24)));
                tcgen05_cp_32x128b_warpx4(tmem_tmem_sfb1, make_sf_cp_desc_lo_sbo512((((smem_qsf_addr) >> 4) + 128)));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb1 + 4), make_sf_cp_desc_lo_sbo512((((smem_qsf_addr) >> 4) + 128 + 8)));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb1 + 8), make_sf_cp_desc_lo_sbo512((((smem_qsf_addr) >> 4) + 128 + 16)));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb1 + 12), make_sf_cp_desc_lo_sbo512((((smem_qsf_addr) >> 4) + 128 + 24)));
            }
            unsigned int _phase_q_rope_full_0 = 0;
            mbarrier_wait_hint(q_rope_full_addr, _phase_q_rope_full_0, 10000000);
            _phase_q_rope_full_0 ^= 1;
            for (int it2_3 = 0; it2_3 < (tiles_per_split + 1) / 2; it2_3++) {
                int it_3 = 2 * it2_3;
                if (it_3 < tiles_per_split) {
                    int first = ((it_3 == 0) ? 1 : 0);
                    mbarrier_wait_hint(kv_full_addr, 0, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfa0, make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4))));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfa0 + 4), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 8)));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfa0 + 8), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 16)));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfa0 + 12), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 24)));
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfa1, make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 128)));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfa1 + 4), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 128 + 8)));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfa1 + 8), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 128 + 16)));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfa1 + 12), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 128 + 24)));
                        int _mma_a_lo_0 = ((smem_krope_0_addr) >> 4) & 0x3FFF;
                        int _mma_b_lo_0 = ((smem_qrope_b_addr) >> 4) & 0x3FFF;
                        if constexpr (TILE_N == 16) {
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
                            "mov.b32 id, 134481040;\n\t"
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
                        } else {
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
                            "mov.b32 id, 134743184;\n\t"
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
                        }
                        int _mma_a_lo_1 = (((smem_kf4_0_addr) >> 4) & 0x3FFF) + (0) * 1024;
                        int _mma_b_lo_1 = (((smem_qf4b_addr) >> 4) & 0x3FFF) + (0) * 256;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);
                            if constexpr (TILE_N == 16) {
                                tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 0, b_desc + 0,
                                    0x8040480U, tmem_tmem_sfa0 + 0, tmem_tmem_sfb0 + 0, 1);
                                tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 2, b_desc + 2,
                                    0x8040480U, tmem_tmem_sfa0 + 4, tmem_tmem_sfb0 + 4, 1);
                                tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 4, b_desc + 4,
                                    0x8040480U, tmem_tmem_sfa0 + 8, tmem_tmem_sfb0 + 8, 1);
                                tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 6, b_desc + 6,
                                    0x8040480U, tmem_tmem_sfa0 + 12, tmem_tmem_sfb0 + 12, 1);
                            } else {
                                tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 0, b_desc + 0,
                                    0x8080480U, tmem_tmem_sfa0 + 0, tmem_tmem_sfb0 + 0, 1);
                                tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 2, b_desc + 2,
                                    0x8080480U, tmem_tmem_sfa0 + 4, tmem_tmem_sfb0 + 4, 1);
                                tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 4, b_desc + 4,
                                    0x8080480U, tmem_tmem_sfa0 + 8, tmem_tmem_sfb0 + 8, 1);
                                tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 6, b_desc + 6,
                                    0x8080480U, tmem_tmem_sfa0 + 12, tmem_tmem_sfb0 + 12, 1);
                            }
                        }
                        int _mma_a_lo_2 = (((smem_kf4_0_addr) >> 4) & 0x3FFF) + (1) * 1024;
                        int _mma_b_lo_2 = (((smem_qf4b_addr) >> 4) & 0x3FFF) + (1) * 256;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);
                            if constexpr (TILE_N == 16) {
                                tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 0, b_desc + 0,
                                    0x8040480U, tmem_tmem_sfa1 + 0, tmem_tmem_sfb1 + 0, 1);
                                tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 2, b_desc + 2,
                                    0x8040480U, tmem_tmem_sfa1 + 4, tmem_tmem_sfb1 + 4, 1);
                                tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 4, b_desc + 4,
                                    0x8040480U, tmem_tmem_sfa1 + 8, tmem_tmem_sfb1 + 8, 1);
                                tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 6, b_desc + 6,
                                    0x8040480U, tmem_tmem_sfa1 + 12, tmem_tmem_sfb1 + 12, 1);
                            } else {
                                tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 0, b_desc + 0,
                                    0x8080480U, tmem_tmem_sfa1 + 0, tmem_tmem_sfb1 + 0, 1);
                                tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 2, b_desc + 2,
                                    0x8080480U, tmem_tmem_sfa1 + 4, tmem_tmem_sfb1 + 4, 1);
                                tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 4, b_desc + 4,
                                    0x8080480U, tmem_tmem_sfa1 + 8, tmem_tmem_sfb1 + 8, 1);
                                tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 6, b_desc + 6,
                                    0x8080480U, tmem_tmem_sfa1 + 12, tmem_tmem_sfb1 + 12, 1);
                            }
                        }
                        tcgen05_commit(s_full_addr);
                    }
                    mbarrier_wait_hint(v_full_addr, 0, 10000000);
                    mbarrier_wait_hint(p_full_addr, 0, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        int _mma_a_lo_3 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (0) * 1024;
                        int _mma_b_lo_3 = (((smem_pb_addr) >> 4) & 0x3FFF) | 524288 * TILE_N;
                        if constexpr (TILE_N == 16) {
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
                            "mov.b32 id, 134512656;\n\t"
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
                        } else {
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
                            "mov.b32 id, 134774800;\n\t"
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
                        }
                        tcgen05_commit(o_full_addr);
                        int _mma_a_lo_4 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (1) * 1024;
                        if constexpr (TILE_N == 16) {
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
                            "mov.b32 id, 134512656;\n\t"
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
                        } else {
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
                            "mov.b32 id, 134774800;\n\t"
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
                        }
                        tcgen05_commit(o_full_addr + 8);
                        int _mma_a_lo_5 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (2) * 1024;
                        if constexpr (TILE_N == 16) {
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
                            "mov.b32 id, 134512656;\n\t"
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
                        } else {
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
                            "mov.b32 id, 134774800;\n\t"
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
                        }
                        tcgen05_commit(o_full_addr + 16);
                        int _mma_a_lo_6 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (3) * 1024;
                        if constexpr (TILE_N == 16) {
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
                            "mov.b32 id, 134512656;\n\t"
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
                        } else {
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
                            "mov.b32 id, 134774800;\n\t"
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
                        }
                        tcgen05_commit(o_full_addr + 24);
                    }
                }
                int it_0_3 = 2 * it2_3 + 1;
                if (it_0_3 < tiles_per_split) {
                    int first_1 = ((it_0_3 == 0) ? 1 : 0);
                    mbarrier_wait_hint(kv_full_addr, 1, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfa0, make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4))));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfa0 + 4), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 8)));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfa0 + 8), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 16)));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfa0 + 12), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 24)));
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfa1, make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 128)));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfa1 + 4), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 128 + 8)));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfa1 + 8), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 128 + 16)));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfa1 + 12), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 128 + 24)));
                        int _mma_a_lo_7 = ((smem_krope_1_addr) >> 4) & 0x3FFF;
                        int _mma_b_lo_7 = ((smem_qrope_b_addr) >> 4) & 0x3FFF;
                        if constexpr (TILE_N == 16) {
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
                            "mov.b32 id, 134481040;\n\t"
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
                        } else {
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
                            "mov.b32 id, 134743184;\n\t"
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
                        }
                        int _mma_a_lo_8 = (((smem_kf4_1_addr) >> 4) & 0x3FFF) + (0) * 1024;
                        int _mma_b_lo_8 = (((smem_qf4b_addr) >> 4) & 0x3FFF) + (0) * 256;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_8) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_8) | ((uint64_t)0x40004040 << 32);
                            if constexpr (TILE_N == 16) {
                                tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 0, b_desc + 0,
                                    0x8040480U, tmem_tmem_sfa0 + 0, tmem_tmem_sfb0 + 0, 1);
                                tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 2, b_desc + 2,
                                    0x8040480U, tmem_tmem_sfa0 + 4, tmem_tmem_sfb0 + 4, 1);
                                tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 4, b_desc + 4,
                                    0x8040480U, tmem_tmem_sfa0 + 8, tmem_tmem_sfb0 + 8, 1);
                                tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 6, b_desc + 6,
                                    0x8040480U, tmem_tmem_sfa0 + 12, tmem_tmem_sfb0 + 12, 1);
                            } else {
                                tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 0, b_desc + 0,
                                    0x8080480U, tmem_tmem_sfa0 + 0, tmem_tmem_sfb0 + 0, 1);
                                tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 2, b_desc + 2,
                                    0x8080480U, tmem_tmem_sfa0 + 4, tmem_tmem_sfb0 + 4, 1);
                                tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 4, b_desc + 4,
                                    0x8080480U, tmem_tmem_sfa0 + 8, tmem_tmem_sfb0 + 8, 1);
                                tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 6, b_desc + 6,
                                    0x8080480U, tmem_tmem_sfa0 + 12, tmem_tmem_sfb0 + 12, 1);
                            }
                        }
                        int _mma_a_lo_9 = (((smem_kf4_1_addr) >> 4) & 0x3FFF) + (1) * 1024;
                        int _mma_b_lo_9 = (((smem_qf4b_addr) >> 4) & 0x3FFF) + (1) * 256;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_9) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_9) | ((uint64_t)0x40004040 << 32);
                            if constexpr (TILE_N == 16) {
                                tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 0, b_desc + 0,
                                    0x8040480U, tmem_tmem_sfa1 + 0, tmem_tmem_sfb1 + 0, 1);
                                tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 2, b_desc + 2,
                                    0x8040480U, tmem_tmem_sfa1 + 4, tmem_tmem_sfb1 + 4, 1);
                                tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 4, b_desc + 4,
                                    0x8040480U, tmem_tmem_sfa1 + 8, tmem_tmem_sfb1 + 8, 1);
                                tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 6, b_desc + 6,
                                    0x8040480U, tmem_tmem_sfa1 + 12, tmem_tmem_sfb1 + 12, 1);
                            } else {
                                tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 0, b_desc + 0,
                                    0x8080480U, tmem_tmem_sfa1 + 0, tmem_tmem_sfb1 + 0, 1);
                                tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 2, b_desc + 2,
                                    0x8080480U, tmem_tmem_sfa1 + 4, tmem_tmem_sfb1 + 4, 1);
                                tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 4, b_desc + 4,
                                    0x8080480U, tmem_tmem_sfa1 + 8, tmem_tmem_sfb1 + 8, 1);
                                tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 6, b_desc + 6,
                                    0x8080480U, tmem_tmem_sfa1 + 12, tmem_tmem_sfb1 + 12, 1);
                            }
                        }
                        tcgen05_commit(s_full_addr);
                    }
                    mbarrier_wait_hint(v_full_addr, 1, 10000000);
                    mbarrier_wait_hint(p_full_addr, 1, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        int _mma_a_lo_10 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (0) * 1024;
                        int _mma_b_lo_10 = (((smem_pb_addr) >> 4) & 0x3FFF) | 524288 * TILE_N;
                        if constexpr (TILE_N == 16) {
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
                            "mov.b32 id, 134512656;\n\t"
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
                        } else {
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
                            "mov.b32 id, 134774800;\n\t"
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
                        }
                        tcgen05_commit(o_full_addr);
                        int _mma_a_lo_11 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (1) * 1024;
                        if constexpr (TILE_N == 16) {
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
                            "mov.b32 id, 134512656;\n\t"
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
                        } else {
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
                            "mov.b32 id, 134774800;\n\t"
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
                        }
                        tcgen05_commit(o_full_addr + 8);
                        int _mma_a_lo_12 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (2) * 1024;
                        if constexpr (TILE_N == 16) {
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
                            "mov.b32 id, 134512656;\n\t"
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
                        } else {
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
                            "mov.b32 id, 134774800;\n\t"
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
                        }
                        tcgen05_commit(o_full_addr + 16);
                        int _mma_a_lo_13 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (3) * 1024;
                        if constexpr (TILE_N == 16) {
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
                            "mov.b32 id, 134512656;\n\t"
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
                        } else {
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
                            "mov.b32 id, 134774800;\n\t"
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
                        }
                        tcgen05_commit(o_full_addr + 24);
                    }
                }
            }
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait_hint(tmem_dealloc_addr, _phase_tmem_dealloc_0, 10000000);
            _phase_tmem_dealloc_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(256));
        }
    }
    // ---- Role: load_warp ----
    if (warp >= 13 && warp <= 15) {
        // load_warp_main
        {
            const int load_tid = (warp - 13) * 32 + lane;
            int work_idx_3 = blockIdx.x;
            int head_tile_3 = work_idx_3 % num_head_tiles;
            int split_work_3 = work_idx_3 / num_head_tiles;
            int query_idx_3 = split_work_3 / num_splits;
            if (load_tid == 0) {
                mbarrier_arrive_expect_tx(q_nope_full0_addr, 384 * TILE_N);
                tma_4d_gmem2smem(smem_qstage_addr, (&tmap_q), 0, head_tile_3 * 128, 0, query_idx_3, q_nope_full0_addr);
                tma_4d_gmem2smem(smem_qstage_addr + 128 * TILE_N, (&tmap_q), 0, head_tile_3 * 128, 1, query_idx_3, q_nope_full0_addr);
                tma_4d_gmem2smem(smem_qstage_addr + 256 * TILE_N, (&tmap_q), 0, head_tile_3 * 128, 2, query_idx_3, q_nope_full0_addr);
                mbarrier_arrive_expect_tx(q_nope_full1_addr, 256 * TILE_N);
                tma_4d_gmem2smem(smem_qstage_addr + 384 * TILE_N, (&tmap_q), 0, head_tile_3 * 128, 3, query_idx_3, q_nope_full1_addr);
                tma_4d_gmem2smem(smem_qstage_addr + 512 * TILE_N, (&tmap_q), 0, head_tile_3 * 128, 4, query_idx_3, q_nope_full1_addr);
                mbarrier_arrive_expect_tx(q_nope_full2_addr, 256 * TILE_N);
                tma_4d_gmem2smem(smem_qstage_addr + 640 * TILE_N, (&tmap_q), 0, head_tile_3 * 128, 5, query_idx_3, q_nope_full2_addr);
                tma_4d_gmem2smem(smem_qstage_addr + 768 * TILE_N, (&tmap_q), 0, head_tile_3 * 128, 6, query_idx_3, q_nope_full2_addr);
                mbarrier_arrive_expect_tx(q_rope_full_addr, 128 * TILE_N);
                tma_4d_gmem2smem(smem_qrope_addr, (&tmap_q), 0, head_tile_3 * 128, 7, query_idx_3, q_rope_full_addr);
            }
            const int load_tid_0 = (warp - 13) * 32 + lane;
            const int l_chunk = load_tid_0 % 24;
            const int l_row0 = load_tid_0 / 24;
            const int l_kind = ((l_chunk < 14) ? 0 : ((l_chunk < 22) ? 1 : 2));
            const int l_sw_chunk = ((l_kind == 0) ? l_chunk : l_chunk - 14);
            long long l_data_off = (long long)(16 * l_chunk);
            long long l_sf_off = (long long)(16 * (l_chunk - 14 - 8));
            int l_dst_0 = ((l_kind == 0) ? smem_kf4_0_addr + (unsigned int)(l_chunk / 8 * 16384) : ((l_kind == 1) ? smem_krope_0_addr : smem_sfs_0_addr + (unsigned int)(16 * (l_chunk - 14 - 8))));
            int l_dst_1 = ((l_kind == 0) ? smem_kf4_1_addr + (unsigned int)(l_chunk / 8 * 16384) : ((l_kind == 1) ? smem_krope_1_addr : smem_sfs_1_addr + (unsigned int)(16 * (l_chunk - 14 - 8))));
            int l_work_idx = blockIdx.x;
            int l_head_tile = l_work_idx % num_head_tiles;
            int l_split_work = l_work_idx / num_head_tiles;
            int l_split_idx = l_split_work % num_splits;
            int l_query_idx = l_split_work / num_splits;
            int tile_lo = l_split_idx * tiles_per_split;
            int idx[2];
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
            for (int it2_4 = 0; it2_4 < (tiles_per_split + 1) / 2; it2_4++) {
                int it_4 = 2 * it2_4;
                if (it_4 < tiles_per_split) {
                    if (it_4 > 1) {
                        mbarrier_wait_hint(tok_free_addr, it2_4 - 1 & 1, 10000000);
                    }
                    for (int rr = 0; rr < 2; rr++) {
                        int r_0 = load_tid_0 + 96 * rr;
                        if (r_0 < 128) {
                            smem_tok_0[r_0] = idx[rr];
                        }
                    }
                    asm volatile("barrier.sync 8, 96;" ::: "memory");
                    int is_main_0 = 1;
                    if (tile_lo + it_4 >= num_main_tiles) {
                        is_main_0 = 0;
                    }
                    uint8_t* cache = ((is_main_0 != 0) ? (main_cache) : (extra_cache));
                    int page_shift = ((is_main_0 != 0) ? main_page_shift : extra_page_shift);
                    long long page_stride = ((is_main_0 != 0) ? main_page_stride : extra_page_stride);
                    long long sf_page_off = (long long)((1 << page_shift) * 352) + l_sf_off;
                    for (int i8 = 0; i8 < 4; i8++) {
                        int ctoks[8];
                        ctoks[0] = smem_tokr_0[l_row0 + 32 * i8];
                        ctoks[1] = smem_tokr_0[l_row0 + 32 * i8 + 4];
                        ctoks[2] = smem_tokr_0[l_row0 + 32 * i8 + 8];
                        ctoks[3] = smem_tokr_0[l_row0 + 32 * i8 + 12];
                        ctoks[4] = smem_tokr_0[l_row0 + 32 * i8 + 16];
                        ctoks[5] = smem_tokr_0[l_row0 + 32 * i8 + 20];
                        ctoks[6] = smem_tokr_0[l_row0 + 32 * i8 + 24];
                        ctoks[7] = smem_tokr_0[l_row0 + 32 * i8 + 28];
                        int crow = l_row0 + 32 * i8;
                        int ctok = ctoks[0];
                        if (ctok >= 0) {
                            int cpage = ctok >> page_shift;
                            int cslot = ctok - (cpage << page_shift);
                            long long cbase = (long long)cpage * page_stride;
                            long long data_src = cbase + (long long)(cslot * 352) + l_data_off;
                            long long sf_src = cbase + sf_page_off + (long long)(cslot * 32);
                            long long src_off = ((l_kind < 2) ? data_src : sf_src);
                            int dst_sw = l_dst_0 + crow * 128 + (((l_sw_chunk ^ crow) & 7) << 4);
                            int dst_s = l_dst_0 + crow * 32;
                            int dst_off = ((l_kind < 2) ? dst_sw : dst_s);
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(dst_off), "l"(cache + src_off));
                        }
                        int crow_0 = l_row0 + 32 * i8 + 4;
                        int ctok_1 = ctoks[1];
                        if (ctok_1 >= 0) {
                            int cpage_1 = ctok_1 >> page_shift;
                            int cslot_1 = ctok_1 - (cpage_1 << page_shift);
                            long long cbase_1 = (long long)cpage_1 * page_stride;
                            long long data_src_1 = cbase_1 + (long long)(cslot_1 * 352) + l_data_off;
                            long long sf_src_1 = cbase_1 + sf_page_off + (long long)(cslot_1 * 32);
                            long long src_off_1 = ((l_kind < 2) ? data_src_1 : sf_src_1);
                            int dst_sw_1 = l_dst_0 + crow_0 * 128 + (((l_sw_chunk ^ crow_0) & 7) << 4);
                            int dst_s_1 = l_dst_0 + crow_0 * 32;
                            int dst_off_1 = ((l_kind < 2) ? dst_sw_1 : dst_s_1);
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(dst_off_1), "l"(cache + src_off_1));
                        }
                        int crow_2 = l_row0 + 32 * i8 + 8;
                        int ctok_3 = ctoks[2];
                        if (ctok_3 >= 0) {
                            int cpage_2 = ctok_3 >> page_shift;
                            int cslot_2 = ctok_3 - (cpage_2 << page_shift);
                            long long cbase_2 = (long long)cpage_2 * page_stride;
                            long long data_src_2 = cbase_2 + (long long)(cslot_2 * 352) + l_data_off;
                            long long sf_src_2 = cbase_2 + sf_page_off + (long long)(cslot_2 * 32);
                            long long src_off_2 = ((l_kind < 2) ? data_src_2 : sf_src_2);
                            int dst_sw_2 = l_dst_0 + crow_2 * 128 + (((l_sw_chunk ^ crow_2) & 7) << 4);
                            int dst_s_2 = l_dst_0 + crow_2 * 32;
                            int dst_off_2 = ((l_kind < 2) ? dst_sw_2 : dst_s_2);
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(dst_off_2), "l"(cache + src_off_2));
                        }
                        int crow_4 = l_row0 + 32 * i8 + 12;
                        int ctok_5 = ctoks[3];
                        if (ctok_5 >= 0) {
                            int cpage_3 = ctok_5 >> page_shift;
                            int cslot_3 = ctok_5 - (cpage_3 << page_shift);
                            long long cbase_3 = (long long)cpage_3 * page_stride;
                            long long data_src_3 = cbase_3 + (long long)(cslot_3 * 352) + l_data_off;
                            long long sf_src_3 = cbase_3 + sf_page_off + (long long)(cslot_3 * 32);
                            long long src_off_3 = ((l_kind < 2) ? data_src_3 : sf_src_3);
                            int dst_sw_3 = l_dst_0 + crow_4 * 128 + (((l_sw_chunk ^ crow_4) & 7) << 4);
                            int dst_s_3 = l_dst_0 + crow_4 * 32;
                            int dst_off_3 = ((l_kind < 2) ? dst_sw_3 : dst_s_3);
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(dst_off_3), "l"(cache + src_off_3));
                        }
                        int crow_6 = l_row0 + 32 * i8 + 16;
                        int ctok_7 = ctoks[4];
                        if (ctok_7 >= 0) {
                            int cpage_4 = ctok_7 >> page_shift;
                            int cslot_4 = ctok_7 - (cpage_4 << page_shift);
                            long long cbase_4 = (long long)cpage_4 * page_stride;
                            long long data_src_4 = cbase_4 + (long long)(cslot_4 * 352) + l_data_off;
                            long long sf_src_4 = cbase_4 + sf_page_off + (long long)(cslot_4 * 32);
                            long long src_off_4 = ((l_kind < 2) ? data_src_4 : sf_src_4);
                            int dst_sw_4 = l_dst_0 + crow_6 * 128 + (((l_sw_chunk ^ crow_6) & 7) << 4);
                            int dst_s_4 = l_dst_0 + crow_6 * 32;
                            int dst_off_4 = ((l_kind < 2) ? dst_sw_4 : dst_s_4);
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(dst_off_4), "l"(cache + src_off_4));
                        }
                        int crow_8 = l_row0 + 32 * i8 + 20;
                        int ctok_9 = ctoks[5];
                        if (ctok_9 >= 0) {
                            int cpage_5 = ctok_9 >> page_shift;
                            int cslot_5 = ctok_9 - (cpage_5 << page_shift);
                            long long cbase_5 = (long long)cpage_5 * page_stride;
                            long long data_src_5 = cbase_5 + (long long)(cslot_5 * 352) + l_data_off;
                            long long sf_src_5 = cbase_5 + sf_page_off + (long long)(cslot_5 * 32);
                            long long src_off_5 = ((l_kind < 2) ? data_src_5 : sf_src_5);
                            int dst_sw_5 = l_dst_0 + crow_8 * 128 + (((l_sw_chunk ^ crow_8) & 7) << 4);
                            int dst_s_5 = l_dst_0 + crow_8 * 32;
                            int dst_off_5 = ((l_kind < 2) ? dst_sw_5 : dst_s_5);
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(dst_off_5), "l"(cache + src_off_5));
                        }
                        int crow_10 = l_row0 + 32 * i8 + 24;
                        int ctok_11 = ctoks[6];
                        if (ctok_11 >= 0) {
                            int cpage_6 = ctok_11 >> page_shift;
                            int cslot_6 = ctok_11 - (cpage_6 << page_shift);
                            long long cbase_6 = (long long)cpage_6 * page_stride;
                            long long data_src_6 = cbase_6 + (long long)(cslot_6 * 352) + l_data_off;
                            long long sf_src_6 = cbase_6 + sf_page_off + (long long)(cslot_6 * 32);
                            long long src_off_6 = ((l_kind < 2) ? data_src_6 : sf_src_6);
                            int dst_sw_6 = l_dst_0 + crow_10 * 128 + (((l_sw_chunk ^ crow_10) & 7) << 4);
                            int dst_s_6 = l_dst_0 + crow_10 * 32;
                            int dst_off_6 = ((l_kind < 2) ? dst_sw_6 : dst_s_6);
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(dst_off_6), "l"(cache + src_off_6));
                        }
                        int crow_12 = l_row0 + 32 * i8 + 28;
                        int ctok_13 = ctoks[7];
                        if (ctok_13 >= 0) {
                            int cpage_7 = ctok_13 >> page_shift;
                            int cslot_7 = ctok_13 - (cpage_7 << page_shift);
                            long long cbase_7 = (long long)cpage_7 * page_stride;
                            long long data_src_7 = cbase_7 + (long long)(cslot_7 * 352) + l_data_off;
                            long long sf_src_7 = cbase_7 + sf_page_off + (long long)(cslot_7 * 32);
                            long long src_off_7 = ((l_kind < 2) ? data_src_7 : sf_src_7);
                            int dst_sw_7 = l_dst_0 + crow_12 * 128 + (((l_sw_chunk ^ crow_12) & 7) << 4);
                            int dst_s_7 = l_dst_0 + crow_12 * 32;
                            int dst_off_7 = ((l_kind < 2) ? dst_sw_7 : dst_s_7);
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(dst_off_7), "l"(cache + src_off_7));
                        }
                    }
                    asm volatile(
                        "{\n\t"
                        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                        "}"
                        :: "r"(tok_full_addr) : "memory");
                    if (it_4 + 1 < tiles_per_split) {
                        int is_main_1 = 1;
                        if (tile_lo + it_4 + 1 >= num_main_tiles) {
                            is_main_1 = 0;
                        }
                        int tile_in_table_2 = ((is_main_1 != 0) ? tile_lo + it_4 + 1 : tile_lo + it_4 + 1 - num_main_tiles);
                        int table_width_3 = ((is_main_1 != 0) ? main_width : extra_width);
                        int* row_ptr_4 = ((is_main_1 != 0) ? (main_indices + (l_query_idx * main_index_stride)) : (extra_indices + (l_query_idx * extra_index_stride)));
                        int active_len_5 = table_width_3;
                        if (is_main_1 != 0) {
                            if (has_main_lengths != 0) {
                                active_len_5 = main_lengths[l_query_idx];
                            }
                        } else if (has_extra_lengths != 0) {
                            active_len_5 = extra_lengths[l_query_idx];
                        }
                        if (active_len_5 < 0) {
                            active_len_5 = 0;
                        }
                        if (active_len_5 > table_width_3) {
                            active_len_5 = table_width_3;
                        }
                        int r_6 = load_tid_0;
                        idx[0] = -1;
                        if (r_6 < 128) {
                            int col_2 = tile_in_table_2 * 128 + r_6;
                            if (col_2 < active_len_5) {
                                idx[0] = row_ptr_4[col_2];
                            }
                        }
                        int r_7 = load_tid_0 + 96;
                        idx[1] = -1;
                        if (r_7 < 128) {
                            int col_3 = tile_in_table_2 * 128 + r_7;
                            if (col_3 < active_len_5) {
                                idx[1] = row_ptr_4[col_3];
                            }
                        }
                    }
                }
                int it_0_4 = 2 * it2_4 + 1;
                if (it_0_4 < tiles_per_split) {
                    mbarrier_wait_hint(tok_free_addr + 8, it2_4 & 1, 10000000);
                    for (int rr_1 = 0; rr_1 < 2; rr_1++) {
                        int r_0_1 = load_tid_0 + 96 * rr_1;
                        if (r_0_1 < 128) {
                            smem_tok_1[r_0_1] = idx[rr_1];
                        }
                    }
                    asm volatile("barrier.sync 8, 96;" ::: "memory");
                    int is_main_0_1 = 1;
                    if (tile_lo + it_0_4 >= num_main_tiles) {
                        is_main_0_1 = 0;
                    }
                    uint8_t* cache_1 = ((is_main_0_1 != 0) ? (main_cache) : (extra_cache));
                    int page_shift_1 = ((is_main_0_1 != 0) ? main_page_shift : extra_page_shift);
                    long long page_stride_1 = ((is_main_0_1 != 0) ? main_page_stride : extra_page_stride);
                    long long sf_page_off_1 = (long long)((1 << page_shift_1) * 352) + l_sf_off;
                    for (int i8_1 = 0; i8_1 < 4; i8_1++) {
                        int ctoks_1[8];
                        ctoks_1[0] = smem_tokr_1[l_row0 + 32 * i8_1];
                        ctoks_1[1] = smem_tokr_1[l_row0 + 32 * i8_1 + 4];
                        ctoks_1[2] = smem_tokr_1[l_row0 + 32 * i8_1 + 8];
                        ctoks_1[3] = smem_tokr_1[l_row0 + 32 * i8_1 + 12];
                        ctoks_1[4] = smem_tokr_1[l_row0 + 32 * i8_1 + 16];
                        ctoks_1[5] = smem_tokr_1[l_row0 + 32 * i8_1 + 20];
                        ctoks_1[6] = smem_tokr_1[l_row0 + 32 * i8_1 + 24];
                        ctoks_1[7] = smem_tokr_1[l_row0 + 32 * i8_1 + 28];
                        int crow_1 = l_row0 + 32 * i8_1;
                        int ctok_2 = ctoks_1[0];
                        if (ctok_2 >= 0) {
                            int cpage_8 = ctok_2 >> page_shift_1;
                            int cslot_8 = ctok_2 - (cpage_8 << page_shift_1);
                            long long cbase_8 = (long long)cpage_8 * page_stride_1;
                            long long data_src_8 = cbase_8 + (long long)(cslot_8 * 352) + l_data_off;
                            long long sf_src_8 = cbase_8 + sf_page_off_1 + (long long)(cslot_8 * 32);
                            long long src_off_8 = ((l_kind < 2) ? data_src_8 : sf_src_8);
                            int dst_sw_8 = l_dst_1 + crow_1 * 128 + (((l_sw_chunk ^ crow_1) & 7) << 4);
                            int dst_s_8 = l_dst_1 + crow_1 * 32;
                            int dst_off_8 = ((l_kind < 2) ? dst_sw_8 : dst_s_8);
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(dst_off_8), "l"(cache_1 + src_off_8));
                        }
                        int crow_0_1 = l_row0 + 32 * i8_1 + 4;
                        int ctok_1_1 = ctoks_1[1];
                        if (ctok_1_1 >= 0) {
                            int cpage_9 = ctok_1_1 >> page_shift_1;
                            int cslot_9 = ctok_1_1 - (cpage_9 << page_shift_1);
                            long long cbase_9 = (long long)cpage_9 * page_stride_1;
                            long long data_src_9 = cbase_9 + (long long)(cslot_9 * 352) + l_data_off;
                            long long sf_src_9 = cbase_9 + sf_page_off_1 + (long long)(cslot_9 * 32);
                            long long src_off_9 = ((l_kind < 2) ? data_src_9 : sf_src_9);
                            int dst_sw_9 = l_dst_1 + crow_0_1 * 128 + (((l_sw_chunk ^ crow_0_1) & 7) << 4);
                            int dst_s_9 = l_dst_1 + crow_0_1 * 32;
                            int dst_off_9 = ((l_kind < 2) ? dst_sw_9 : dst_s_9);
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(dst_off_9), "l"(cache_1 + src_off_9));
                        }
                        int crow_2_1 = l_row0 + 32 * i8_1 + 8;
                        int ctok_3_1 = ctoks_1[2];
                        if (ctok_3_1 >= 0) {
                            int cpage_10 = ctok_3_1 >> page_shift_1;
                            int cslot_10 = ctok_3_1 - (cpage_10 << page_shift_1);
                            long long cbase_10 = (long long)cpage_10 * page_stride_1;
                            long long data_src_10 = cbase_10 + (long long)(cslot_10 * 352) + l_data_off;
                            long long sf_src_10 = cbase_10 + sf_page_off_1 + (long long)(cslot_10 * 32);
                            long long src_off_10 = ((l_kind < 2) ? data_src_10 : sf_src_10);
                            int dst_sw_10 = l_dst_1 + crow_2_1 * 128 + (((l_sw_chunk ^ crow_2_1) & 7) << 4);
                            int dst_s_10 = l_dst_1 + crow_2_1 * 32;
                            int dst_off_10 = ((l_kind < 2) ? dst_sw_10 : dst_s_10);
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(dst_off_10), "l"(cache_1 + src_off_10));
                        }
                        int crow_4_1 = l_row0 + 32 * i8_1 + 12;
                        int ctok_5_1 = ctoks_1[3];
                        if (ctok_5_1 >= 0) {
                            int cpage_11 = ctok_5_1 >> page_shift_1;
                            int cslot_11 = ctok_5_1 - (cpage_11 << page_shift_1);
                            long long cbase_11 = (long long)cpage_11 * page_stride_1;
                            long long data_src_11 = cbase_11 + (long long)(cslot_11 * 352) + l_data_off;
                            long long sf_src_11 = cbase_11 + sf_page_off_1 + (long long)(cslot_11 * 32);
                            long long src_off_11 = ((l_kind < 2) ? data_src_11 : sf_src_11);
                            int dst_sw_11 = l_dst_1 + crow_4_1 * 128 + (((l_sw_chunk ^ crow_4_1) & 7) << 4);
                            int dst_s_11 = l_dst_1 + crow_4_1 * 32;
                            int dst_off_11 = ((l_kind < 2) ? dst_sw_11 : dst_s_11);
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(dst_off_11), "l"(cache_1 + src_off_11));
                        }
                        int crow_6_1 = l_row0 + 32 * i8_1 + 16;
                        int ctok_7_1 = ctoks_1[4];
                        if (ctok_7_1 >= 0) {
                            int cpage_12 = ctok_7_1 >> page_shift_1;
                            int cslot_12 = ctok_7_1 - (cpage_12 << page_shift_1);
                            long long cbase_12 = (long long)cpage_12 * page_stride_1;
                            long long data_src_12 = cbase_12 + (long long)(cslot_12 * 352) + l_data_off;
                            long long sf_src_12 = cbase_12 + sf_page_off_1 + (long long)(cslot_12 * 32);
                            long long src_off_12 = ((l_kind < 2) ? data_src_12 : sf_src_12);
                            int dst_sw_12 = l_dst_1 + crow_6_1 * 128 + (((l_sw_chunk ^ crow_6_1) & 7) << 4);
                            int dst_s_12 = l_dst_1 + crow_6_1 * 32;
                            int dst_off_12 = ((l_kind < 2) ? dst_sw_12 : dst_s_12);
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(dst_off_12), "l"(cache_1 + src_off_12));
                        }
                        int crow_8_1 = l_row0 + 32 * i8_1 + 20;
                        int ctok_9_1 = ctoks_1[5];
                        if (ctok_9_1 >= 0) {
                            int cpage_13 = ctok_9_1 >> page_shift_1;
                            int cslot_13 = ctok_9_1 - (cpage_13 << page_shift_1);
                            long long cbase_13 = (long long)cpage_13 * page_stride_1;
                            long long data_src_13 = cbase_13 + (long long)(cslot_13 * 352) + l_data_off;
                            long long sf_src_13 = cbase_13 + sf_page_off_1 + (long long)(cslot_13 * 32);
                            long long src_off_13 = ((l_kind < 2) ? data_src_13 : sf_src_13);
                            int dst_sw_13 = l_dst_1 + crow_8_1 * 128 + (((l_sw_chunk ^ crow_8_1) & 7) << 4);
                            int dst_s_13 = l_dst_1 + crow_8_1 * 32;
                            int dst_off_13 = ((l_kind < 2) ? dst_sw_13 : dst_s_13);
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(dst_off_13), "l"(cache_1 + src_off_13));
                        }
                        int crow_10_1 = l_row0 + 32 * i8_1 + 24;
                        int ctok_11_1 = ctoks_1[6];
                        if (ctok_11_1 >= 0) {
                            int cpage_14 = ctok_11_1 >> page_shift_1;
                            int cslot_14 = ctok_11_1 - (cpage_14 << page_shift_1);
                            long long cbase_14 = (long long)cpage_14 * page_stride_1;
                            long long data_src_14 = cbase_14 + (long long)(cslot_14 * 352) + l_data_off;
                            long long sf_src_14 = cbase_14 + sf_page_off_1 + (long long)(cslot_14 * 32);
                            long long src_off_14 = ((l_kind < 2) ? data_src_14 : sf_src_14);
                            int dst_sw_14 = l_dst_1 + crow_10_1 * 128 + (((l_sw_chunk ^ crow_10_1) & 7) << 4);
                            int dst_s_14 = l_dst_1 + crow_10_1 * 32;
                            int dst_off_14 = ((l_kind < 2) ? dst_sw_14 : dst_s_14);
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(dst_off_14), "l"(cache_1 + src_off_14));
                        }
                        int crow_12_1 = l_row0 + 32 * i8_1 + 28;
                        int ctok_13_1 = ctoks_1[7];
                        if (ctok_13_1 >= 0) {
                            int cpage_15 = ctok_13_1 >> page_shift_1;
                            int cslot_15 = ctok_13_1 - (cpage_15 << page_shift_1);
                            long long cbase_15 = (long long)cpage_15 * page_stride_1;
                            long long data_src_15 = cbase_15 + (long long)(cslot_15 * 352) + l_data_off;
                            long long sf_src_15 = cbase_15 + sf_page_off_1 + (long long)(cslot_15 * 32);
                            long long src_off_15 = ((l_kind < 2) ? data_src_15 : sf_src_15);
                            int dst_sw_15 = l_dst_1 + crow_12_1 * 128 + (((l_sw_chunk ^ crow_12_1) & 7) << 4);
                            int dst_s_15 = l_dst_1 + crow_12_1 * 32;
                            int dst_off_15 = ((l_kind < 2) ? dst_sw_15 : dst_s_15);
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(dst_off_15), "l"(cache_1 + src_off_15));
                        }
                    }
                    asm volatile(
                        "{\n\t"
                        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                        "}"
                        :: "r"(tok_full_addr + 8) : "memory");
                    if (it_0_4 + 1 < tiles_per_split) {
                        int is_main_1_1 = 1;
                        if (tile_lo + it_0_4 + 1 >= num_main_tiles) {
                            is_main_1_1 = 0;
                        }
                        int tile_in_table_2_1 = ((is_main_1_1 != 0) ? tile_lo + it_0_4 + 1 : tile_lo + it_0_4 + 1 - num_main_tiles);
                        int table_width_3_1 = ((is_main_1_1 != 0) ? main_width : extra_width);
                        int* row_ptr_4_1 = ((is_main_1_1 != 0) ? (main_indices + (l_query_idx * main_index_stride)) : (extra_indices + (l_query_idx * extra_index_stride)));
                        int active_len_5_1 = table_width_3_1;
                        if (is_main_1_1 != 0) {
                            if (has_main_lengths != 0) {
                                active_len_5_1 = main_lengths[l_query_idx];
                            }
                        } else if (has_extra_lengths != 0) {
                            active_len_5_1 = extra_lengths[l_query_idx];
                        }
                        if (active_len_5_1 < 0) {
                            active_len_5_1 = 0;
                        }
                        if (active_len_5_1 > table_width_3_1) {
                            active_len_5_1 = table_width_3_1;
                        }
                        int r_6_1 = load_tid_0;
                        idx[0] = -1;
                        if (r_6_1 < 128) {
                            int col_4 = tile_in_table_2_1 * 128 + r_6_1;
                            if (col_4 < active_len_5_1) {
                                idx[0] = row_ptr_4_1[col_4];
                            }
                        }
                        int r_7_1 = load_tid_0 + 96;
                        idx[1] = -1;
                        if (r_7_1 < 128) {
                            int col_5 = tile_in_table_2_1 * 128 + r_7_1;
                            if (col_5 < active_len_5_1) {
                                idx[1] = row_ptr_4_1[col_5];
                            }
                        }
                    }
                }
            }
            asm volatile("cp.async.wait_group 0;");
        }
    }
    // Cleanup
}

// Explicit instantiations.  FlashInfer compiles this unit once per registered variant with
// -DCAKE_DSV4_NVFP4_DECODE_PV_SELECT=1 -D<variant macro>=1 so a module instantiates its own kernel only; without
// CAKE_DSV4_NVFP4_DECODE_PV_SELECT (the SASS proof) every instantiation is compiled.
#if !defined(CAKE_DSV4_NVFP4_DECODE_PV_SELECT) || defined(CAKE_DSV4_NVFP4_DECODE_PV_N16_OC1)
template __global__ void decode_pv<16>(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_out, const __grid_constant__ CUtensorMap tmap_g4d, const __grid_constant__ CUtensorMap tmap_g4f, const __grid_constant__ CUtensorMap tmap_g4dx, const __grid_constant__ CUtensorMap tmap_g4fx, __nv_bfloat16* __restrict__ q_rows, uint8_t* __restrict__ main_cache, uint8_t* __restrict__ extra_cache, int* __restrict__ main_indices, int* __restrict__ extra_indices, int* __restrict__ main_lengths, int* __restrict__ extra_lengths, float* __restrict__ sinks, float* __restrict__ bmm1_scale, float* __restrict__ bmm2_scale, __nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_lse, __nv_bfloat16* __restrict__ O, float* __restrict__ lse_out, int num_heads, int num_head_tiles, int num_splits, int num_main_tiles, int tiles_per_split, int main_width, int extra_width, int main_index_stride, int extra_index_stride, int has_main_lengths, int has_extra_lengths, int main_page_shift, int extra_page_shift, long long main_page_stride, long long extra_page_stride, int has_sinks, float lse_partial_scale, float lse_scale);
#endif
#if !defined(CAKE_DSV4_NVFP4_DECODE_PV_SELECT) || defined(CAKE_DSV4_NVFP4_DECODE_PV_N32_OC1)
template __global__ void decode_pv<32>(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_out, const __grid_constant__ CUtensorMap tmap_g4d, const __grid_constant__ CUtensorMap tmap_g4f, const __grid_constant__ CUtensorMap tmap_g4dx, const __grid_constant__ CUtensorMap tmap_g4fx, __nv_bfloat16* __restrict__ q_rows, uint8_t* __restrict__ main_cache, uint8_t* __restrict__ extra_cache, int* __restrict__ main_indices, int* __restrict__ extra_indices, int* __restrict__ main_lengths, int* __restrict__ extra_lengths, float* __restrict__ sinks, float* __restrict__ bmm1_scale, float* __restrict__ bmm2_scale, __nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_lse, __nv_bfloat16* __restrict__ O, float* __restrict__ lse_out, int num_heads, int num_head_tiles, int num_splits, int num_main_tiles, int tiles_per_split, int main_width, int extra_width, int main_index_stride, int extra_index_stride, int has_main_lengths, int has_extra_lengths, int main_page_shift, int extra_page_shift, long long main_page_stride, long long extra_page_stride, int has_sinks, float lse_partial_scale, float lse_scale);
#endif
} // namespace cake::dsv4_nvfp4
