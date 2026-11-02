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
#define TMEM_NCOLS 416
#define TMEM_TMEM_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 320
#define TMEM_TMEM_SFB_OFFSET 384
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_V0_OFF 1024
#define SMEM_SMEM_V0_STAGE_BYTES 2048
#define SMEM_SMEM_V0_STRIDE 2048
#define SMEM_SMEM_V1_OFF 3072
#define SMEM_SMEM_V1_STAGE_BYTES 2048
#define SMEM_SMEM_V1_STRIDE 2048
#define SMEM_SMEM_QR_OFF 5120
#define SMEM_SMEM_QR_STAGE_BYTES 1024
#define SMEM_SMEM_QR_STRIDE 1024
#define SMEM_SMEM_QS_OFF 6144
#define SMEM_SMEM_QS_STAGE_BYTES 512
#define SMEM_SMEM_QS_STRIDE 512
#define SMEM_SMEM_SFB_OFF 7168
#define SMEM_SMEM_SFB_STAGE_BYTES 2048
#define SMEM_SMEM_SFB_STRIDE 2048
#define SMEM_SMEM_SF_WORDS_OFF 7168
#define SMEM_SMEM_SF_WORDS_STAGE_BYTES 4096
#define SMEM_SMEM_SF_WORDS_STRIDE 4096
#define SMEM_SMEM_V6_OFF 11264
#define SMEM_SMEM_V6_STAGE_BYTES 16384
#define SMEM_SMEM_V6_STRIDE 45056
#define SMEM_SMEM_V7_OFF 27648
#define SMEM_SMEM_V7_STAGE_BYTES 16384
#define SMEM_SMEM_V7_STRIDE 45056
#define SMEM_SMEM_KS_OFF 44032
#define SMEM_SMEM_KS_STAGE_BYTES 4096
#define SMEM_SMEM_KS_STRIDE 45056
#define SMEM_SMEM_KR_OFF 48128
#define SMEM_SMEM_KR_STAGE_BYTES 8192
#define SMEM_SMEM_KR_STRIDE 45056
#define SMEM_SMEM_V_OFF 101376
#define SMEM_SMEM_V_STAGE_BYTES 16384
#define SMEM_SMEM_V_STRIDE 16384
#define SMEM_SMEM_PT_OFF 166912
#define SMEM_SMEM_PT_STAGE_BYTES 16384
#define SMEM_SMEM_PT_STRIDE 16384
#define SMEM_SMEM_PT_WORDS_OFF 166912
#define SMEM_SMEM_PT_WORDS_STAGE_BYTES 32768
#define SMEM_SMEM_PT_WORDS_STRIDE 32768
#define SMEM_SMEM_STATS_OFF 199680
#define SMEM_SMEM_STATS_STAGE_BYTES 1344
#define SMEM_SMEM_STATS_STRIDE 1344
#define SMEM_SMEM_VIS_OFF 201024
#define SMEM_SMEM_VIS_STAGE_BYTES 64
#define SMEM_SMEM_VIS_STRIDE 64
#define SMEM_SMEM_FLAGS_OFF 201088
#define SMEM_SMEM_FLAGS_STAGE_BYTES 64
#define SMEM_SMEM_FLAGS_STRIDE 64
#define SMEM_SMEM_STATS_W_OFF 199680
#define SMEM_SMEM_STATS_W_STAGE_BYTES 1344
#define SMEM_SMEM_STATS_W_STRIDE 1344
#define SMEM_SMEM_VIS_W_OFF 201024
#define SMEM_SMEM_VIS_W_STAGE_BYTES 64
#define SMEM_SMEM_VIS_W_STRIDE 64
#define SMEM_SMEM_FLAGS_W_OFF 201088
#define SMEM_SMEM_FLAGS_W_STAGE_BYTES 64
#define SMEM_SMEM_FLAGS_W_STRIDE 64
#define SMEM_SMEM_KBUF_OFF 201216
#define SMEM_SMEM_KBUF_STAGE_BYTES 2048
#define SMEM_SMEM_KBUF_STRIDE 2048
#define SMEM_TOTAL 203264
#define THREADS 640
#define LAUNCH_MIN_BLOCKS 1

extern "C" {

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) void
kernel_cake_mla_nvfp4_paged_decode_d2829e4e3bf495b79161(const __grid_constant__ CUtensorMap tmap_qn, const __grid_constant__ CUtensorMap tmap_qs, const __grid_constant__ CUtensorMap tmap_qr, const __grid_constant__ CUtensorMap tmap_k, const __grid_constant__ CUtensorMap tmap_ks, const __grid_constant__ CUtensorMap tmap_kr, float* __restrict__ q_scale, __nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_max, float* __restrict__ partial_sum, float* __restrict__ lse, int* __restrict__ seq_lens, int* __restrict__ kv_len_global, int* __restrict__ cum_seq_lens_q, int* __restrict__ page_table, float softmax_scale_log2, float bmm2_scale, int num_heads, int num_split, int max_pages_per_seq, int page_shift, int cp_world, int cp_rank, int has_lse)
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
    #define sfb_full_addr (mbar_base + 8)
    #define kv_full_addr (mbar_base + 16)
    #define kv_empty_qk_addr (mbar_base + 32)
    #define kv_empty_tr_addr (mbar_base + 48)
    #define sfa_full_addr (mbar_base + 64)
    #define sfa_empty_addr (mbar_base + 80)
    #define k_full_addr (mbar_base + 96)
    #define k_empty_addr (mbar_base + 128)
    #define v_full_addr (mbar_base + 160)
    #define v_empty_addr (mbar_base + 192)
    #define s_full_addr (mbar_base + 224)
    #define s_free_addr (mbar_base + 240)
    #define p_full_addr (mbar_base + 256)
    #define pt_free_addr (mbar_base + 272)
    #define pv_done_addr (mbar_base + 288)
    #define pv_issued_addr (mbar_base + 320)
    #define o_done_addr (mbar_base + 336)
    #define tmem_dealloc_addr (mbar_base + 344)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* smem_v0 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V0_OFF);
    const int smem_v0_addr = smem + SMEM_SMEM_V0_OFF;
    uint8_t* smem_v1 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V1_OFF);
    const int smem_v1_addr = smem + SMEM_SMEM_V1_OFF;
    uint8_t* smem_qr = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_QR_OFF);
    const int smem_qr_addr = smem + SMEM_SMEM_QR_OFF;
    uint8_t* smem_qs = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_QS_OFF);
    const int smem_qs_addr = smem + SMEM_SMEM_QS_OFF;
    uint8_t* smem_sfb = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_SFB_OFF);
    const int smem_sfb_addr = smem + SMEM_SMEM_SFB_OFF;
    unsigned int* smem_sf_words = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_SF_WORDS_OFF);
    const int smem_sf_words_addr = smem + SMEM_SMEM_SF_WORDS_OFF;
    uint8_t* smem_v6 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V6_OFF);
    const int smem_v6_addr = smem + SMEM_SMEM_V6_OFF;
    uint8_t* smem_v7 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V7_OFF);
    const int smem_v7_addr = smem + SMEM_SMEM_V7_OFF;
    uint8_t* smem_ks = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_KS_OFF);
    const int smem_ks_addr = smem + SMEM_SMEM_KS_OFF;
    uint8_t* smem_kr = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_KR_OFF);
    const int smem_kr_addr = smem + SMEM_SMEM_KR_OFF;
    uint8_t* smem_v = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V_OFF);
    const int smem_v_addr = smem + SMEM_SMEM_V_OFF;
    uint8_t* smem_pt = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_PT_OFF);
    const int smem_pt_addr = smem + SMEM_SMEM_PT_OFF;
    unsigned int* smem_pt_words = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_PT_WORDS_OFF);
    const int smem_pt_words_addr = smem + SMEM_SMEM_PT_WORDS_OFF;
    float* smem_stats = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_STATS_OFF);
    const int smem_stats_addr = smem + SMEM_SMEM_STATS_OFF;
    int* smem_vis = reinterpret_cast<int*>(smem_raw + SMEM_SMEM_VIS_OFF);
    const int smem_vis_addr = smem + SMEM_SMEM_VIS_OFF;
    int* smem_flags = reinterpret_cast<int*>(smem_raw + SMEM_SMEM_FLAGS_OFF);
    const int smem_flags_addr = smem + SMEM_SMEM_FLAGS_OFF;
    unsigned int* smem_stats_w = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_STATS_W_OFF);
    const int smem_stats_w_addr = smem + SMEM_SMEM_STATS_W_OFF;
    unsigned int* smem_vis_w = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_VIS_W_OFF);
    const int smem_vis_w_addr = smem + SMEM_SMEM_VIS_W_OFF;
    unsigned int* smem_flags_w = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_FLAGS_W_OFF);
    const int smem_flags_w_addr = smem + SMEM_SMEM_FLAGS_W_OFF;
    float* smem_kbuf = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_KBUF_OFF);
    const int smem_kbuf_addr = smem + SMEM_SMEM_KBUF_OFF;

    // Mbarrier init (19 pipeline groups, 0 ordered-sequence groups, 44 barriers)
    // Mbarriers at smem_raw[0..352)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // sfb_full: 1 barriers, init_count=128
            mbarrier_init(smem + 8, 128);
            // kv_full: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // kv_empty_qk: 2 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // kv_empty_tr: 2 barriers, init_count=256
            mbarrier_init(smem + 48, 256);
            mbarrier_init(smem + 56, 256);
            // sfa_full: 2 barriers, init_count=256
            mbarrier_init(smem + 64, 256);
            mbarrier_init(smem + 72, 256);
            // sfa_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // k_full: 4 barriers, init_count=128
            mbarrier_init(smem + 96, 128);
            mbarrier_init(smem + 104, 128);
            mbarrier_init(smem + 112, 128);
            mbarrier_init(smem + 120, 128);
            // k_empty: 4 barriers, init_count=128
            mbarrier_init(smem + 128, 128);
            mbarrier_init(smem + 136, 128);
            mbarrier_init(smem + 144, 128);
            mbarrier_init(smem + 152, 128);
            // v_full: 4 barriers, init_count=128
            mbarrier_init(smem + 160, 128);
            mbarrier_init(smem + 168, 128);
            mbarrier_init(smem + 176, 128);
            mbarrier_init(smem + 184, 128);
            // v_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            // s_free: 2 barriers, init_count=128
            mbarrier_init(smem + 240, 128);
            mbarrier_init(smem + 248, 128);
            // p_full: 2 barriers, init_count=128
            mbarrier_init(smem + 256, 128);
            mbarrier_init(smem + 264, 128);
            // pt_free: 2 barriers, init_count=1
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            // pv_done: 4 barriers, init_count=1
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            // pv_issued: 2 barriers, init_count=1
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            // o_done: 1 barriers, init_count=1
            mbarrier_init(smem + 336, 1);
            // tmem_dealloc: 1 barriers, init_count=608
            mbarrier_init(smem + 344, 608);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 416 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 352);
    if (warp == 0) {
        int _tmem_hold = smem + 352;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem = taddr;
    const int tmem_tmem_sfa = taddr + 320;
    const int tmem_tmem_sfb = taddr + 384;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 16 && warp <= 19) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 32;");
    }

    // ---- Role: softmax_wg0 ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 112;");
        { // softmax_wg0_main
            float psum0[8];
            float psum1[8];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                psum0[j] = 0.0f;
            }
            #pragma unroll
            for (int j_1 = 0; j_1 < 8; j_1++) {
                psum1[j_1] = 0.0f;
            }
            int split_idx = blockIdx.x;
            int m_tile = gridDim.y - 1 - blockIdx.y;
            int b = blockIdx.z;
            int q_start = cum_seq_lens_q[b];
            int q_len_b = cum_seq_lens_q[b + 1] - q_start;
            int kv_len = seq_lens[b];
            int g_len = kv_len_global[b];
            int rows_b = q_len_b * num_heads;
            int row0 = m_tile * 16;
            int rows_left = rows_b - row0;
            int rows_pos = ((rows_left < 0) ? 0 : rows_left);
            int rows_valid = ((rows_pos > 16) ? 16 : rows_pos);
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
            int my_start = split_idx * tiles_per_split;
            int my_end_raw = my_start + tiles_per_split;
            int my_end = ((my_end_raw > n_tiles_total) ? n_tiles_total : my_end_raw);
            int my_n_raw = my_end - my_start;
            int my_n_tiles = ((my_n_raw < 0) ? 0 : my_n_raw);
            int pt_base = b * max_pages_per_seq;
            const int warp_in_wg = warp % 4;
            const int lane_base = warp_in_wg * 32;
            const int my_tok = lane_base + lane;
            int red_col = warp_in_wg * 4 + lane;
            int is_reducer = lane < 4;
            int row_stride = num_split * 512;
            int out_base = (row_base_global * num_split + split_idx) * 512 + my_tok;
            if (is_reducer != 0) {
                smem_stats[96 + red_col] = 1027.8073549220576f;
                int q_tok = (row0 + red_col) / num_heads;
                int num_0 = g_len - q_len_b + q_tok - cp_rank;
                int vis_raw_1 = num_0 / cp_world + 1;
                int vis_cap_2 = ((vis_raw_1 > kv_len) ? kv_len : vis_raw_1);
                int vis_out_3 = ((num_0 < 0) ? 0 : vis_cap_2);
                smem_vis[red_col] = vis_out_3;
                int qs_ld = ((red_col < rows_valid) ? row_base_global + red_col : row_base_global);
                float qs_raw = q_scale[qs_ld];
                float qs_v = ((red_col < rows_valid) ? qs_raw : 1.0f);
                smem_stats[256 + red_col] = softmax_scale_log2 * qs_v;
            }
            asm volatile("barrier.sync 10, 256;" ::: "memory");
            int vis_min = smem_vis[0];
            int n_mine = (my_n_tiles + 2 - 1) / 2;
            #pragma unroll 1
            for (int it = 0; it < n_mine; it++) {
                int tile = it * 2;
                int pphase = tile & 1;
                int aphase = it & 1;
                int sphase = tile & 1;
                int kb = tile & 3;
                int s_wait = tile >> 1 & 1;
                mbarrier_wait(s_full_addr + (sphase) * 8, s_wait);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int t_abs = (my_start + tile) * 128 + my_tok;
                float _tmem_load_0[8];
                tmem_ld_x8(&_tmem_load_0[0], taddr + (unsigned int)(sphase * 32) + (unsigned int)(lane_base << 16));
                float _tmem_load_1[8];
                tmem_ld_x8(&_tmem_load_1[0], taddr + (unsigned int)(sphase * 32) + 8 + (unsigned int)(lane_base << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(s_free_addr + (sphase) * 8);
                int tile_last = (my_start + tile) * 128 + 127;
                if (tile_last >= vis_min) {
                    #pragma unroll
                    for (int k = 0; k < 8; k += 4) {
                        uint32_t _smem_vis_w_reg_0[4];
                        __int128_t _smem_b128_0;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_0) : "r"(smem_vis_w_addr + (k) * 4));
                        _smem_vis_w_reg_0[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_0)[0];
                        _smem_vis_w_reg_0[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_0)[1];
                        _smem_vis_w_reg_0[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_0)[2];
                        _smem_vis_w_reg_0[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_0)[3];
                        #pragma unroll
                        for (int e = 0; e < 4; e++) {
                            int vis_e = 0;
                            vis_e = reinterpret_cast<int*>(&_smem_vis_w_reg_0[e])[0];
                            if (t_abs >= vis_e) {
                                _tmem_load_0[k + e] = -CAKE_INF;
                            }
                        }
                    }
                    #pragma unroll
                    for (int k_1 = 0; k_1 < 8; k_1 += 4) {
                        uint32_t _smem_vis_w_reg_1[4];
                        __int128_t _smem_b128_1;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_1) : "r"(smem_vis_w_addr + (8 + k_1) * 4));
                        _smem_vis_w_reg_1[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_1)[0];
                        _smem_vis_w_reg_1[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_1)[1];
                        _smem_vis_w_reg_1[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_1)[2];
                        _smem_vis_w_reg_1[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_1)[3];
                        #pragma unroll
                        for (int e_1 = 0; e_1 < 4; e_1++) {
                            int vis_e_1 = 0;
                            vis_e_1 = reinterpret_cast<int*>(&_smem_vis_w_reg_1[e_1])[0];
                            if (t_abs >= vis_e_1) {
                                _tmem_load_1[k_1 + e_1] = -CAKE_INF;
                            }
                        }
                    }
                }
                float pm = -CAKE_INF;
                #pragma unroll
                for (int k_2 = 0; k_2 < 8; k_2 += 4) {
                    uint32_t _smem_stats_w_reg_0[4];
                    __int128_t _smem_b128_2;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_2) : "r"(smem_stats_w_addr + (96 + k_2) * 4));
                    _smem_stats_w_reg_0[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_2)[0];
                    _smem_stats_w_reg_0[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_2)[1];
                    _smem_stats_w_reg_0[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_2)[2];
                    _smem_stats_w_reg_0[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_2)[3];
                    uint32_t _smem_stats_w_reg_1[4];
                    __int128_t _smem_b128_3;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_3) : "r"(smem_stats_w_addr + (256 + k_2) * 4));
                    _smem_stats_w_reg_1[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_3)[0];
                    _smem_stats_w_reg_1[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_3)[1];
                    _smem_stats_w_reg_1[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_3)[2];
                    _smem_stats_w_reg_1[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_3)[3];
                    #pragma unroll
                    for (int e_2 = 0; e_2 < 4; e_2++) {
                        _tmem_load_0[k_2 + e_2] = _tmem_load_0[k_2 + e_2] * __uint_as_float(_smem_stats_w_reg_1[e_2]) + __uint_as_float(_smem_stats_w_reg_0[e_2]);
                        float _max_0 = max_noftz(pm, _tmem_load_0[k_2 + e_2]);
                        pm = _max_0;
                    }
                }
                float pm_0 = -CAKE_INF;
                #pragma unroll
                for (int k_3 = 0; k_3 < 8; k_3 += 4) {
                    uint32_t _smem_stats_w_reg_2[4];
                    __int128_t _smem_b128_4;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_4) : "r"(smem_stats_w_addr + (104 + k_3) * 4));
                    _smem_stats_w_reg_2[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_4)[0];
                    _smem_stats_w_reg_2[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_4)[1];
                    _smem_stats_w_reg_2[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_4)[2];
                    _smem_stats_w_reg_2[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_4)[3];
                    uint32_t _smem_stats_w_reg_3[4];
                    __int128_t _smem_b128_5;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_5) : "r"(smem_stats_w_addr + (264 + k_3) * 4));
                    _smem_stats_w_reg_3[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_5)[0];
                    _smem_stats_w_reg_3[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_5)[1];
                    _smem_stats_w_reg_3[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_5)[2];
                    _smem_stats_w_reg_3[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_5)[3];
                    #pragma unroll
                    for (int e_3 = 0; e_3 < 4; e_3++) {
                        _tmem_load_1[k_3 + e_3] = _tmem_load_1[k_3 + e_3] * __uint_as_float(_smem_stats_w_reg_3[e_3]) + __uint_as_float(_smem_stats_w_reg_2[e_3]);
                        float _max_1 = max_noftz(pm_0, _tmem_load_1[k_3 + e_3]);
                        pm_0 = _max_1;
                    }
                }
                float _max_2 = max_noftz(pm, pm_0);
                float pm_1 = _max_2;
                int _vote_0 = __any_sync(0xFFFFFFFF, pm_1 > 5.807354922057604f);
                int warp_exceed = _vote_0;
                if (lane == 0) {
                    smem_flags[aphase * 8 + warp_in_wg] = warp_exceed;
                }
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                uint32_t _smem_flags_w_reg_0[4];
                __int128_t _smem_b128_6;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_6) : "r"(smem_flags_w_addr + (aphase * 8) * 4));
                _smem_flags_w_reg_0[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_6)[0];
                _smem_flags_w_reg_0[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_6)[1];
                _smem_flags_w_reg_0[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_6)[2];
                _smem_flags_w_reg_0[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_6)[3];
                unsigned int f_or = _smem_flags_w_reg_0[0] | _smem_flags_w_reg_0[1] | (_smem_flags_w_reg_0[2] | _smem_flags_w_reg_0[3]);
                int wg_exceed = 0;
                wg_exceed = reinterpret_cast<int*>(&f_or)[0];
                if (wg_exceed != 0) {
                    float red[8];
                    #pragma unroll
                    for (int j_2 = 0; j_2 < 8; j_2++) {
                        red[j_2] = _tmem_load_0[j_2];
                    }
                    int upper = (lane & 16) != 0;
                    #pragma unroll
                    for (int j_3 = 0; j_3 < 4; j_3++) {
                        float send = ((upper != 0) ? red[j_3] : red[j_3 + 4]);
                        float keep = ((upper != 0) ? red[j_3 + 4] : red[j_3]);
                        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, send, 16);
                        float recv = _shfl_xor_0;
                        float _max_3 = max_noftz(keep, recv);
                        red[j_3] = _max_3;
                    }
                    int upper_0 = (lane & 8) != 0;
                    #pragma unroll
                    for (int j_4 = 0; j_4 < 2; j_4++) {
                        float send_1 = ((upper_0 != 0) ? red[j_4] : red[j_4 + 2]);
                        float keep_1 = ((upper_0 != 0) ? red[j_4 + 2] : red[j_4]);
                        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, send_1, 8);
                        float recv_1 = _shfl_xor_1;
                        float _max_4 = max_noftz(keep_1, recv_1);
                        red[j_4] = _max_4;
                    }
                    int upper_1 = (lane & 4) != 0;
                    #pragma unroll
                    for (int j_5 = 0; j_5 < 1; j_5++) {
                        float send_2 = ((upper_1 != 0) ? red[j_5] : red[j_5 + 1]);
                        float keep_2 = ((upper_1 != 0) ? red[j_5 + 1] : red[j_5]);
                        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, send_2, 4);
                        float recv_2 = _shfl_xor_2;
                        float _max_5 = max_noftz(keep_2, recv_2);
                        red[j_5] = _max_5;
                    }
                    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, red[0], 2);
                    float other = _shfl_xor_3;
                    float _max_6 = max_noftz(red[0], other);
                    red[0] = _max_6;
                    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, red[0], 1);
                    float other_2 = _shfl_xor_4;
                    float _max_7 = max_noftz(red[0], other_2);
                    red[0] = _max_7;
                    if ((lane & 3) == 0) {
                        smem_stats[32 + warp_in_wg * 16 + (lane >> 2)] = red[0];
                    }
                    float red_3[8];
                    #pragma unroll
                    for (int j_6 = 0; j_6 < 8; j_6++) {
                        red_3[j_6] = _tmem_load_1[j_6];
                    }
                    int upper_4 = (lane & 16) != 0;
                    #pragma unroll
                    for (int j_7 = 0; j_7 < 4; j_7++) {
                        float send_3 = ((upper_4 != 0) ? red_3[j_7] : red_3[j_7 + 4]);
                        float keep_3 = ((upper_4 != 0) ? red_3[j_7 + 4] : red_3[j_7]);
                        float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, send_3, 16);
                        float recv_3 = _shfl_xor_5;
                        float _max_8 = max_noftz(keep_3, recv_3);
                        red_3[j_7] = _max_8;
                    }
                    int upper_5 = (lane & 8) != 0;
                    #pragma unroll
                    for (int j_8 = 0; j_8 < 2; j_8++) {
                        float send_4 = ((upper_5 != 0) ? red_3[j_8] : red_3[j_8 + 2]);
                        float keep_4 = ((upper_5 != 0) ? red_3[j_8 + 2] : red_3[j_8]);
                        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, send_4, 8);
                        float recv_4 = _shfl_xor_6;
                        float _max_9 = max_noftz(keep_4, recv_4);
                        red_3[j_8] = _max_9;
                    }
                    int upper_6 = (lane & 4) != 0;
                    #pragma unroll
                    for (int j_9 = 0; j_9 < 1; j_9++) {
                        float send_5 = ((upper_6 != 0) ? red_3[j_9] : red_3[j_9 + 1]);
                        float keep_5 = ((upper_6 != 0) ? red_3[j_9 + 1] : red_3[j_9]);
                        float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, send_5, 4);
                        float recv_5 = _shfl_xor_7;
                        float _max_10 = max_noftz(keep_5, recv_5);
                        red_3[j_9] = _max_10;
                    }
                    float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, red_3[0], 2);
                    float other_7 = _shfl_xor_8;
                    float _max_11 = max_noftz(red_3[0], other_7);
                    red_3[0] = _max_11;
                    float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, red_3[0], 1);
                    float other_8 = _shfl_xor_9;
                    float _max_12 = max_noftz(red_3[0], other_8);
                    red_3[0] = _max_12;
                    if ((lane & 3) == 0) {
                        smem_stats[32 + warp_in_wg * 16 + 8 + (lane >> 2)] = red_3[0];
                    }
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    if (is_reducer != 0) {
                        float m0 = smem_stats[32 + red_col];
                        float m1 = smem_stats[48 + red_col];
                        float m2 = smem_stats[64 + red_col];
                        float m3 = smem_stats[80 + red_col];
                        float _max_13 = max_noftz(m0, m1);
                        float _max_14 = max_noftz(m2, m3);
                        float _max_15 = max_noftz(_max_13, _max_14);
                        float tm = _max_15;
                        float c_old = smem_stats[96 + red_col];
                        float c_upd = 3.8073549220576037f - tm + c_old;
                        float c_new = ((tm > 5.807354922057604f) ? c_upd : c_old);
                        float delta = c_new - c_old;
                        float _exp2_0 = approx_exp2(delta);
                        float alpha = _exp2_0;
                        smem_stats[96 + red_col] = c_new;
                        smem_stats[16 + red_col] = delta;
                        smem_stats[red_col] = alpha;
                    }
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    #pragma unroll
                    for (int k_4 = 0; k_4 < 8; k_4 += 4) {
                        uint32_t _smem_stats_w_reg_4[4];
                        __int128_t _smem_b128_7;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_7) : "r"(smem_stats_w_addr + (16 + k_4) * 4));
                        _smem_stats_w_reg_4[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_7)[0];
                        _smem_stats_w_reg_4[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_7)[1];
                        _smem_stats_w_reg_4[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_7)[2];
                        _smem_stats_w_reg_4[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_7)[3];
                        uint32_t _smem_stats_w_reg_5[4];
                        __int128_t _smem_b128_8;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_8) : "r"(smem_stats_w_addr + (k_4) * 4));
                        _smem_stats_w_reg_5[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_8)[0];
                        _smem_stats_w_reg_5[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_8)[1];
                        _smem_stats_w_reg_5[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_8)[2];
                        _smem_stats_w_reg_5[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_8)[3];
                        #pragma unroll
                        for (int e_4 = 0; e_4 < 4; e_4++) {
                            _tmem_load_0[k_4 + e_4] = _tmem_load_0[k_4 + e_4] + __uint_as_float(_smem_stats_w_reg_4[e_4]);
                            psum0[k_4 + e_4] = psum0[k_4 + e_4] * __uint_as_float(_smem_stats_w_reg_5[e_4]);
                        }
                    }
                    #pragma unroll
                    for (int k_5 = 0; k_5 < 8; k_5 += 4) {
                        uint32_t _smem_stats_w_reg_6[4];
                        __int128_t _smem_b128_9;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_9) : "r"(smem_stats_w_addr + (24 + k_5) * 4));
                        _smem_stats_w_reg_6[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_9)[0];
                        _smem_stats_w_reg_6[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_9)[1];
                        _smem_stats_w_reg_6[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_9)[2];
                        _smem_stats_w_reg_6[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_9)[3];
                        uint32_t _smem_stats_w_reg_7[4];
                        __int128_t _smem_b128_10;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_10) : "r"(smem_stats_w_addr + (8 + k_5) * 4));
                        _smem_stats_w_reg_7[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_10)[0];
                        _smem_stats_w_reg_7[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_10)[1];
                        _smem_stats_w_reg_7[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_10)[2];
                        _smem_stats_w_reg_7[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_10)[3];
                        #pragma unroll
                        for (int e_5 = 0; e_5 < 4; e_5++) {
                            _tmem_load_1[k_5 + e_5] = _tmem_load_1[k_5 + e_5] + __uint_as_float(_smem_stats_w_reg_6[e_5]);
                            psum1[k_5 + e_5] = psum1[k_5 + e_5] * __uint_as_float(_smem_stats_w_reg_7[e_5]);
                        }
                    }
                }
                if (tile >= 2) {
                    mbarrier_wait(pt_free_addr + (pphase) * 8, (tile >> 1) - 1 & 1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                }
                mbarrier_wait(k_full_addr + (kb) * 8, tile >> 2 & 1);
                float pk = smem_kbuf[kb * 128 + my_tok];
                #pragma unroll
                for (int j_10 = 0; j_10 < 8; j_10++) {
                    float _exp2_1 = approx_exp2(_tmem_load_0[j_10]);
                    _tmem_load_0[j_10] = _exp2_1;
                    psum0[j_10] = psum0[j_10] + _tmem_load_0[j_10];
                    _tmem_load_0[j_10] = _tmem_load_0[j_10] * pk;
                }
                unsigned int packed[2];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(_tmem_load_0[0]), "f"(_tmem_load_0[1]),
                                           "f"(_tmem_load_0[2]), "f"(_tmem_load_0[3]));
                    packed[0] = _packed;
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
                        : "=r"(_packed) : "f"(_tmem_load_0[4]), "f"(_tmem_load_0[5]),
                                           "f"(_tmem_load_0[6]), "f"(_tmem_load_0[7]));
                    packed[1] = _packed;
                }
                int row_addr = smem_pt_addr + (unsigned int)(pphase * 16384) + (unsigned int)(my_tok * 128);
                int row_rel = pphase * 16384 + my_tok * 128;
                int dst = row_addr + ((0 ^ my_tok & 7) << 4);
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(dst), "r"(packed[0]), "r"(packed[1]) : "memory");
                #pragma unroll
                for (int j_11 = 0; j_11 < 8; j_11++) {
                    float _exp2_2 = approx_exp2(_tmem_load_1[j_11]);
                    _tmem_load_1[j_11] = _exp2_2;
                    psum1[j_11] = psum1[j_11] + _tmem_load_1[j_11];
                    _tmem_load_1[j_11] = _tmem_load_1[j_11] * pk;
                }
                unsigned int packed_2[2];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(_tmem_load_1[0]), "f"(_tmem_load_1[1]),
                                           "f"(_tmem_load_1[2]), "f"(_tmem_load_1[3]));
                    packed_2[0] = _packed;
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
                        : "=r"(_packed) : "f"(_tmem_load_1[4]), "f"(_tmem_load_1[5]),
                                           "f"(_tmem_load_1[6]), "f"(_tmem_load_1[7]));
                    packed_2[1] = _packed;
                }
                int row_addr_3 = smem_pt_addr + (unsigned int)(pphase * 16384) + (unsigned int)(my_tok * 128);
                int row_rel_4 = pphase * 16384 + my_tok * 128;
                int dst_5 = row_addr_3 + ((0 ^ my_tok & 7) << 4) + 8;
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(dst_5), "r"(packed_2[0]), "r"(packed_2[1]) : "memory");
                mbarrier_arrive(k_empty_addr + (kb) * 8);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (wg_exceed != 0) {
                    if (it > 0) {
                        mbarrier_wait(pv_done_addr + (it - 1 & 1) * 8, it - 1 >> 1 & 1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        #pragma unroll
                        for (int vs = 0; vs < 4; vs++) {
                            int o_a = taddr + 64 + (unsigned int)(vs * 32) + (unsigned int)(lane_base << 16);
                            int o_b = taddr + 64 + (unsigned int)(vs * 32) + 8 + (unsigned int)(lane_base << 16);
                            float _tmem_load_2[8];
                            tmem_ld_x8(&_tmem_load_2[0], o_a);
                            float _tmem_load_3[8];
                            tmem_ld_x8(&_tmem_load_3[0], o_b);
                            float vals[8];
                            #pragma unroll
                            for (int k_6 = 0; k_6 < 8; k_6 += 4) {
                                uint32_t _smem_stats_w_reg_8[4];
                                __int128_t _smem_b128_11;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_11) : "r"(smem_stats_w_addr + (k_6) * 4));
                                _smem_stats_w_reg_8[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_11)[0];
                                _smem_stats_w_reg_8[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_11)[1];
                                _smem_stats_w_reg_8[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_11)[2];
                                _smem_stats_w_reg_8[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_11)[3];
                                #pragma unroll
                                for (int e_6 = 0; e_6 < 4; e_6++) {
                                    vals[k_6 + e_6] = __uint_as_float(_smem_stats_w_reg_8[e_6]);
                                }
                            }
                            float vals_0[8];
                            #pragma unroll
                            for (int k_7 = 0; k_7 < 8; k_7 += 4) {
                                uint32_t _smem_stats_w_reg_9[4];
                                __int128_t _smem_b128_12;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_12) : "r"(smem_stats_w_addr + (8 + k_7) * 4));
                                _smem_stats_w_reg_9[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_12)[0];
                                _smem_stats_w_reg_9[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_12)[1];
                                _smem_stats_w_reg_9[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_12)[2];
                                _smem_stats_w_reg_9[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_12)[3];
                                #pragma unroll
                                for (int e_7 = 0; e_7 < 4; e_7++) {
                                    vals_0[k_7 + e_7] = __uint_as_float(_smem_stats_w_reg_9[e_7]);
                                }
                            }
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            #pragma unroll
                            for (int j_12 = 0; j_12 < 8; j_12++) {
                                _tmem_load_2[j_12] = _tmem_load_2[j_12] * vals[j_12];
                            }
                            #pragma unroll
                            for (int j_13 = 0; j_13 < 8; j_13++) {
                                _tmem_load_3[j_13] = _tmem_load_3[j_13] * vals_0[j_13];
                            }
                            tmem_st_x8_f32(o_a, _tmem_load_2);
                            tmem_st_x8_f32(o_b, _tmem_load_3);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                }
                mbarrier_arrive(p_full_addr + (pphase) * 8);
                if (wg_exceed == 0) {
                    if (it > 0) {
                        mbarrier_wait(pv_done_addr + (it - 1 & 1) * 8, it - 1 >> 1 & 1);
                    }
                }
            }
            #pragma unroll
            for (int j_14 = 0; j_14 < 8; j_14++) {
                float _warp_reduce_0 = psum0[j_14];
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
                float wsum_j = _warp_reduce_0;
                if (lane == j_14 % 32) {
                    smem_stats[32 + warp_in_wg * 16 + j_14] = wsum_j;
                }
            }
            #pragma unroll
            for (int j_15 = 0; j_15 < 8; j_15++) {
                float _warp_reduce_1 = psum1[j_15];
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
                float wsum_j_1 = _warp_reduce_1;
                if (lane == (8 + j_15) % 32) {
                    smem_stats[32 + warp_in_wg * 16 + 8 + j_15] = wsum_j_1;
                }
            }
            asm volatile("barrier.sync 8, 128;" ::: "memory");
            float csum = 0.0f;
            if (is_reducer != 0) {
                float s0 = smem_stats[32 + red_col];
                float s1 = smem_stats[48 + red_col];
                float s2 = smem_stats[64 + red_col];
                float s3 = smem_stats[80 + red_col];
                csum = s0 + s1 + (s2 + s3);
            }
            if (is_reducer != 0) {
                float c_fin = smem_stats[96 + red_col];
                smem_stats[272 + red_col] = 3.8073549220576037f - c_fin;
                smem_stats[304 + red_col] = csum;
            }
            asm volatile("barrier.sync 10, 256;" ::: "memory");
            if (is_reducer != 0) {
                float r0 = smem_stats[272 + red_col];
                float r1 = smem_stats[288 + red_col];
                float u0 = smem_stats[304 + red_col];
                float u1 = smem_stats[320 + red_col];
                float r0m = ((u0 > 0.0f) ? r0 : -CAKE_INF);
                float r1m = ((u1 > 0.0f) ? r1 : -CAKE_INF);
                float _max_16 = max_noftz(r0m, r1m);
                float mm = _max_16;
                float _exp2_3 = approx_exp2(r0 - mm);
                float e0 = ((u0 > 0.0f) ? _exp2_3 : 0.0f);
                float _exp2_4 = approx_exp2(r1 - mm);
                float e1 = ((u1 > 0.0f) ? _exp2_4 : 0.0f);
                float den = e0 * u0 + e1 * u1;
                float safe_den = ((den > 0.0f) ? den : 1.0f);
                float out_scale = ((num_split == 1) ? bmm2_scale : 1.0f);
                float _rcp_0 = approx_rcp(safe_den);
                float inv_d = _rcp_0 * out_scale;
                smem_stats[304 + red_col] = e0 * inv_d;
                smem_stats[320 + red_col] = e1 * inv_d;
                float mm_fin = ((den > 0.0f) ? mm : -1024.0f);
                if (red_col < rows_valid) {
                    int stat_off = (row_base_global + red_col) * num_split + split_idx;
                    *(reinterpret_cast<float*>(partial_max + stat_off) + (0)) = mm_fin;
                    *(reinterpret_cast<float*>(partial_sum + stat_off) + (0)) = den;
                    if (has_lse != 0) {
                        if (num_split == 1) {
                            float safe_t = ((den > 0.0f) ? den : 1.0f);
                            float _log2_0;
                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(safe_t));
                            float lse_l2 = mm_fin + _log2_0 - 3.8073549220576037f;
                            float lse_v = ((den > 0.0f) ? lse_l2 * 0.6931471805599453f : -CAKE_INF);
                            *(reinterpret_cast<float*>(lse + (row_base_global + red_col)) + (0)) = lse_v;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 10, 256;" ::: "memory");
            unsigned int _phase_o_done_0 = 0;
            if (my_n_tiles > 0) {
                mbarrier_wait(o_done_addr, _phase_o_done_0);
                _phase_o_done_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_4[8];
                tmem_ld_x8(&_tmem_load_4[0], taddr + 64 + (unsigned int)(lane_base << 16));
                float _tmem_load_5[8];
                tmem_ld_x8(&_tmem_load_5[0], taddr + 64 + 128 + (unsigned int)(lane_base << 16));
                float vals_1[8];
                #pragma unroll
                for (int k_8 = 0; k_8 < 8; k_8 += 4) {
                    uint32_t _smem_stats_w_reg_10[4];
                    __int128_t _smem_b128_13;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_13) : "r"(smem_stats_w_addr + (304 + k_8) * 4));
                    _smem_stats_w_reg_10[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_13)[0];
                    _smem_stats_w_reg_10[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_13)[1];
                    _smem_stats_w_reg_10[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_13)[2];
                    _smem_stats_w_reg_10[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_13)[3];
                    #pragma unroll
                    for (int e_8 = 0; e_8 < 4; e_8++) {
                        vals_1[k_8 + e_8] = __uint_as_float(_smem_stats_w_reg_10[e_8]);
                    }
                }
                float vals_0_1[8];
                #pragma unroll
                for (int k_9 = 0; k_9 < 8; k_9 += 4) {
                    uint32_t _smem_stats_w_reg_11[4];
                    __int128_t _smem_b128_14;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_14) : "r"(smem_stats_w_addr + (320 + k_9) * 4));
                    _smem_stats_w_reg_11[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_14)[0];
                    _smem_stats_w_reg_11[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_14)[1];
                    _smem_stats_w_reg_11[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_14)[2];
                    _smem_stats_w_reg_11[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_14)[3];
                    #pragma unroll
                    for (int e_9 = 0; e_9 < 4; e_9++) {
                        vals_0_1[k_9 + e_9] = __uint_as_float(_smem_stats_w_reg_11[e_9]);
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                #pragma unroll
                for (int j_16 = 0; j_16 < 8; j_16++) {
                    int c = j_16;
                    float a0 = vals_1[j_16];
                    float a1 = vals_0_1[j_16];
                    float t0 = ((a0 > 0.0f) ? _tmem_load_4[j_16] * a0 : 0.0f);
                    float t1 = ((a1 > 0.0f) ? _tmem_load_5[j_16] * a1 : 0.0f);
                    if (c < rows_valid) {
                        *(reinterpret_cast<__nv_bfloat16*>(partial_O + (out_base + c * row_stride)) + (0)) = __float2bfloat16_rn(t0 + t1);
                    }
                }
                float _tmem_load_6[8];
                tmem_ld_x8(&_tmem_load_6[0], taddr + 64 + 8 + (unsigned int)(lane_base << 16));
                float _tmem_load_7[8];
                tmem_ld_x8(&_tmem_load_7[0], taddr + 64 + 128 + 8 + (unsigned int)(lane_base << 16));
                float vals_1_1[8];
                #pragma unroll
                for (int k_10 = 0; k_10 < 8; k_10 += 4) {
                    uint32_t _smem_stats_w_reg_12[4];
                    __int128_t _smem_b128_15;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_15) : "r"(smem_stats_w_addr + (312 + k_10) * 4));
                    _smem_stats_w_reg_12[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_15)[0];
                    _smem_stats_w_reg_12[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_15)[1];
                    _smem_stats_w_reg_12[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_15)[2];
                    _smem_stats_w_reg_12[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_15)[3];
                    #pragma unroll
                    for (int e_10 = 0; e_10 < 4; e_10++) {
                        vals_1_1[k_10 + e_10] = __uint_as_float(_smem_stats_w_reg_12[e_10]);
                    }
                }
                float vals_2[8];
                #pragma unroll
                for (int k_11 = 0; k_11 < 8; k_11 += 4) {
                    uint32_t _smem_stats_w_reg_13[4];
                    __int128_t _smem_b128_16;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_16) : "r"(smem_stats_w_addr + (328 + k_11) * 4));
                    _smem_stats_w_reg_13[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_16)[0];
                    _smem_stats_w_reg_13[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_16)[1];
                    _smem_stats_w_reg_13[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_16)[2];
                    _smem_stats_w_reg_13[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_16)[3];
                    #pragma unroll
                    for (int e_11 = 0; e_11 < 4; e_11++) {
                        vals_2[k_11 + e_11] = __uint_as_float(_smem_stats_w_reg_13[e_11]);
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                #pragma unroll
                for (int j_17 = 0; j_17 < 8; j_17++) {
                    int c_1 = 8 + j_17;
                    float a0_1 = vals_1_1[j_17];
                    float a1_1 = vals_2[j_17];
                    float t0_1 = ((a0_1 > 0.0f) ? _tmem_load_6[j_17] * a0_1 : 0.0f);
                    float t1_1 = ((a1_1 > 0.0f) ? _tmem_load_7[j_17] * a1_1 : 0.0f);
                    if (c_1 < rows_valid) {
                        *(reinterpret_cast<__nv_bfloat16*>(partial_O + (out_base + c_1 * row_stride)) + (0)) = __float2bfloat16_rn(t0_1 + t1_1);
                    }
                }
                float _tmem_load_8[8];
                tmem_ld_x8(&_tmem_load_8[0], taddr + 64 + 32 + (unsigned int)(lane_base << 16));
                float _tmem_load_9[8];
                tmem_ld_x8(&_tmem_load_9[0], taddr + 64 + 160 + (unsigned int)(lane_base << 16));
                float vals_3[8];
                #pragma unroll
                for (int k_12 = 0; k_12 < 8; k_12 += 4) {
                    uint32_t _smem_stats_w_reg_14[4];
                    __int128_t _smem_b128_17;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_17) : "r"(smem_stats_w_addr + (304 + k_12) * 4));
                    _smem_stats_w_reg_14[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_17)[0];
                    _smem_stats_w_reg_14[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_17)[1];
                    _smem_stats_w_reg_14[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_17)[2];
                    _smem_stats_w_reg_14[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_17)[3];
                    #pragma unroll
                    for (int e_12 = 0; e_12 < 4; e_12++) {
                        vals_3[k_12 + e_12] = __uint_as_float(_smem_stats_w_reg_14[e_12]);
                    }
                }
                float vals_4[8];
                #pragma unroll
                for (int k_13 = 0; k_13 < 8; k_13 += 4) {
                    uint32_t _smem_stats_w_reg_15[4];
                    __int128_t _smem_b128_18;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_18) : "r"(smem_stats_w_addr + (320 + k_13) * 4));
                    _smem_stats_w_reg_15[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_18)[0];
                    _smem_stats_w_reg_15[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_18)[1];
                    _smem_stats_w_reg_15[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_18)[2];
                    _smem_stats_w_reg_15[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_18)[3];
                    #pragma unroll
                    for (int e_13 = 0; e_13 < 4; e_13++) {
                        vals_4[k_13 + e_13] = __uint_as_float(_smem_stats_w_reg_15[e_13]);
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                #pragma unroll
                for (int j_18 = 0; j_18 < 8; j_18++) {
                    int c_2 = j_18;
                    float a0_2 = vals_3[j_18];
                    float a1_2 = vals_4[j_18];
                    float t0_2 = ((a0_2 > 0.0f) ? _tmem_load_8[j_18] * a0_2 : 0.0f);
                    float t1_2 = ((a1_2 > 0.0f) ? _tmem_load_9[j_18] * a1_2 : 0.0f);
                    if (c_2 < rows_valid) {
                        *(reinterpret_cast<__nv_bfloat16*>(partial_O + (out_base + c_2 * row_stride + 128)) + (0)) = __float2bfloat16_rn(t0_2 + t1_2);
                    }
                }
                float _tmem_load_10[8];
                tmem_ld_x8(&_tmem_load_10[0], taddr + 64 + 32 + 8 + (unsigned int)(lane_base << 16));
                float _tmem_load_11[8];
                tmem_ld_x8(&_tmem_load_11[0], taddr + 64 + 160 + 8 + (unsigned int)(lane_base << 16));
                float vals_5[8];
                #pragma unroll
                for (int k_14 = 0; k_14 < 8; k_14 += 4) {
                    uint32_t _smem_stats_w_reg_16[4];
                    __int128_t _smem_b128_19;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_19) : "r"(smem_stats_w_addr + (312 + k_14) * 4));
                    _smem_stats_w_reg_16[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_19)[0];
                    _smem_stats_w_reg_16[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_19)[1];
                    _smem_stats_w_reg_16[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_19)[2];
                    _smem_stats_w_reg_16[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_19)[3];
                    #pragma unroll
                    for (int e_14 = 0; e_14 < 4; e_14++) {
                        vals_5[k_14 + e_14] = __uint_as_float(_smem_stats_w_reg_16[e_14]);
                    }
                }
                float vals_6[8];
                #pragma unroll
                for (int k_15 = 0; k_15 < 8; k_15 += 4) {
                    uint32_t _smem_stats_w_reg_17[4];
                    __int128_t _smem_b128_20;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_20) : "r"(smem_stats_w_addr + (328 + k_15) * 4));
                    _smem_stats_w_reg_17[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_20)[0];
                    _smem_stats_w_reg_17[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_20)[1];
                    _smem_stats_w_reg_17[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_20)[2];
                    _smem_stats_w_reg_17[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_20)[3];
                    #pragma unroll
                    for (int e_15 = 0; e_15 < 4; e_15++) {
                        vals_6[k_15 + e_15] = __uint_as_float(_smem_stats_w_reg_17[e_15]);
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                #pragma unroll
                for (int j_19 = 0; j_19 < 8; j_19++) {
                    int c_3 = 8 + j_19;
                    float a0_3 = vals_5[j_19];
                    float a1_3 = vals_6[j_19];
                    float t0_3 = ((a0_3 > 0.0f) ? _tmem_load_10[j_19] * a0_3 : 0.0f);
                    float t1_3 = ((a1_3 > 0.0f) ? _tmem_load_11[j_19] * a1_3 : 0.0f);
                    if (c_3 < rows_valid) {
                        *(reinterpret_cast<__nv_bfloat16*>(partial_O + (out_base + c_3 * row_stride + 128)) + (0)) = __float2bfloat16_rn(t0_3 + t1_3);
                    }
                }
            } else if (num_split == 1) {
                #pragma unroll
                for (int j_20 = 0; j_20 < 16; j_20++) {
                    if (rows_valid > j_20) {
                        *(reinterpret_cast<__nv_bfloat16*>(partial_O + (out_base + j_20 * row_stride)) + (0)) = __float2bfloat16_rn(0.0f);
                    }
                }
                #pragma unroll
                for (int j_21 = 0; j_21 < 16; j_21++) {
                    if (rows_valid > j_21) {
                        *(reinterpret_cast<__nv_bfloat16*>(partial_O + (out_base + j_21 * row_stride + 128)) + (0)) = __float2bfloat16_rn(0.0f);
                    }
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: softmax_wg1 ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 112;");
        { // softmax_wg1_main
            float psum0_1[8];
            float psum1_1[8];
            #pragma unroll
            for (int j_22 = 0; j_22 < 8; j_22++) {
                psum0_1[j_22] = 0.0f;
            }
            #pragma unroll
            for (int j_23 = 0; j_23 < 8; j_23++) {
                psum1_1[j_23] = 0.0f;
            }
            int split_idx_1 = blockIdx.x;
            int m_tile_1 = gridDim.y - 1 - blockIdx.y;
            int b_1 = blockIdx.z;
            int q_start_1 = cum_seq_lens_q[b_1];
            int q_len_b_1 = cum_seq_lens_q[b_1 + 1] - q_start_1;
            int kv_len_1 = seq_lens[b_1];
            int g_len_1 = kv_len_global[b_1];
            int rows_b_1 = q_len_b_1 * num_heads;
            int row0_1 = m_tile_1 * 16;
            int rows_left_1 = rows_b_1 - row0_1;
            int rows_pos_1 = ((rows_left_1 < 0) ? 0 : rows_left_1);
            int rows_valid_1 = ((rows_pos_1 > 16) ? 16 : rows_pos_1);
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
            int my_start_1 = split_idx_1 * tiles_per_split_1;
            int my_end_raw_1 = my_start_1 + tiles_per_split_1;
            int my_end_1 = ((my_end_raw_1 > n_tiles_total_1) ? n_tiles_total_1 : my_end_raw_1);
            int my_n_raw_1 = my_end_1 - my_start_1;
            int my_n_tiles_1 = ((my_n_raw_1 < 0) ? 0 : my_n_raw_1);
            int pt_base_1 = b_1 * max_pages_per_seq;
            const int warp_in_wg_1 = warp % 4;
            const int lane_base_1 = warp_in_wg_1 * 32;
            const int my_tok_1 = lane_base_1 + lane;
            int red_col_1 = warp_in_wg_1 * 4 + lane;
            int is_reducer_1 = lane < 4;
            int row_stride_1 = num_split * 512;
            int out_base_1 = (row_base_global_1 * num_split + split_idx_1) * 512 + my_tok_1;
            if (is_reducer_1 != 0) {
                smem_stats[224 + red_col_1] = 1027.8073549220576f;
            }
            asm volatile("barrier.sync 10, 256;" ::: "memory");
            int vis_min_1 = smem_vis[0];
            int n_mine_1 = (my_n_tiles_1 - 1 + 2 - 1) / 2;
            #pragma unroll 1
            for (int it_1 = 0; it_1 < n_mine_1; it_1++) {
                int tile_1 = it_1 * 2 + 1;
                int pphase_1 = tile_1 & 1;
                int aphase_1 = it_1 & 1;
                int sphase_1 = tile_1 & 1;
                int kb_1 = tile_1 & 3;
                int s_wait_1 = tile_1 >> 1 & 1;
                mbarrier_wait(s_full_addr + (sphase_1) * 8, s_wait_1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int t_abs_1 = (my_start_1 + tile_1) * 128 + my_tok_1;
                float _tmem_load_12[8];
                tmem_ld_x8(&_tmem_load_12[0], taddr + (unsigned int)(sphase_1 * 32) + (unsigned int)(lane_base_1 << 16));
                float _tmem_load_13[8];
                tmem_ld_x8(&_tmem_load_13[0], taddr + (unsigned int)(sphase_1 * 32) + 8 + (unsigned int)(lane_base_1 << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(s_free_addr + (sphase_1) * 8);
                int tile_last_1 = (my_start_1 + tile_1) * 128 + 127;
                if (tile_last_1 >= vis_min_1) {
                    #pragma unroll
                    for (int k_16 = 0; k_16 < 8; k_16 += 4) {
                        uint32_t _smem_vis_w_reg_2[4];
                        __int128_t _smem_b128_0;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_0) : "r"(smem_vis_w_addr + (k_16) * 4));
                        _smem_vis_w_reg_2[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_0)[0];
                        _smem_vis_w_reg_2[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_0)[1];
                        _smem_vis_w_reg_2[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_0)[2];
                        _smem_vis_w_reg_2[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_0)[3];
                        #pragma unroll
                        for (int e_16 = 0; e_16 < 4; e_16++) {
                            int vis_e_2 = 0;
                            vis_e_2 = reinterpret_cast<int*>(&_smem_vis_w_reg_2[e_16])[0];
                            if (t_abs_1 >= vis_e_2) {
                                _tmem_load_12[k_16 + e_16] = -CAKE_INF;
                            }
                        }
                    }
                    #pragma unroll
                    for (int k_17 = 0; k_17 < 8; k_17 += 4) {
                        uint32_t _smem_vis_w_reg_3[4];
                        __int128_t _smem_b128_1;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_1) : "r"(smem_vis_w_addr + (8 + k_17) * 4));
                        _smem_vis_w_reg_3[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_1)[0];
                        _smem_vis_w_reg_3[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_1)[1];
                        _smem_vis_w_reg_3[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_1)[2];
                        _smem_vis_w_reg_3[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_1)[3];
                        #pragma unroll
                        for (int e_17 = 0; e_17 < 4; e_17++) {
                            int vis_e_3 = 0;
                            vis_e_3 = reinterpret_cast<int*>(&_smem_vis_w_reg_3[e_17])[0];
                            if (t_abs_1 >= vis_e_3) {
                                _tmem_load_13[k_17 + e_17] = -CAKE_INF;
                            }
                        }
                    }
                }
                float pm_2 = -CAKE_INF;
                #pragma unroll
                for (int k_18 = 0; k_18 < 8; k_18 += 4) {
                    uint32_t _smem_stats_w_reg_18[4];
                    __int128_t _smem_b128_2;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_2) : "r"(smem_stats_w_addr + (224 + k_18) * 4));
                    _smem_stats_w_reg_18[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_2)[0];
                    _smem_stats_w_reg_18[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_2)[1];
                    _smem_stats_w_reg_18[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_2)[2];
                    _smem_stats_w_reg_18[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_2)[3];
                    uint32_t _smem_stats_w_reg_19[4];
                    __int128_t _smem_b128_3;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_3) : "r"(smem_stats_w_addr + (256 + k_18) * 4));
                    _smem_stats_w_reg_19[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_3)[0];
                    _smem_stats_w_reg_19[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_3)[1];
                    _smem_stats_w_reg_19[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_3)[2];
                    _smem_stats_w_reg_19[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_3)[3];
                    #pragma unroll
                    for (int e_18 = 0; e_18 < 4; e_18++) {
                        _tmem_load_12[k_18 + e_18] = _tmem_load_12[k_18 + e_18] * __uint_as_float(_smem_stats_w_reg_19[e_18]) + __uint_as_float(_smem_stats_w_reg_18[e_18]);
                        float _max_17 = max_noftz(pm_2, _tmem_load_12[k_18 + e_18]);
                        pm_2 = _max_17;
                    }
                }
                float pm_0_1 = -CAKE_INF;
                #pragma unroll
                for (int k_19 = 0; k_19 < 8; k_19 += 4) {
                    uint32_t _smem_stats_w_reg_20[4];
                    __int128_t _smem_b128_4;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_4) : "r"(smem_stats_w_addr + (232 + k_19) * 4));
                    _smem_stats_w_reg_20[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_4)[0];
                    _smem_stats_w_reg_20[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_4)[1];
                    _smem_stats_w_reg_20[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_4)[2];
                    _smem_stats_w_reg_20[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_4)[3];
                    uint32_t _smem_stats_w_reg_21[4];
                    __int128_t _smem_b128_5;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_5) : "r"(smem_stats_w_addr + (264 + k_19) * 4));
                    _smem_stats_w_reg_21[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_5)[0];
                    _smem_stats_w_reg_21[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_5)[1];
                    _smem_stats_w_reg_21[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_5)[2];
                    _smem_stats_w_reg_21[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_5)[3];
                    #pragma unroll
                    for (int e_19 = 0; e_19 < 4; e_19++) {
                        _tmem_load_13[k_19 + e_19] = _tmem_load_13[k_19 + e_19] * __uint_as_float(_smem_stats_w_reg_21[e_19]) + __uint_as_float(_smem_stats_w_reg_20[e_19]);
                        float _max_18 = max_noftz(pm_0_1, _tmem_load_13[k_19 + e_19]);
                        pm_0_1 = _max_18;
                    }
                }
                float _max_19 = max_noftz(pm_2, pm_0_1);
                float pm_1_1 = _max_19;
                int _vote_1 = __any_sync(0xFFFFFFFF, pm_1_1 > 5.807354922057604f);
                int warp_exceed_1 = _vote_1;
                if (lane == 0) {
                    smem_flags[aphase_1 * 8 + 4 + warp_in_wg_1] = warp_exceed_1;
                }
                asm volatile("barrier.sync 9, 128;" ::: "memory");
                uint32_t _smem_flags_w_reg_1[4];
                __int128_t _smem_b128_6;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_6) : "r"(smem_flags_w_addr + (aphase_1 * 8 + 4) * 4));
                _smem_flags_w_reg_1[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_6)[0];
                _smem_flags_w_reg_1[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_6)[1];
                _smem_flags_w_reg_1[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_6)[2];
                _smem_flags_w_reg_1[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_6)[3];
                unsigned int f_or_1 = _smem_flags_w_reg_1[0] | _smem_flags_w_reg_1[1] | (_smem_flags_w_reg_1[2] | _smem_flags_w_reg_1[3]);
                int wg_exceed_1 = 0;
                wg_exceed_1 = reinterpret_cast<int*>(&f_or_1)[0];
                if (wg_exceed_1 != 0) {
                    float red_1[8];
                    #pragma unroll
                    for (int j_24 = 0; j_24 < 8; j_24++) {
                        red_1[j_24] = _tmem_load_12[j_24];
                    }
                    int upper_2 = (lane & 16) != 0;
                    #pragma unroll
                    for (int j_25 = 0; j_25 < 4; j_25++) {
                        float send_6 = ((upper_2 != 0) ? red_1[j_25] : red_1[j_25 + 4]);
                        float keep_6 = ((upper_2 != 0) ? red_1[j_25 + 4] : red_1[j_25]);
                        float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, send_6, 16);
                        float recv_6 = _shfl_xor_10;
                        float _max_20 = max_noftz(keep_6, recv_6);
                        red_1[j_25] = _max_20;
                    }
                    int upper_0_1 = (lane & 8) != 0;
                    #pragma unroll
                    for (int j_26 = 0; j_26 < 2; j_26++) {
                        float send_7 = ((upper_0_1 != 0) ? red_1[j_26] : red_1[j_26 + 2]);
                        float keep_7 = ((upper_0_1 != 0) ? red_1[j_26 + 2] : red_1[j_26]);
                        float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, send_7, 8);
                        float recv_7 = _shfl_xor_11;
                        float _max_21 = max_noftz(keep_7, recv_7);
                        red_1[j_26] = _max_21;
                    }
                    int upper_1_1 = (lane & 4) != 0;
                    #pragma unroll
                    for (int j_27 = 0; j_27 < 1; j_27++) {
                        float send_8 = ((upper_1_1 != 0) ? red_1[j_27] : red_1[j_27 + 1]);
                        float keep_8 = ((upper_1_1 != 0) ? red_1[j_27 + 1] : red_1[j_27]);
                        float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, send_8, 4);
                        float recv_8 = _shfl_xor_12;
                        float _max_22 = max_noftz(keep_8, recv_8);
                        red_1[j_27] = _max_22;
                    }
                    float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, red_1[0], 2);
                    float other_1 = _shfl_xor_13;
                    float _max_23 = max_noftz(red_1[0], other_1);
                    red_1[0] = _max_23;
                    float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, red_1[0], 1);
                    float other_2_1 = _shfl_xor_14;
                    float _max_24 = max_noftz(red_1[0], other_2_1);
                    red_1[0] = _max_24;
                    if ((lane & 3) == 0) {
                        smem_stats[160 + warp_in_wg_1 * 16 + (lane >> 2)] = red_1[0];
                    }
                    float red_3_1[8];
                    #pragma unroll
                    for (int j_28 = 0; j_28 < 8; j_28++) {
                        red_3_1[j_28] = _tmem_load_13[j_28];
                    }
                    int upper_4_1 = (lane & 16) != 0;
                    #pragma unroll
                    for (int j_29 = 0; j_29 < 4; j_29++) {
                        float send_9 = ((upper_4_1 != 0) ? red_3_1[j_29] : red_3_1[j_29 + 4]);
                        float keep_9 = ((upper_4_1 != 0) ? red_3_1[j_29 + 4] : red_3_1[j_29]);
                        float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, send_9, 16);
                        float recv_9 = _shfl_xor_15;
                        float _max_25 = max_noftz(keep_9, recv_9);
                        red_3_1[j_29] = _max_25;
                    }
                    int upper_5_1 = (lane & 8) != 0;
                    #pragma unroll
                    for (int j_30 = 0; j_30 < 2; j_30++) {
                        float send_10 = ((upper_5_1 != 0) ? red_3_1[j_30] : red_3_1[j_30 + 2]);
                        float keep_10 = ((upper_5_1 != 0) ? red_3_1[j_30 + 2] : red_3_1[j_30]);
                        float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, send_10, 8);
                        float recv_10 = _shfl_xor_16;
                        float _max_26 = max_noftz(keep_10, recv_10);
                        red_3_1[j_30] = _max_26;
                    }
                    int upper_6_1 = (lane & 4) != 0;
                    #pragma unroll
                    for (int j_31 = 0; j_31 < 1; j_31++) {
                        float send_11 = ((upper_6_1 != 0) ? red_3_1[j_31] : red_3_1[j_31 + 1]);
                        float keep_11 = ((upper_6_1 != 0) ? red_3_1[j_31 + 1] : red_3_1[j_31]);
                        float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, send_11, 4);
                        float recv_11 = _shfl_xor_17;
                        float _max_27 = max_noftz(keep_11, recv_11);
                        red_3_1[j_31] = _max_27;
                    }
                    float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, red_3_1[0], 2);
                    float other_7_1 = _shfl_xor_18;
                    float _max_28 = max_noftz(red_3_1[0], other_7_1);
                    red_3_1[0] = _max_28;
                    float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, red_3_1[0], 1);
                    float other_8_1 = _shfl_xor_19;
                    float _max_29 = max_noftz(red_3_1[0], other_8_1);
                    red_3_1[0] = _max_29;
                    if ((lane & 3) == 0) {
                        smem_stats[160 + warp_in_wg_1 * 16 + 8 + (lane >> 2)] = red_3_1[0];
                    }
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    if (is_reducer_1 != 0) {
                        float m0_1 = smem_stats[160 + red_col_1];
                        float m1_1 = smem_stats[176 + red_col_1];
                        float m2_1 = smem_stats[192 + red_col_1];
                        float m3_1 = smem_stats[208 + red_col_1];
                        float _max_30 = max_noftz(m0_1, m1_1);
                        float _max_31 = max_noftz(m2_1, m3_1);
                        float _max_32 = max_noftz(_max_30, _max_31);
                        float tm_1 = _max_32;
                        float c_old_1 = smem_stats[224 + red_col_1];
                        float c_upd_1 = 3.8073549220576037f - tm_1 + c_old_1;
                        float c_new_1 = ((tm_1 > 5.807354922057604f) ? c_upd_1 : c_old_1);
                        float delta_1 = c_new_1 - c_old_1;
                        float _exp2_5 = approx_exp2(delta_1);
                        float alpha_1 = _exp2_5;
                        smem_stats[224 + red_col_1] = c_new_1;
                        smem_stats[144 + red_col_1] = delta_1;
                        smem_stats[128 + red_col_1] = alpha_1;
                    }
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    #pragma unroll
                    for (int k_20 = 0; k_20 < 8; k_20 += 4) {
                        uint32_t _smem_stats_w_reg_22[4];
                        __int128_t _smem_b128_7;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_7) : "r"(smem_stats_w_addr + (144 + k_20) * 4));
                        _smem_stats_w_reg_22[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_7)[0];
                        _smem_stats_w_reg_22[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_7)[1];
                        _smem_stats_w_reg_22[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_7)[2];
                        _smem_stats_w_reg_22[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_7)[3];
                        uint32_t _smem_stats_w_reg_23[4];
                        __int128_t _smem_b128_8;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_8) : "r"(smem_stats_w_addr + (128 + k_20) * 4));
                        _smem_stats_w_reg_23[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_8)[0];
                        _smem_stats_w_reg_23[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_8)[1];
                        _smem_stats_w_reg_23[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_8)[2];
                        _smem_stats_w_reg_23[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_8)[3];
                        #pragma unroll
                        for (int e_20 = 0; e_20 < 4; e_20++) {
                            _tmem_load_12[k_20 + e_20] = _tmem_load_12[k_20 + e_20] + __uint_as_float(_smem_stats_w_reg_22[e_20]);
                            psum0_1[k_20 + e_20] = psum0_1[k_20 + e_20] * __uint_as_float(_smem_stats_w_reg_23[e_20]);
                        }
                    }
                    #pragma unroll
                    for (int k_21 = 0; k_21 < 8; k_21 += 4) {
                        uint32_t _smem_stats_w_reg_24[4];
                        __int128_t _smem_b128_9;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_9) : "r"(smem_stats_w_addr + (152 + k_21) * 4));
                        _smem_stats_w_reg_24[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_9)[0];
                        _smem_stats_w_reg_24[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_9)[1];
                        _smem_stats_w_reg_24[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_9)[2];
                        _smem_stats_w_reg_24[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_9)[3];
                        uint32_t _smem_stats_w_reg_25[4];
                        __int128_t _smem_b128_10;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_10) : "r"(smem_stats_w_addr + (136 + k_21) * 4));
                        _smem_stats_w_reg_25[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_10)[0];
                        _smem_stats_w_reg_25[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_10)[1];
                        _smem_stats_w_reg_25[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_10)[2];
                        _smem_stats_w_reg_25[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_10)[3];
                        #pragma unroll
                        for (int e_21 = 0; e_21 < 4; e_21++) {
                            _tmem_load_13[k_21 + e_21] = _tmem_load_13[k_21 + e_21] + __uint_as_float(_smem_stats_w_reg_24[e_21]);
                            psum1_1[k_21 + e_21] = psum1_1[k_21 + e_21] * __uint_as_float(_smem_stats_w_reg_25[e_21]);
                        }
                    }
                }
                if (tile_1 >= 2) {
                    mbarrier_wait(pt_free_addr + (pphase_1) * 8, (tile_1 >> 1) - 1 & 1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                }
                mbarrier_wait(k_full_addr + (kb_1) * 8, tile_1 >> 2 & 1);
                float pk_1 = smem_kbuf[kb_1 * 128 + my_tok_1];
                #pragma unroll
                for (int j_32 = 0; j_32 < 8; j_32++) {
                    float _exp2_6 = approx_exp2(_tmem_load_12[j_32]);
                    _tmem_load_12[j_32] = _exp2_6;
                    psum0_1[j_32] = psum0_1[j_32] + _tmem_load_12[j_32];
                    _tmem_load_12[j_32] = _tmem_load_12[j_32] * pk_1;
                }
                unsigned int packed_1[2];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(_tmem_load_12[0]), "f"(_tmem_load_12[1]),
                                           "f"(_tmem_load_12[2]), "f"(_tmem_load_12[3]));
                    packed_1[0] = _packed;
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
                        : "=r"(_packed) : "f"(_tmem_load_12[4]), "f"(_tmem_load_12[5]),
                                           "f"(_tmem_load_12[6]), "f"(_tmem_load_12[7]));
                    packed_1[1] = _packed;
                }
                int row_addr_1 = smem_pt_addr + (unsigned int)(pphase_1 * 16384) + (unsigned int)(my_tok_1 * 128);
                int row_rel_1 = pphase_1 * 16384 + my_tok_1 * 128;
                int dst_1 = row_addr_1 + ((0 ^ my_tok_1 & 7) << 4);
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(dst_1), "r"(packed_1[0]), "r"(packed_1[1]) : "memory");
                #pragma unroll
                for (int j_33 = 0; j_33 < 8; j_33++) {
                    float _exp2_7 = approx_exp2(_tmem_load_13[j_33]);
                    _tmem_load_13[j_33] = _exp2_7;
                    psum1_1[j_33] = psum1_1[j_33] + _tmem_load_13[j_33];
                    _tmem_load_13[j_33] = _tmem_load_13[j_33] * pk_1;
                }
                unsigned int packed_2_1[2];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(_tmem_load_13[0]), "f"(_tmem_load_13[1]),
                                           "f"(_tmem_load_13[2]), "f"(_tmem_load_13[3]));
                    packed_2_1[0] = _packed;
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
                        : "=r"(_packed) : "f"(_tmem_load_13[4]), "f"(_tmem_load_13[5]),
                                           "f"(_tmem_load_13[6]), "f"(_tmem_load_13[7]));
                    packed_2_1[1] = _packed;
                }
                int row_addr_3_1 = smem_pt_addr + (unsigned int)(pphase_1 * 16384) + (unsigned int)(my_tok_1 * 128);
                int row_rel_4_1 = pphase_1 * 16384 + my_tok_1 * 128;
                int dst_5_1 = row_addr_3_1 + ((0 ^ my_tok_1 & 7) << 4) + 8;
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(dst_5_1), "r"(packed_2_1[0]), "r"(packed_2_1[1]) : "memory");
                mbarrier_arrive(k_empty_addr + (kb_1) * 8);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (wg_exceed_1 != 0) {
                    if (it_1 > 0) {
                        mbarrier_wait(pv_done_addr + (2 + (it_1 - 1 & 1)) * 8, it_1 - 1 >> 1 & 1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        #pragma unroll
                        for (int vs_1 = 0; vs_1 < 4; vs_1++) {
                            int o_a_1 = taddr + 192 + (unsigned int)(vs_1 * 32) + (unsigned int)(lane_base_1 << 16);
                            int o_b_1 = taddr + 192 + (unsigned int)(vs_1 * 32) + 8 + (unsigned int)(lane_base_1 << 16);
                            float _tmem_load_14[8];
                            tmem_ld_x8(&_tmem_load_14[0], o_a_1);
                            float _tmem_load_15[8];
                            tmem_ld_x8(&_tmem_load_15[0], o_b_1);
                            float vals_7[8];
                            #pragma unroll
                            for (int k_22 = 0; k_22 < 8; k_22 += 4) {
                                uint32_t _smem_stats_w_reg_26[4];
                                __int128_t _smem_b128_11;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_11) : "r"(smem_stats_w_addr + (128 + k_22) * 4));
                                _smem_stats_w_reg_26[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_11)[0];
                                _smem_stats_w_reg_26[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_11)[1];
                                _smem_stats_w_reg_26[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_11)[2];
                                _smem_stats_w_reg_26[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_11)[3];
                                #pragma unroll
                                for (int e_22 = 0; e_22 < 4; e_22++) {
                                    vals_7[k_22 + e_22] = __uint_as_float(_smem_stats_w_reg_26[e_22]);
                                }
                            }
                            float vals_0_2[8];
                            #pragma unroll
                            for (int k_23 = 0; k_23 < 8; k_23 += 4) {
                                uint32_t _smem_stats_w_reg_27[4];
                                __int128_t _smem_b128_12;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_12) : "r"(smem_stats_w_addr + (136 + k_23) * 4));
                                _smem_stats_w_reg_27[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_12)[0];
                                _smem_stats_w_reg_27[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_12)[1];
                                _smem_stats_w_reg_27[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_12)[2];
                                _smem_stats_w_reg_27[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_12)[3];
                                #pragma unroll
                                for (int e_23 = 0; e_23 < 4; e_23++) {
                                    vals_0_2[k_23 + e_23] = __uint_as_float(_smem_stats_w_reg_27[e_23]);
                                }
                            }
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            #pragma unroll
                            for (int j_34 = 0; j_34 < 8; j_34++) {
                                _tmem_load_14[j_34] = _tmem_load_14[j_34] * vals_7[j_34];
                            }
                            #pragma unroll
                            for (int j_35 = 0; j_35 < 8; j_35++) {
                                _tmem_load_15[j_35] = _tmem_load_15[j_35] * vals_0_2[j_35];
                            }
                            tmem_st_x8_f32(o_a_1, _tmem_load_14);
                            tmem_st_x8_f32(o_b_1, _tmem_load_15);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                }
                mbarrier_arrive(p_full_addr + (pphase_1) * 8);
                if (wg_exceed_1 == 0) {
                    if (it_1 > 0) {
                        mbarrier_wait(pv_done_addr + (2 + (it_1 - 1 & 1)) * 8, it_1 - 1 >> 1 & 1);
                    }
                }
            }
            #pragma unroll
            for (int j_36 = 0; j_36 < 8; j_36++) {
                float _warp_reduce_2 = psum0_1[j_36];
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_2 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_2, offset);
                float wsum_j_2 = _warp_reduce_2;
                if (lane == j_36 % 32) {
                    smem_stats[160 + warp_in_wg_1 * 16 + j_36] = wsum_j_2;
                }
            }
            #pragma unroll
            for (int j_37 = 0; j_37 < 8; j_37++) {
                float _warp_reduce_3 = psum1_1[j_37];
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_3 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_3, offset);
                float wsum_j_3 = _warp_reduce_3;
                if (lane == (8 + j_37) % 32) {
                    smem_stats[160 + warp_in_wg_1 * 16 + 8 + j_37] = wsum_j_3;
                }
            }
            asm volatile("barrier.sync 9, 128;" ::: "memory");
            float csum_1 = 0.0f;
            if (is_reducer_1 != 0) {
                float s0_1 = smem_stats[160 + red_col_1];
                float s1_1 = smem_stats[176 + red_col_1];
                float s2_1 = smem_stats[192 + red_col_1];
                float s3_1 = smem_stats[208 + red_col_1];
                csum_1 = s0_1 + s1_1 + (s2_1 + s3_1);
            }
            if (is_reducer_1 != 0) {
                float c_fin_1 = smem_stats[224 + red_col_1];
                smem_stats[288 + red_col_1] = 3.8073549220576037f - c_fin_1;
                smem_stats[320 + red_col_1] = csum_1;
            }
            asm volatile("barrier.sync 10, 256;" ::: "memory");
            asm volatile("barrier.sync 10, 256;" ::: "memory");
            unsigned int _phase_o_done_0_1 = 0;
            if (my_n_tiles_1 > 0) {
                mbarrier_wait(o_done_addr, _phase_o_done_0_1);
                _phase_o_done_0_1 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_16[8];
                tmem_ld_x8(&_tmem_load_16[0], taddr + 64 + 64 + (unsigned int)(lane_base_1 << 16));
                float _tmem_load_17[8];
                tmem_ld_x8(&_tmem_load_17[0], taddr + 64 + 192 + (unsigned int)(lane_base_1 << 16));
                float vals_8[8];
                #pragma unroll
                for (int k_24 = 0; k_24 < 8; k_24 += 4) {
                    uint32_t _smem_stats_w_reg_28[4];
                    __int128_t _smem_b128_13;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_13) : "r"(smem_stats_w_addr + (304 + k_24) * 4));
                    _smem_stats_w_reg_28[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_13)[0];
                    _smem_stats_w_reg_28[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_13)[1];
                    _smem_stats_w_reg_28[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_13)[2];
                    _smem_stats_w_reg_28[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_13)[3];
                    #pragma unroll
                    for (int e_24 = 0; e_24 < 4; e_24++) {
                        vals_8[k_24 + e_24] = __uint_as_float(_smem_stats_w_reg_28[e_24]);
                    }
                }
                float vals_0_3[8];
                #pragma unroll
                for (int k_25 = 0; k_25 < 8; k_25 += 4) {
                    uint32_t _smem_stats_w_reg_29[4];
                    __int128_t _smem_b128_14;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_14) : "r"(smem_stats_w_addr + (320 + k_25) * 4));
                    _smem_stats_w_reg_29[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_14)[0];
                    _smem_stats_w_reg_29[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_14)[1];
                    _smem_stats_w_reg_29[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_14)[2];
                    _smem_stats_w_reg_29[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_14)[3];
                    #pragma unroll
                    for (int e_25 = 0; e_25 < 4; e_25++) {
                        vals_0_3[k_25 + e_25] = __uint_as_float(_smem_stats_w_reg_29[e_25]);
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                #pragma unroll
                for (int j_38 = 0; j_38 < 8; j_38++) {
                    int c_4 = j_38;
                    float a0_4 = vals_8[j_38];
                    float a1_4 = vals_0_3[j_38];
                    float t0_4 = ((a0_4 > 0.0f) ? _tmem_load_16[j_38] * a0_4 : 0.0f);
                    float t1_4 = ((a1_4 > 0.0f) ? _tmem_load_17[j_38] * a1_4 : 0.0f);
                    if (c_4 < rows_valid_1) {
                        *(reinterpret_cast<__nv_bfloat16*>(partial_O + (out_base_1 + c_4 * row_stride_1 + 256)) + (0)) = __float2bfloat16_rn(t0_4 + t1_4);
                    }
                }
                float _tmem_load_18[8];
                tmem_ld_x8(&_tmem_load_18[0], taddr + 64 + 64 + 8 + (unsigned int)(lane_base_1 << 16));
                float _tmem_load_19[8];
                tmem_ld_x8(&_tmem_load_19[0], taddr + 64 + 192 + 8 + (unsigned int)(lane_base_1 << 16));
                float vals_1_2[8];
                #pragma unroll
                for (int k_26 = 0; k_26 < 8; k_26 += 4) {
                    uint32_t _smem_stats_w_reg_30[4];
                    __int128_t _smem_b128_15;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_15) : "r"(smem_stats_w_addr + (312 + k_26) * 4));
                    _smem_stats_w_reg_30[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_15)[0];
                    _smem_stats_w_reg_30[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_15)[1];
                    _smem_stats_w_reg_30[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_15)[2];
                    _smem_stats_w_reg_30[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_15)[3];
                    #pragma unroll
                    for (int e_26 = 0; e_26 < 4; e_26++) {
                        vals_1_2[k_26 + e_26] = __uint_as_float(_smem_stats_w_reg_30[e_26]);
                    }
                }
                float vals_2_1[8];
                #pragma unroll
                for (int k_27 = 0; k_27 < 8; k_27 += 4) {
                    uint32_t _smem_stats_w_reg_31[4];
                    __int128_t _smem_b128_16;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_16) : "r"(smem_stats_w_addr + (328 + k_27) * 4));
                    _smem_stats_w_reg_31[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_16)[0];
                    _smem_stats_w_reg_31[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_16)[1];
                    _smem_stats_w_reg_31[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_16)[2];
                    _smem_stats_w_reg_31[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_16)[3];
                    #pragma unroll
                    for (int e_27 = 0; e_27 < 4; e_27++) {
                        vals_2_1[k_27 + e_27] = __uint_as_float(_smem_stats_w_reg_31[e_27]);
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                #pragma unroll
                for (int j_39 = 0; j_39 < 8; j_39++) {
                    int c_5 = 8 + j_39;
                    float a0_5 = vals_1_2[j_39];
                    float a1_5 = vals_2_1[j_39];
                    float t0_5 = ((a0_5 > 0.0f) ? _tmem_load_18[j_39] * a0_5 : 0.0f);
                    float t1_5 = ((a1_5 > 0.0f) ? _tmem_load_19[j_39] * a1_5 : 0.0f);
                    if (c_5 < rows_valid_1) {
                        *(reinterpret_cast<__nv_bfloat16*>(partial_O + (out_base_1 + c_5 * row_stride_1 + 256)) + (0)) = __float2bfloat16_rn(t0_5 + t1_5);
                    }
                }
                float _tmem_load_20[8];
                tmem_ld_x8(&_tmem_load_20[0], taddr + 64 + 96 + (unsigned int)(lane_base_1 << 16));
                float _tmem_load_21[8];
                tmem_ld_x8(&_tmem_load_21[0], taddr + 64 + 224 + (unsigned int)(lane_base_1 << 16));
                float vals_3_1[8];
                #pragma unroll
                for (int k_28 = 0; k_28 < 8; k_28 += 4) {
                    uint32_t _smem_stats_w_reg_32[4];
                    __int128_t _smem_b128_17;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_17) : "r"(smem_stats_w_addr + (304 + k_28) * 4));
                    _smem_stats_w_reg_32[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_17)[0];
                    _smem_stats_w_reg_32[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_17)[1];
                    _smem_stats_w_reg_32[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_17)[2];
                    _smem_stats_w_reg_32[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_17)[3];
                    #pragma unroll
                    for (int e_28 = 0; e_28 < 4; e_28++) {
                        vals_3_1[k_28 + e_28] = __uint_as_float(_smem_stats_w_reg_32[e_28]);
                    }
                }
                float vals_4_1[8];
                #pragma unroll
                for (int k_29 = 0; k_29 < 8; k_29 += 4) {
                    uint32_t _smem_stats_w_reg_33[4];
                    __int128_t _smem_b128_18;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_18) : "r"(smem_stats_w_addr + (320 + k_29) * 4));
                    _smem_stats_w_reg_33[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_18)[0];
                    _smem_stats_w_reg_33[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_18)[1];
                    _smem_stats_w_reg_33[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_18)[2];
                    _smem_stats_w_reg_33[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_18)[3];
                    #pragma unroll
                    for (int e_29 = 0; e_29 < 4; e_29++) {
                        vals_4_1[k_29 + e_29] = __uint_as_float(_smem_stats_w_reg_33[e_29]);
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                #pragma unroll
                for (int j_40 = 0; j_40 < 8; j_40++) {
                    int c_6 = j_40;
                    float a0_6 = vals_3_1[j_40];
                    float a1_6 = vals_4_1[j_40];
                    float t0_6 = ((a0_6 > 0.0f) ? _tmem_load_20[j_40] * a0_6 : 0.0f);
                    float t1_6 = ((a1_6 > 0.0f) ? _tmem_load_21[j_40] * a1_6 : 0.0f);
                    if (c_6 < rows_valid_1) {
                        *(reinterpret_cast<__nv_bfloat16*>(partial_O + (out_base_1 + c_6 * row_stride_1 + 384)) + (0)) = __float2bfloat16_rn(t0_6 + t1_6);
                    }
                }
                float _tmem_load_22[8];
                tmem_ld_x8(&_tmem_load_22[0], taddr + 64 + 96 + 8 + (unsigned int)(lane_base_1 << 16));
                float _tmem_load_23[8];
                tmem_ld_x8(&_tmem_load_23[0], taddr + 64 + 224 + 8 + (unsigned int)(lane_base_1 << 16));
                float vals_5_1[8];
                #pragma unroll
                for (int k_30 = 0; k_30 < 8; k_30 += 4) {
                    uint32_t _smem_stats_w_reg_34[4];
                    __int128_t _smem_b128_19;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_19) : "r"(smem_stats_w_addr + (312 + k_30) * 4));
                    _smem_stats_w_reg_34[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_19)[0];
                    _smem_stats_w_reg_34[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_19)[1];
                    _smem_stats_w_reg_34[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_19)[2];
                    _smem_stats_w_reg_34[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_19)[3];
                    #pragma unroll
                    for (int e_30 = 0; e_30 < 4; e_30++) {
                        vals_5_1[k_30 + e_30] = __uint_as_float(_smem_stats_w_reg_34[e_30]);
                    }
                }
                float vals_6_1[8];
                #pragma unroll
                for (int k_31 = 0; k_31 < 8; k_31 += 4) {
                    uint32_t _smem_stats_w_reg_35[4];
                    __int128_t _smem_b128_20;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_20) : "r"(smem_stats_w_addr + (328 + k_31) * 4));
                    _smem_stats_w_reg_35[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_20)[0];
                    _smem_stats_w_reg_35[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_20)[1];
                    _smem_stats_w_reg_35[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_20)[2];
                    _smem_stats_w_reg_35[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_20)[3];
                    #pragma unroll
                    for (int e_31 = 0; e_31 < 4; e_31++) {
                        vals_6_1[k_31 + e_31] = __uint_as_float(_smem_stats_w_reg_35[e_31]);
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                #pragma unroll
                for (int j_41 = 0; j_41 < 8; j_41++) {
                    int c_7 = 8 + j_41;
                    float a0_7 = vals_5_1[j_41];
                    float a1_7 = vals_6_1[j_41];
                    float t0_7 = ((a0_7 > 0.0f) ? _tmem_load_22[j_41] * a0_7 : 0.0f);
                    float t1_7 = ((a1_7 > 0.0f) ? _tmem_load_23[j_41] * a1_7 : 0.0f);
                    if (c_7 < rows_valid_1) {
                        *(reinterpret_cast<__nv_bfloat16*>(partial_O + (out_base_1 + c_7 * row_stride_1 + 384)) + (0)) = __float2bfloat16_rn(t0_7 + t1_7);
                    }
                }
            } else if (num_split == 1) {
                #pragma unroll
                for (int j_42 = 0; j_42 < 16; j_42++) {
                    if (rows_valid_1 > j_42) {
                        *(reinterpret_cast<__nv_bfloat16*>(partial_O + (out_base_1 + j_42 * row_stride_1 + 256)) + (0)) = __float2bfloat16_rn(0.0f);
                    }
                }
                #pragma unroll
                for (int j_43 = 0; j_43 < 16; j_43++) {
                    if (rows_valid_1 > j_43) {
                        *(reinterpret_cast<__nv_bfloat16*>(partial_O + (out_base_1 + j_43 * row_stride_1 + 384)) + (0)) = __float2bfloat16_rn(0.0f);
                    }
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: transform_wg0 ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 112;");
        { // transform_wg0_main
            int split_idx_2 = blockIdx.x;
            int m_tile_2 = gridDim.y - 1 - blockIdx.y;
            int b_2 = blockIdx.z;
            int q_start_2 = cum_seq_lens_q[b_2];
            int q_len_b_2 = cum_seq_lens_q[b_2 + 1] - q_start_2;
            int kv_len_2 = seq_lens[b_2];
            int g_len_2 = kv_len_global[b_2];
            int rows_b_2 = q_len_b_2 * num_heads;
            int row0_2 = m_tile_2 * 16;
            int rows_left_2 = rows_b_2 - row0_2;
            int rows_pos_2 = ((rows_left_2 < 0) ? 0 : rows_left_2);
            int rows_valid_2 = ((rows_pos_2 > 16) ? 16 : rows_pos_2);
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
            int my_start_2 = split_idx_2 * tiles_per_split_2;
            int my_end_raw_2 = my_start_2 + tiles_per_split_2;
            int my_end_2 = ((my_end_raw_2 > n_tiles_total_2) ? n_tiles_total_2 : my_end_raw_2);
            int my_n_raw_2 = my_end_2 - my_start_2;
            int my_n_tiles_2 = ((my_n_raw_2 < 0) ? 0 : my_n_raw_2);
            int pt_base_2 = b_2 * max_pages_per_seq;
            const int warp_in_wg_2 = warp % 4;
            const int lane_base_2 = warp_in_wg_2 * 32;
            const int my_tok_2 = lane_base_2 + lane;
            const int swz = my_tok_2 & 7;
            const int ssw_half = my_tok_2 >> 2 & 1;
            unsigned int _phase_q_full_0 = 0;
            mbarrier_wait(q_full_addr, _phase_q_full_0);
            _phase_q_full_0 ^= 1;
            unsigned int qsw[8];
            #pragma unroll
            for (int j_44 = 0; j_44 < 8; j_44++) {
                qsw[j_44] = 0;
            }
            if (my_tok_2 < 16) {
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&qsw[0])), "=r"(*reinterpret_cast<uint32_t*>(&qsw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qsw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qsw[(0) + 3]))
                    : "r"(smem_qs_addr + (unsigned int)(my_tok_2 * 32)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&qsw[4])), "=r"(*reinterpret_cast<uint32_t*>(&qsw[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qsw[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qsw[(4) + 3]))
                    : "r"(smem_qs_addr + (unsigned int)(my_tok_2 * 32) + 16));
            }
            smem_sf_words[my_tok_2 % 32 / 8 * 512 + my_tok_2 % 8 * 16 + my_tok_2 / 32 % 4 * 4 >> 2] = qsw[0];
            smem_sf_words[my_tok_2 % 32 / 8 * 512 + 128 + my_tok_2 % 8 * 16 + my_tok_2 / 32 % 4 * 4 >> 2] = qsw[1];
            smem_sf_words[my_tok_2 % 32 / 8 * 512 + 256 + my_tok_2 % 8 * 16 + my_tok_2 / 32 % 4 * 4 >> 2] = qsw[2];
            smem_sf_words[my_tok_2 % 32 / 8 * 512 + 384 + my_tok_2 % 8 * 16 + my_tok_2 / 32 % 4 * 4 >> 2] = qsw[3];
            smem_sf_words[2048 + my_tok_2 % 32 / 8 * 512 + my_tok_2 % 8 * 16 + my_tok_2 / 32 % 4 * 4 >> 2] = qsw[4];
            smem_sf_words[2048 + my_tok_2 % 32 / 8 * 512 + 128 + my_tok_2 % 8 * 16 + my_tok_2 / 32 % 4 * 4 >> 2] = qsw[5];
            smem_sf_words[2048 + my_tok_2 % 32 / 8 * 512 + 256 + my_tok_2 % 8 * 16 + my_tok_2 / 32 % 4 * 4 >> 2] = qsw[6];
            smem_sf_words[2048 + my_tok_2 % 32 / 8 * 512 + 384 + my_tok_2 % 8 * 16 + my_tok_2 / 32 % 4 * 4 >> 2] = qsw[7];
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(sfb_full_addr);
            unsigned int kreg[32];
            unsigned int kregn[32];
            unsigned int sfw[8];
            unsigned int sfn[8];
            #pragma unroll
            for (int j_45 = 0; j_45 < 8; j_45++) {
                sfw[j_45] = 0;
                sfn[j_45] = 0;
            }
            if (my_n_tiles_2 > 0) {
                int st_t = 0;
                mbarrier_wait(kv_full_addr + (st_t) * 8, 0);
                int ks_row = smem_ks_addr + (unsigned int)(st_t * 45056) + (unsigned int)(my_tok_2 * 32);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sfw[0])), "=r"(*reinterpret_cast<uint32_t*>(&sfw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sfw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sfw[(0) + 3]))
                    : "r"(ks_row + (ssw_half << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sfw[4])), "=r"(*reinterpret_cast<uint32_t*>(&sfw[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sfw[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sfw[(4) + 3]))
                    : "r"(ks_row + (1 - ssw_half << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&kreg[0])), "=r"(*reinterpret_cast<uint32_t*>(&kreg[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kreg[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kreg[(0) + 3]))
                    : "r"(smem_v6_addr + (unsigned int)(st_t * 45056) + (unsigned int)(my_tok_2 * 128) + (unsigned int)((0 ^ swz) << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&kreg[4])), "=r"(*reinterpret_cast<uint32_t*>(&kreg[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kreg[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kreg[(4) + 3]))
                    : "r"(smem_v6_addr + (unsigned int)(st_t * 45056) + (unsigned int)(my_tok_2 * 128) + (unsigned int)((1 ^ swz) << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&kreg[8])), "=r"(*reinterpret_cast<uint32_t*>(&kreg[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kreg[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kreg[(8) + 3]))
                    : "r"(smem_v6_addr + (unsigned int)(st_t * 45056) + (unsigned int)(my_tok_2 * 128) + (unsigned int)((2 ^ swz) << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&kreg[12])), "=r"(*reinterpret_cast<uint32_t*>(&kreg[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kreg[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kreg[(12) + 3]))
                    : "r"(smem_v6_addr + (unsigned int)(st_t * 45056) + (unsigned int)(my_tok_2 * 128) + (unsigned int)((3 ^ swz) << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&kreg[16])), "=r"(*reinterpret_cast<uint32_t*>(&kreg[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kreg[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kreg[(16) + 3]))
                    : "r"(smem_v6_addr + (unsigned int)(st_t * 45056) + (unsigned int)(my_tok_2 * 128) + (unsigned int)((4 ^ swz) << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&kreg[20])), "=r"(*reinterpret_cast<uint32_t*>(&kreg[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kreg[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kreg[(20) + 3]))
                    : "r"(smem_v6_addr + (unsigned int)(st_t * 45056) + (unsigned int)(my_tok_2 * 128) + (unsigned int)((5 ^ swz) << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&kreg[24])), "=r"(*reinterpret_cast<uint32_t*>(&kreg[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kreg[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kreg[(24) + 3]))
                    : "r"(smem_v6_addr + (unsigned int)(st_t * 45056) + (unsigned int)(my_tok_2 * 128) + (unsigned int)((6 ^ swz) << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&kreg[28])), "=r"(*reinterpret_cast<uint32_t*>(&kreg[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kreg[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kreg[(28) + 3]))
                    : "r"(smem_v6_addr + (unsigned int)(st_t * 45056) + (unsigned int)(my_tok_2 * 128) + (unsigned int)((7 ^ swz) << 4)));
                int t_abs_t = my_start_2 * 128 + my_tok_2;
                if (t_abs_t >= kv_end_2) {
                    #pragma unroll
                    for (int j_46 = 0; j_46 < 8; j_46++) {
                        sfw[j_46] = 0;
                    }
                }
                int sb = 0;
                int st_t_0 = 0;
                int sfa_col = taddr + 320 + (unsigned int)(sb * 32) + (unsigned int)warp_in_wg_2 + (unsigned int)(lane_base_2 << 16);
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(sfa_col), "r"(sfw[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(sfa_col + 4), "r"((sfw + 1)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(sfa_col + 8), "r"((sfw + 2)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(sfa_col + 12), "r"((sfw + 3)[0]));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(kv_empty_tr_addr + (st_t_0) * 8);
                mbarrier_arrive(sfa_full_addr + (sb) * 8);
                int kb_t = 0;
                unsigned int a1_8 = 0;
                unsigned int a2 = 0;
                unsigned int a3 = 0;
                #pragma unroll
                for (int j_47 = 0; j_47 < 8; j_47++) {
                    a1_8 = a1_8 | sfw[j_47] + 370546198;
                    a2 = a2 | sfw[j_47] + 235802126;
                    a3 = a3 | sfw[j_47] + 101058054;
                }
                unsigned int k1 = (((a1_8 & 2155905152u) != 0) ? 1 : 0);
                unsigned int k2 = (((a2 & 2155905152u) != 0) ? 1 : 0);
                unsigned int k3 = (((a3 & 2155905152u) != 0) ? 1 : 0);
                unsigned int kshift = k1 + k2 + k3;
                smem_kbuf[kb_t * 128 + warp % 4 * 32 + lane] = __uint_as_float(127 + kshift << 23);
                mbarrier_arrive(k_full_addr + (kb_t) * 8);
            }
            #pragma unroll 1
            for (int tile_2 = 0; tile_2 < my_n_tiles_2; tile_2++) {
                int nxt = tile_2 + 1;
                if (nxt < my_n_tiles_2) {
                    int st_t_1 = nxt % 2;
                    mbarrier_wait(kv_full_addr + (st_t_1) * 8, nxt / 2 & 1);
                    int ks_row_1 = smem_ks_addr + (unsigned int)(st_t_1 * 45056) + (unsigned int)(my_tok_2 * 32);
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sfn[0])), "=r"(*reinterpret_cast<uint32_t*>(&sfn[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sfn[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sfn[(0) + 3]))
                        : "r"(ks_row_1 + (ssw_half << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sfn[4])), "=r"(*reinterpret_cast<uint32_t*>(&sfn[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sfn[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sfn[(4) + 3]))
                        : "r"(ks_row_1 + (1 - ssw_half << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kregn[0])), "=r"(*reinterpret_cast<uint32_t*>(&kregn[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kregn[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kregn[(0) + 3]))
                        : "r"(smem_v6_addr + (unsigned int)(st_t_1 * 45056) + (unsigned int)(my_tok_2 * 128) + (unsigned int)((0 ^ swz) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kregn[4])), "=r"(*reinterpret_cast<uint32_t*>(&kregn[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kregn[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kregn[(4) + 3]))
                        : "r"(smem_v6_addr + (unsigned int)(st_t_1 * 45056) + (unsigned int)(my_tok_2 * 128) + (unsigned int)((1 ^ swz) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kregn[8])), "=r"(*reinterpret_cast<uint32_t*>(&kregn[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kregn[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kregn[(8) + 3]))
                        : "r"(smem_v6_addr + (unsigned int)(st_t_1 * 45056) + (unsigned int)(my_tok_2 * 128) + (unsigned int)((2 ^ swz) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kregn[12])), "=r"(*reinterpret_cast<uint32_t*>(&kregn[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kregn[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kregn[(12) + 3]))
                        : "r"(smem_v6_addr + (unsigned int)(st_t_1 * 45056) + (unsigned int)(my_tok_2 * 128) + (unsigned int)((3 ^ swz) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kregn[16])), "=r"(*reinterpret_cast<uint32_t*>(&kregn[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kregn[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kregn[(16) + 3]))
                        : "r"(smem_v6_addr + (unsigned int)(st_t_1 * 45056) + (unsigned int)(my_tok_2 * 128) + (unsigned int)((4 ^ swz) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kregn[20])), "=r"(*reinterpret_cast<uint32_t*>(&kregn[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kregn[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kregn[(20) + 3]))
                        : "r"(smem_v6_addr + (unsigned int)(st_t_1 * 45056) + (unsigned int)(my_tok_2 * 128) + (unsigned int)((5 ^ swz) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kregn[24])), "=r"(*reinterpret_cast<uint32_t*>(&kregn[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kregn[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kregn[(24) + 3]))
                        : "r"(smem_v6_addr + (unsigned int)(st_t_1 * 45056) + (unsigned int)(my_tok_2 * 128) + (unsigned int)((6 ^ swz) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kregn[28])), "=r"(*reinterpret_cast<uint32_t*>(&kregn[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kregn[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kregn[(28) + 3]))
                        : "r"(smem_v6_addr + (unsigned int)(st_t_1 * 45056) + (unsigned int)(my_tok_2 * 128) + (unsigned int)((7 ^ swz) << 4)));
                    int t_abs_t_1 = (my_start_2 + nxt) * 128 + my_tok_2;
                    if (t_abs_t_1 >= kv_end_2) {
                        #pragma unroll
                        for (int j_48 = 0; j_48 < 8; j_48++) {
                            sfn[j_48] = 0;
                        }
                    }
                    int sb_1 = nxt & 1;
                    int st_t_0_1 = nxt % 2;
                    if (nxt >= 2) {
                        mbarrier_wait(sfa_empty_addr + (sb_1) * 8, (nxt >> 1) - 1 & 1);
                    }
                    int sfa_col_1 = taddr + 320 + (unsigned int)(sb_1 * 32) + (unsigned int)warp_in_wg_2 + (unsigned int)(lane_base_2 << 16);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfa_col_1), "r"(sfn[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfa_col_1 + 4), "r"((sfn + 1)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfa_col_1 + 8), "r"((sfn + 2)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfa_col_1 + 12), "r"((sfn + 3)[0]));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(kv_empty_tr_addr + (st_t_0_1) * 8);
                    mbarrier_arrive(sfa_full_addr + (sb_1) * 8);
                    int kb_t_1 = nxt & 3;
                    if (nxt >= 4) {
                        mbarrier_wait(k_empty_addr + (kb_t_1) * 8, (nxt >> 2) - 1 & 1);
                    }
                    unsigned int a1_9 = 0;
                    unsigned int a2_1 = 0;
                    unsigned int a3_1 = 0;
                    #pragma unroll
                    for (int j_49 = 0; j_49 < 8; j_49++) {
                        a1_9 = a1_9 | sfn[j_49] + 370546198;
                        a2_1 = a2_1 | sfn[j_49] + 235802126;
                        a3_1 = a3_1 | sfn[j_49] + 101058054;
                    }
                    unsigned int k1_1 = (((a1_9 & 2155905152u) != 0) ? 1 : 0);
                    unsigned int k2_1 = (((a2_1 & 2155905152u) != 0) ? 1 : 0);
                    unsigned int k3_1 = (((a3_1 & 2155905152u) != 0) ? 1 : 0);
                    unsigned int kshift_1 = k1_1 + k2_1 + k3_1;
                    smem_kbuf[kb_t_1 * 128 + warp % 4 * 32 + lane] = __uint_as_float(127 + kshift_1 << 23);
                    mbarrier_arrive(k_full_addr + (kb_t_1) * 8);
                }
                unsigned int a1_10 = 0;
                unsigned int a2_2 = 0;
                unsigned int a3_2 = 0;
                #pragma unroll
                for (int j_50 = 0; j_50 < 8; j_50++) {
                    a1_10 = a1_10 | sfw[j_50] + 370546198;
                    a2_2 = a2_2 | sfw[j_50] + 235802126;
                    a3_2 = a3_2 | sfw[j_50] + 101058054;
                }
                unsigned int k1_2 = (((a1_10 & 2155905152u) != 0) ? 1 : 0);
                unsigned int k2_2 = (((a2_2 & 2155905152u) != 0) ? 1 : 0);
                unsigned int k3_2 = (((a3_2 & 2155905152u) != 0) ? 1 : 0);
                unsigned int kshift_2 = k1_2 + k2_2 + k3_2;
                unsigned int hk = 15 - kshift_2 << 10;
                unsigned int mulk = hk | hk << 16;
                unsigned int ssw[4];
                uint32_t _e4m3x2_to_f16x2_0;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_0) : "h"((uint16_t)((uint16_t)(sfw[0] & 65535))));
                uint32_t _f16x2_mul_0;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_0) : "r"(_e4m3x2_to_f16x2_0), "r"(mulk));
                uint32_t _e4m3x2_to_f16x2_1;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_1) : "h"((uint16_t)((uint16_t)(sfw[0] >> 16))));
                uint32_t _f16x2_mul_1;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_1) : "r"(_e4m3x2_to_f16x2_1), "r"(mulk));
                uint16_t _e4m3x2_0;
                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_e4m3x2_0) : "r"(_f16x2_mul_0));
                uint16_t _e4m3x2_1;
                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_e4m3x2_1) : "r"(_f16x2_mul_1));
                ssw[0] = (unsigned int)_e4m3x2_0 | (unsigned int)_e4m3x2_1 << 16;
                uint32_t _e4m3x2_to_f16x2_2;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_2) : "h"((uint16_t)((uint16_t)(sfw[1] & 65535))));
                uint32_t _f16x2_mul_2;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_2) : "r"(_e4m3x2_to_f16x2_2), "r"(mulk));
                uint32_t _e4m3x2_to_f16x2_3;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_3) : "h"((uint16_t)((uint16_t)(sfw[1] >> 16))));
                uint32_t _f16x2_mul_3;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_3) : "r"(_e4m3x2_to_f16x2_3), "r"(mulk));
                uint16_t _e4m3x2_2;
                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_e4m3x2_2) : "r"(_f16x2_mul_2));
                uint16_t _e4m3x2_3;
                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_e4m3x2_3) : "r"(_f16x2_mul_3));
                ssw[1] = (unsigned int)_e4m3x2_2 | (unsigned int)_e4m3x2_3 << 16;
                uint32_t _e4m3x2_to_f16x2_4;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_4) : "h"((uint16_t)((uint16_t)(sfw[2] & 65535))));
                uint32_t _f16x2_mul_4;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_4) : "r"(_e4m3x2_to_f16x2_4), "r"(mulk));
                uint32_t _e4m3x2_to_f16x2_5;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_5) : "h"((uint16_t)((uint16_t)(sfw[2] >> 16))));
                uint32_t _f16x2_mul_5;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_5) : "r"(_e4m3x2_to_f16x2_5), "r"(mulk));
                uint16_t _e4m3x2_4;
                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_e4m3x2_4) : "r"(_f16x2_mul_4));
                uint16_t _e4m3x2_5;
                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_e4m3x2_5) : "r"(_f16x2_mul_5));
                ssw[2] = (unsigned int)_e4m3x2_4 | (unsigned int)_e4m3x2_5 << 16;
                uint32_t _e4m3x2_to_f16x2_6;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_6) : "h"((uint16_t)((uint16_t)(sfw[3] & 65535))));
                uint32_t _f16x2_mul_6;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_6) : "r"(_e4m3x2_to_f16x2_6), "r"(mulk));
                uint32_t _e4m3x2_to_f16x2_7;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_7) : "h"((uint16_t)((uint16_t)(sfw[3] >> 16))));
                uint32_t _f16x2_mul_7;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_7) : "r"(_e4m3x2_to_f16x2_7), "r"(mulk));
                uint16_t _e4m3x2_6;
                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_e4m3x2_6) : "r"(_f16x2_mul_6));
                uint16_t _e4m3x2_7;
                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_e4m3x2_7) : "r"(_f16x2_mul_7));
                ssw[3] = (unsigned int)_e4m3x2_6 | (unsigned int)_e4m3x2_7 << 16;
                if (tile_2 >= 1) {
                    mbarrier_wait(v_empty_addr, tile_2 - 1 & 1);
                }
                unsigned int v8[8];
                uint32_t _prmt_b32_0;
                asm("prmt.b32 %0, %1, %2, 0x0000;" : "=r"(_prmt_b32_0) : "r"(ssw[0]), "r"(ssw[0]));
                unsigned int sc0 = _prmt_b32_0;
                uint32_t _prmt_b32_1;
                asm("prmt.b32 %0, %1, %2, 0x1111;" : "=r"(_prmt_b32_1) : "r"(ssw[0]), "r"(ssw[0]));
                unsigned int sc1 = _prmt_b32_1;
                {
                    v8[0] = cake_mla_nvfp4_qmul4<5>(kreg[0], sc0);
                }
                {
                    v8[1] = cake_mla_nvfp4_qmul4<6>(kreg[0], sc0);
                }
                {
                    v8[2] = cake_mla_nvfp4_qmul4<5>(kreg[1], sc0);
                }
                {
                    v8[3] = cake_mla_nvfp4_qmul4<6>(kreg[1], sc0);
                }
                {
                    v8[4] = cake_mla_nvfp4_qmul4<5>(kreg[2], sc1);
                }
                {
                    v8[5] = cake_mla_nvfp4_qmul4<6>(kreg[2], sc1);
                }
                {
                    v8[6] = cake_mla_nvfp4_qmul4<5>(kreg[3], sc1);
                }
                {
                    v8[7] = cake_mla_nvfp4_qmul4<6>(kreg[3], sc1);
                }
                int vrow = smem_v_addr + (unsigned int)(my_tok_2 * 128);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(vrow + ((0 ^ swz) << 4)), "r"(*reinterpret_cast<uint32_t*>(&v8[0])), "r"(*reinterpret_cast<uint32_t*>(&v8[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(vrow + ((1 ^ swz) << 4)), "r"(*reinterpret_cast<uint32_t*>(&v8[4])), "r"(*reinterpret_cast<uint32_t*>(&v8[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8[(4) + 3])));
                unsigned int v8_0[8];
                uint32_t _prmt_b32_2;
                asm("prmt.b32 %0, %1, %2, 0x2222;" : "=r"(_prmt_b32_2) : "r"(ssw[0]), "r"(ssw[0]));
                unsigned int sc0_1 = _prmt_b32_2;
                uint32_t _prmt_b32_3;
                asm("prmt.b32 %0, %1, %2, 0x3333;" : "=r"(_prmt_b32_3) : "r"(ssw[0]), "r"(ssw[0]));
                unsigned int sc1_2 = _prmt_b32_3;
                {
                    v8_0[0] = cake_mla_nvfp4_qmul4<5>(kreg[4], sc0_1);
                }
                {
                    v8_0[1] = cake_mla_nvfp4_qmul4<6>(kreg[4], sc0_1);
                }
                {
                    v8_0[2] = cake_mla_nvfp4_qmul4<5>(kreg[5], sc0_1);
                }
                {
                    v8_0[3] = cake_mla_nvfp4_qmul4<6>(kreg[5], sc0_1);
                }
                {
                    v8_0[4] = cake_mla_nvfp4_qmul4<5>(kreg[6], sc1_2);
                }
                {
                    v8_0[5] = cake_mla_nvfp4_qmul4<6>(kreg[6], sc1_2);
                }
                {
                    v8_0[6] = cake_mla_nvfp4_qmul4<5>(kreg[7], sc1_2);
                }
                {
                    v8_0[7] = cake_mla_nvfp4_qmul4<6>(kreg[7], sc1_2);
                }
                int vrow_3 = smem_v_addr + (unsigned int)(my_tok_2 * 128);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(vrow_3 + ((2 ^ swz) << 4)), "r"(*reinterpret_cast<uint32_t*>(&v8_0[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_0[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_0[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_0[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(vrow_3 + ((3 ^ swz) << 4)), "r"(*reinterpret_cast<uint32_t*>(&v8_0[4])), "r"(*reinterpret_cast<uint32_t*>(&v8_0[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_0[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_0[(4) + 3])));
                unsigned int v8_4[8];
                uint32_t _prmt_b32_4;
                asm("prmt.b32 %0, %1, %2, 0x0000;" : "=r"(_prmt_b32_4) : "r"(ssw[1]), "r"(ssw[1]));
                unsigned int sc0_5 = _prmt_b32_4;
                uint32_t _prmt_b32_5;
                asm("prmt.b32 %0, %1, %2, 0x1111;" : "=r"(_prmt_b32_5) : "r"(ssw[1]), "r"(ssw[1]));
                unsigned int sc1_6 = _prmt_b32_5;
                {
                    v8_4[0] = cake_mla_nvfp4_qmul4<5>(kreg[8], sc0_5);
                }
                {
                    v8_4[1] = cake_mla_nvfp4_qmul4<6>(kreg[8], sc0_5);
                }
                {
                    v8_4[2] = cake_mla_nvfp4_qmul4<5>(kreg[9], sc0_5);
                }
                {
                    v8_4[3] = cake_mla_nvfp4_qmul4<6>(kreg[9], sc0_5);
                }
                {
                    v8_4[4] = cake_mla_nvfp4_qmul4<5>(kreg[10], sc1_6);
                }
                {
                    v8_4[5] = cake_mla_nvfp4_qmul4<6>(kreg[10], sc1_6);
                }
                {
                    v8_4[6] = cake_mla_nvfp4_qmul4<5>(kreg[11], sc1_6);
                }
                {
                    v8_4[7] = cake_mla_nvfp4_qmul4<6>(kreg[11], sc1_6);
                }
                int vrow_7 = smem_v_addr + (unsigned int)(my_tok_2 * 128);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(vrow_7 + ((4 ^ swz) << 4)), "r"(*reinterpret_cast<uint32_t*>(&v8_4[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_4[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_4[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_4[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(vrow_7 + ((5 ^ swz) << 4)), "r"(*reinterpret_cast<uint32_t*>(&v8_4[4])), "r"(*reinterpret_cast<uint32_t*>(&v8_4[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_4[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_4[(4) + 3])));
                unsigned int v8_8[8];
                uint32_t _prmt_b32_6;
                asm("prmt.b32 %0, %1, %2, 0x2222;" : "=r"(_prmt_b32_6) : "r"(ssw[1]), "r"(ssw[1]));
                unsigned int sc0_9 = _prmt_b32_6;
                uint32_t _prmt_b32_7;
                asm("prmt.b32 %0, %1, %2, 0x3333;" : "=r"(_prmt_b32_7) : "r"(ssw[1]), "r"(ssw[1]));
                unsigned int sc1_10 = _prmt_b32_7;
                {
                    v8_8[0] = cake_mla_nvfp4_qmul4<5>(kreg[12], sc0_9);
                }
                {
                    v8_8[1] = cake_mla_nvfp4_qmul4<6>(kreg[12], sc0_9);
                }
                {
                    v8_8[2] = cake_mla_nvfp4_qmul4<5>(kreg[13], sc0_9);
                }
                {
                    v8_8[3] = cake_mla_nvfp4_qmul4<6>(kreg[13], sc0_9);
                }
                {
                    v8_8[4] = cake_mla_nvfp4_qmul4<5>(kreg[14], sc1_10);
                }
                {
                    v8_8[5] = cake_mla_nvfp4_qmul4<6>(kreg[14], sc1_10);
                }
                {
                    v8_8[6] = cake_mla_nvfp4_qmul4<5>(kreg[15], sc1_10);
                }
                {
                    v8_8[7] = cake_mla_nvfp4_qmul4<6>(kreg[15], sc1_10);
                }
                int vrow_11 = smem_v_addr + (unsigned int)(my_tok_2 * 128);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(vrow_11 + ((6 ^ swz) << 4)), "r"(*reinterpret_cast<uint32_t*>(&v8_8[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_8[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_8[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_8[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(vrow_11 + ((7 ^ swz) << 4)), "r"(*reinterpret_cast<uint32_t*>(&v8_8[4])), "r"(*reinterpret_cast<uint32_t*>(&v8_8[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_8[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_8[(4) + 3])));
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(v_full_addr);
                if (tile_2 >= 1) {
                    mbarrier_wait(v_empty_addr + 8, tile_2 - 1 & 1);
                }
                unsigned int v8_12[8];
                uint32_t _prmt_b32_8;
                asm("prmt.b32 %0, %1, %2, 0x0000;" : "=r"(_prmt_b32_8) : "r"(ssw[2]), "r"(ssw[2]));
                unsigned int sc0_13 = _prmt_b32_8;
                uint32_t _prmt_b32_9;
                asm("prmt.b32 %0, %1, %2, 0x1111;" : "=r"(_prmt_b32_9) : "r"(ssw[2]), "r"(ssw[2]));
                unsigned int sc1_14 = _prmt_b32_9;
                {
                    v8_12[0] = cake_mla_nvfp4_qmul4<5>(kreg[16], sc0_13);
                }
                {
                    v8_12[1] = cake_mla_nvfp4_qmul4<6>(kreg[16], sc0_13);
                }
                {
                    v8_12[2] = cake_mla_nvfp4_qmul4<5>(kreg[17], sc0_13);
                }
                {
                    v8_12[3] = cake_mla_nvfp4_qmul4<6>(kreg[17], sc0_13);
                }
                {
                    v8_12[4] = cake_mla_nvfp4_qmul4<5>(kreg[18], sc1_14);
                }
                {
                    v8_12[5] = cake_mla_nvfp4_qmul4<6>(kreg[18], sc1_14);
                }
                {
                    v8_12[6] = cake_mla_nvfp4_qmul4<5>(kreg[19], sc1_14);
                }
                {
                    v8_12[7] = cake_mla_nvfp4_qmul4<6>(kreg[19], sc1_14);
                }
                int vrow_15 = smem_v_addr + 16384 + (unsigned int)(my_tok_2 * 128);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(vrow_15 + ((0 ^ swz) << 4)), "r"(*reinterpret_cast<uint32_t*>(&v8_12[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_12[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_12[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_12[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(vrow_15 + ((1 ^ swz) << 4)), "r"(*reinterpret_cast<uint32_t*>(&v8_12[4])), "r"(*reinterpret_cast<uint32_t*>(&v8_12[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_12[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_12[(4) + 3])));
                unsigned int v8_16[8];
                uint32_t _prmt_b32_10;
                asm("prmt.b32 %0, %1, %2, 0x2222;" : "=r"(_prmt_b32_10) : "r"(ssw[2]), "r"(ssw[2]));
                unsigned int sc0_17 = _prmt_b32_10;
                uint32_t _prmt_b32_11;
                asm("prmt.b32 %0, %1, %2, 0x3333;" : "=r"(_prmt_b32_11) : "r"(ssw[2]), "r"(ssw[2]));
                unsigned int sc1_18 = _prmt_b32_11;
                {
                    v8_16[0] = cake_mla_nvfp4_qmul4<5>(kreg[20], sc0_17);
                }
                {
                    v8_16[1] = cake_mla_nvfp4_qmul4<6>(kreg[20], sc0_17);
                }
                {
                    v8_16[2] = cake_mla_nvfp4_qmul4<5>(kreg[21], sc0_17);
                }
                {
                    v8_16[3] = cake_mla_nvfp4_qmul4<6>(kreg[21], sc0_17);
                }
                {
                    v8_16[4] = cake_mla_nvfp4_qmul4<5>(kreg[22], sc1_18);
                }
                {
                    v8_16[5] = cake_mla_nvfp4_qmul4<6>(kreg[22], sc1_18);
                }
                {
                    v8_16[6] = cake_mla_nvfp4_qmul4<5>(kreg[23], sc1_18);
                }
                {
                    v8_16[7] = cake_mla_nvfp4_qmul4<6>(kreg[23], sc1_18);
                }
                int vrow_19 = smem_v_addr + 16384 + (unsigned int)(my_tok_2 * 128);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(vrow_19 + ((2 ^ swz) << 4)), "r"(*reinterpret_cast<uint32_t*>(&v8_16[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_16[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_16[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_16[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(vrow_19 + ((3 ^ swz) << 4)), "r"(*reinterpret_cast<uint32_t*>(&v8_16[4])), "r"(*reinterpret_cast<uint32_t*>(&v8_16[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_16[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_16[(4) + 3])));
                unsigned int v8_20[8];
                uint32_t _prmt_b32_12;
                asm("prmt.b32 %0, %1, %2, 0x0000;" : "=r"(_prmt_b32_12) : "r"(ssw[3]), "r"(ssw[3]));
                unsigned int sc0_21 = _prmt_b32_12;
                uint32_t _prmt_b32_13;
                asm("prmt.b32 %0, %1, %2, 0x1111;" : "=r"(_prmt_b32_13) : "r"(ssw[3]), "r"(ssw[3]));
                unsigned int sc1_22 = _prmt_b32_13;
                {
                    v8_20[0] = cake_mla_nvfp4_qmul4<5>(kreg[24], sc0_21);
                }
                {
                    v8_20[1] = cake_mla_nvfp4_qmul4<6>(kreg[24], sc0_21);
                }
                {
                    v8_20[2] = cake_mla_nvfp4_qmul4<5>(kreg[25], sc0_21);
                }
                {
                    v8_20[3] = cake_mla_nvfp4_qmul4<6>(kreg[25], sc0_21);
                }
                {
                    v8_20[4] = cake_mla_nvfp4_qmul4<5>(kreg[26], sc1_22);
                }
                {
                    v8_20[5] = cake_mla_nvfp4_qmul4<6>(kreg[26], sc1_22);
                }
                {
                    v8_20[6] = cake_mla_nvfp4_qmul4<5>(kreg[27], sc1_22);
                }
                {
                    v8_20[7] = cake_mla_nvfp4_qmul4<6>(kreg[27], sc1_22);
                }
                int vrow_23 = smem_v_addr + 16384 + (unsigned int)(my_tok_2 * 128);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(vrow_23 + ((4 ^ swz) << 4)), "r"(*reinterpret_cast<uint32_t*>(&v8_20[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_20[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_20[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_20[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(vrow_23 + ((5 ^ swz) << 4)), "r"(*reinterpret_cast<uint32_t*>(&v8_20[4])), "r"(*reinterpret_cast<uint32_t*>(&v8_20[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_20[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_20[(4) + 3])));
                unsigned int v8_24[8];
                uint32_t _prmt_b32_14;
                asm("prmt.b32 %0, %1, %2, 0x2222;" : "=r"(_prmt_b32_14) : "r"(ssw[3]), "r"(ssw[3]));
                unsigned int sc0_25 = _prmt_b32_14;
                uint32_t _prmt_b32_15;
                asm("prmt.b32 %0, %1, %2, 0x3333;" : "=r"(_prmt_b32_15) : "r"(ssw[3]), "r"(ssw[3]));
                unsigned int sc1_26 = _prmt_b32_15;
                {
                    v8_24[0] = cake_mla_nvfp4_qmul4<5>(kreg[28], sc0_25);
                }
                {
                    v8_24[1] = cake_mla_nvfp4_qmul4<6>(kreg[28], sc0_25);
                }
                {
                    v8_24[2] = cake_mla_nvfp4_qmul4<5>(kreg[29], sc0_25);
                }
                {
                    v8_24[3] = cake_mla_nvfp4_qmul4<6>(kreg[29], sc0_25);
                }
                {
                    v8_24[4] = cake_mla_nvfp4_qmul4<5>(kreg[30], sc1_26);
                }
                {
                    v8_24[5] = cake_mla_nvfp4_qmul4<6>(kreg[30], sc1_26);
                }
                {
                    v8_24[6] = cake_mla_nvfp4_qmul4<5>(kreg[31], sc1_26);
                }
                {
                    v8_24[7] = cake_mla_nvfp4_qmul4<6>(kreg[31], sc1_26);
                }
                int vrow_27 = smem_v_addr + 16384 + (unsigned int)(my_tok_2 * 128);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(vrow_27 + ((6 ^ swz) << 4)), "r"(*reinterpret_cast<uint32_t*>(&v8_24[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_24[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_24[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_24[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(vrow_27 + ((7 ^ swz) << 4)), "r"(*reinterpret_cast<uint32_t*>(&v8_24[4])), "r"(*reinterpret_cast<uint32_t*>(&v8_24[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_24[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_24[(4) + 3])));
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(v_full_addr + 8);
                #pragma unroll
                for (int j_51 = 0; j_51 < 8; j_51++) {
                    sfw[j_51] = sfn[j_51];
                }
                #pragma unroll
                for (int j_52 = 0; j_52 < 32; j_52++) {
                    kreg[j_52] = kregn[j_52];
                }
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: transform_wg1 ----
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 112;");
        { // transform_wg1_main
            int split_idx_3 = blockIdx.x;
            int m_tile_3 = gridDim.y - 1 - blockIdx.y;
            int b_3 = blockIdx.z;
            int q_start_3 = cum_seq_lens_q[b_3];
            int q_len_b_3 = cum_seq_lens_q[b_3 + 1] - q_start_3;
            int kv_len_3 = seq_lens[b_3];
            int g_len_3 = kv_len_global[b_3];
            int rows_b_3 = q_len_b_3 * num_heads;
            int row0_3 = m_tile_3 * 16;
            int rows_left_3 = rows_b_3 - row0_3;
            int rows_pos_3 = ((rows_left_3 < 0) ? 0 : rows_left_3);
            int rows_valid_3 = ((rows_pos_3 > 16) ? 16 : rows_pos_3);
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
            int my_start_3 = split_idx_3 * tiles_per_split_3;
            int my_end_raw_3 = my_start_3 + tiles_per_split_3;
            int my_end_3 = ((my_end_raw_3 > n_tiles_total_3) ? n_tiles_total_3 : my_end_raw_3);
            int my_n_raw_3 = my_end_3 - my_start_3;
            int my_n_tiles_3 = ((my_n_raw_3 < 0) ? 0 : my_n_raw_3);
            int pt_base_3 = b_3 * max_pages_per_seq;
            const int warp_in_wg_3 = warp % 4;
            const int lane_base_3 = warp_in_wg_3 * 32;
            const int my_tok_3 = lane_base_3 + lane;
            const int swz_1 = my_tok_3 & 7;
            const int ssw_half_1 = my_tok_3 >> 2 & 1;
            unsigned int kreg_1[32];
            unsigned int kregn_1[32];
            unsigned int sfw_1[8];
            unsigned int sfn_1[8];
            #pragma unroll
            for (int j_53 = 0; j_53 < 8; j_53++) {
                sfw_1[j_53] = 0;
                sfn_1[j_53] = 0;
            }
            if (my_n_tiles_3 > 0) {
                int st_t_2 = 0;
                mbarrier_wait(kv_full_addr + (st_t_2) * 8, 0);
                int ks_row_2 = smem_ks_addr + (unsigned int)(st_t_2 * 45056) + (unsigned int)(my_tok_3 * 32);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sfw_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&sfw_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sfw_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sfw_1[(0) + 3]))
                    : "r"(ks_row_2 + (ssw_half_1 << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sfw_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&sfw_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sfw_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sfw_1[(4) + 3]))
                    : "r"(ks_row_2 + (1 - ssw_half_1 << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&kreg_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&kreg_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kreg_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kreg_1[(0) + 3]))
                    : "r"(smem_v7_addr + (unsigned int)(st_t_2 * 45056) + (unsigned int)(my_tok_3 * 128) + (unsigned int)((0 ^ swz_1) << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&kreg_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&kreg_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kreg_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kreg_1[(4) + 3]))
                    : "r"(smem_v7_addr + (unsigned int)(st_t_2 * 45056) + (unsigned int)(my_tok_3 * 128) + (unsigned int)((1 ^ swz_1) << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&kreg_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&kreg_1[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kreg_1[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kreg_1[(8) + 3]))
                    : "r"(smem_v7_addr + (unsigned int)(st_t_2 * 45056) + (unsigned int)(my_tok_3 * 128) + (unsigned int)((2 ^ swz_1) << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&kreg_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&kreg_1[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kreg_1[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kreg_1[(12) + 3]))
                    : "r"(smem_v7_addr + (unsigned int)(st_t_2 * 45056) + (unsigned int)(my_tok_3 * 128) + (unsigned int)((3 ^ swz_1) << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&kreg_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&kreg_1[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kreg_1[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kreg_1[(16) + 3]))
                    : "r"(smem_v7_addr + (unsigned int)(st_t_2 * 45056) + (unsigned int)(my_tok_3 * 128) + (unsigned int)((4 ^ swz_1) << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&kreg_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&kreg_1[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kreg_1[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kreg_1[(20) + 3]))
                    : "r"(smem_v7_addr + (unsigned int)(st_t_2 * 45056) + (unsigned int)(my_tok_3 * 128) + (unsigned int)((5 ^ swz_1) << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&kreg_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&kreg_1[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kreg_1[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kreg_1[(24) + 3]))
                    : "r"(smem_v7_addr + (unsigned int)(st_t_2 * 45056) + (unsigned int)(my_tok_3 * 128) + (unsigned int)((6 ^ swz_1) << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&kreg_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&kreg_1[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kreg_1[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kreg_1[(28) + 3]))
                    : "r"(smem_v7_addr + (unsigned int)(st_t_2 * 45056) + (unsigned int)(my_tok_3 * 128) + (unsigned int)((7 ^ swz_1) << 4)));
                int t_abs_t_2 = my_start_3 * 128 + my_tok_3;
                if (t_abs_t_2 >= kv_end_3) {
                    #pragma unroll
                    for (int j_54 = 0; j_54 < 8; j_54++) {
                        sfw_1[j_54] = 0;
                    }
                }
                int sb_2 = 0;
                int st_t_0_2 = 0;
                int sfa_col_2 = taddr + 320 + (unsigned int)(sb_2 * 32) + (unsigned int)warp_in_wg_3 + (unsigned int)(lane_base_3 << 16);
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(sfa_col_2 + 16), "r"((sfw_1 + 4)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(sfa_col_2 + 20), "r"((sfw_1 + 5)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(sfa_col_2 + 24), "r"((sfw_1 + 6)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(sfa_col_2 + 28), "r"((sfw_1 + 7)[0]));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(kv_empty_tr_addr + (st_t_0_2) * 8);
                mbarrier_arrive(sfa_full_addr + (sb_2) * 8);
            }
            #pragma unroll 1
            for (int tile_3 = 0; tile_3 < my_n_tiles_3; tile_3++) {
                int nxt_1 = tile_3 + 1;
                if (nxt_1 < my_n_tiles_3) {
                    int st_t_3 = nxt_1 % 2;
                    mbarrier_wait(kv_full_addr + (st_t_3) * 8, nxt_1 / 2 & 1);
                    int ks_row_3 = smem_ks_addr + (unsigned int)(st_t_3 * 45056) + (unsigned int)(my_tok_3 * 32);
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sfn_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&sfn_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sfn_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sfn_1[(0) + 3]))
                        : "r"(ks_row_3 + (ssw_half_1 << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sfn_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&sfn_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sfn_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sfn_1[(4) + 3]))
                        : "r"(ks_row_3 + (1 - ssw_half_1 << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kregn_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&kregn_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kregn_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kregn_1[(0) + 3]))
                        : "r"(smem_v7_addr + (unsigned int)(st_t_3 * 45056) + (unsigned int)(my_tok_3 * 128) + (unsigned int)((0 ^ swz_1) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kregn_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&kregn_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kregn_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kregn_1[(4) + 3]))
                        : "r"(smem_v7_addr + (unsigned int)(st_t_3 * 45056) + (unsigned int)(my_tok_3 * 128) + (unsigned int)((1 ^ swz_1) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kregn_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&kregn_1[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kregn_1[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kregn_1[(8) + 3]))
                        : "r"(smem_v7_addr + (unsigned int)(st_t_3 * 45056) + (unsigned int)(my_tok_3 * 128) + (unsigned int)((2 ^ swz_1) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kregn_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&kregn_1[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kregn_1[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kregn_1[(12) + 3]))
                        : "r"(smem_v7_addr + (unsigned int)(st_t_3 * 45056) + (unsigned int)(my_tok_3 * 128) + (unsigned int)((3 ^ swz_1) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kregn_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&kregn_1[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kregn_1[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kregn_1[(16) + 3]))
                        : "r"(smem_v7_addr + (unsigned int)(st_t_3 * 45056) + (unsigned int)(my_tok_3 * 128) + (unsigned int)((4 ^ swz_1) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kregn_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&kregn_1[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kregn_1[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kregn_1[(20) + 3]))
                        : "r"(smem_v7_addr + (unsigned int)(st_t_3 * 45056) + (unsigned int)(my_tok_3 * 128) + (unsigned int)((5 ^ swz_1) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kregn_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&kregn_1[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kregn_1[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kregn_1[(24) + 3]))
                        : "r"(smem_v7_addr + (unsigned int)(st_t_3 * 45056) + (unsigned int)(my_tok_3 * 128) + (unsigned int)((6 ^ swz_1) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kregn_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&kregn_1[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kregn_1[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kregn_1[(28) + 3]))
                        : "r"(smem_v7_addr + (unsigned int)(st_t_3 * 45056) + (unsigned int)(my_tok_3 * 128) + (unsigned int)((7 ^ swz_1) << 4)));
                    int t_abs_t_3 = (my_start_3 + nxt_1) * 128 + my_tok_3;
                    if (t_abs_t_3 >= kv_end_3) {
                        #pragma unroll
                        for (int j_55 = 0; j_55 < 8; j_55++) {
                            sfn_1[j_55] = 0;
                        }
                    }
                    int sb_3 = nxt_1 & 1;
                    int st_t_0_3 = nxt_1 % 2;
                    if (nxt_1 >= 2) {
                        mbarrier_wait(sfa_empty_addr + (sb_3) * 8, (nxt_1 >> 1) - 1 & 1);
                    }
                    int sfa_col_3 = taddr + 320 + (unsigned int)(sb_3 * 32) + (unsigned int)warp_in_wg_3 + (unsigned int)(lane_base_3 << 16);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfa_col_3 + 16), "r"((sfn_1 + 4)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfa_col_3 + 20), "r"((sfn_1 + 5)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfa_col_3 + 24), "r"((sfn_1 + 6)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfa_col_3 + 28), "r"((sfn_1 + 7)[0]));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(kv_empty_tr_addr + (st_t_0_3) * 8);
                    mbarrier_arrive(sfa_full_addr + (sb_3) * 8);
                }
                unsigned int a1_11 = 0;
                unsigned int a2_3 = 0;
                unsigned int a3_3 = 0;
                #pragma unroll
                for (int j_56 = 0; j_56 < 8; j_56++) {
                    a1_11 = a1_11 | sfw_1[j_56] + 370546198;
                    a2_3 = a2_3 | sfw_1[j_56] + 235802126;
                    a3_3 = a3_3 | sfw_1[j_56] + 101058054;
                }
                unsigned int k1_3 = (((a1_11 & 2155905152u) != 0) ? 1 : 0);
                unsigned int k2_3 = (((a2_3 & 2155905152u) != 0) ? 1 : 0);
                unsigned int k3_3 = (((a3_3 & 2155905152u) != 0) ? 1 : 0);
                unsigned int kshift_3 = k1_3 + k2_3 + k3_3;
                unsigned int hk_1 = 15 - kshift_3 << 10;
                unsigned int mulk_1 = hk_1 | hk_1 << 16;
                unsigned int ssw_1[4];
                uint32_t _e4m3x2_to_f16x2_8;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_8) : "h"((uint16_t)((uint16_t)(sfw_1[4] & 65535))));
                uint32_t _f16x2_mul_8;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_8) : "r"(_e4m3x2_to_f16x2_8), "r"(mulk_1));
                uint32_t _e4m3x2_to_f16x2_9;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_9) : "h"((uint16_t)((uint16_t)(sfw_1[4] >> 16))));
                uint32_t _f16x2_mul_9;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_9) : "r"(_e4m3x2_to_f16x2_9), "r"(mulk_1));
                uint16_t _e4m3x2_8;
                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_e4m3x2_8) : "r"(_f16x2_mul_8));
                uint16_t _e4m3x2_9;
                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_e4m3x2_9) : "r"(_f16x2_mul_9));
                ssw_1[0] = (unsigned int)_e4m3x2_8 | (unsigned int)_e4m3x2_9 << 16;
                uint32_t _e4m3x2_to_f16x2_10;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_10) : "h"((uint16_t)((uint16_t)(sfw_1[5] & 65535))));
                uint32_t _f16x2_mul_10;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_10) : "r"(_e4m3x2_to_f16x2_10), "r"(mulk_1));
                uint32_t _e4m3x2_to_f16x2_11;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_11) : "h"((uint16_t)((uint16_t)(sfw_1[5] >> 16))));
                uint32_t _f16x2_mul_11;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_11) : "r"(_e4m3x2_to_f16x2_11), "r"(mulk_1));
                uint16_t _e4m3x2_10;
                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_e4m3x2_10) : "r"(_f16x2_mul_10));
                uint16_t _e4m3x2_11;
                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_e4m3x2_11) : "r"(_f16x2_mul_11));
                ssw_1[1] = (unsigned int)_e4m3x2_10 | (unsigned int)_e4m3x2_11 << 16;
                uint32_t _e4m3x2_to_f16x2_12;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_12) : "h"((uint16_t)((uint16_t)(sfw_1[6] & 65535))));
                uint32_t _f16x2_mul_12;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_12) : "r"(_e4m3x2_to_f16x2_12), "r"(mulk_1));
                uint32_t _e4m3x2_to_f16x2_13;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_13) : "h"((uint16_t)((uint16_t)(sfw_1[6] >> 16))));
                uint32_t _f16x2_mul_13;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_13) : "r"(_e4m3x2_to_f16x2_13), "r"(mulk_1));
                uint16_t _e4m3x2_12;
                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_e4m3x2_12) : "r"(_f16x2_mul_12));
                uint16_t _e4m3x2_13;
                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_e4m3x2_13) : "r"(_f16x2_mul_13));
                ssw_1[2] = (unsigned int)_e4m3x2_12 | (unsigned int)_e4m3x2_13 << 16;
                uint32_t _e4m3x2_to_f16x2_14;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_14) : "h"((uint16_t)((uint16_t)(sfw_1[7] & 65535))));
                uint32_t _f16x2_mul_14;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_14) : "r"(_e4m3x2_to_f16x2_14), "r"(mulk_1));
                uint32_t _e4m3x2_to_f16x2_15;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_15) : "h"((uint16_t)((uint16_t)(sfw_1[7] >> 16))));
                uint32_t _f16x2_mul_15;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_15) : "r"(_e4m3x2_to_f16x2_15), "r"(mulk_1));
                uint16_t _e4m3x2_14;
                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_e4m3x2_14) : "r"(_f16x2_mul_14));
                uint16_t _e4m3x2_15;
                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_e4m3x2_15) : "r"(_f16x2_mul_15));
                ssw_1[3] = (unsigned int)_e4m3x2_14 | (unsigned int)_e4m3x2_15 << 16;
                if (tile_3 >= 1) {
                    mbarrier_wait(v_empty_addr + 16, tile_3 - 1 & 1);
                }
                unsigned int v8_1[8];
                uint32_t _prmt_b32_16;
                asm("prmt.b32 %0, %1, %2, 0x0000;" : "=r"(_prmt_b32_16) : "r"(ssw_1[0]), "r"(ssw_1[0]));
                unsigned int sc0_2 = _prmt_b32_16;
                uint32_t _prmt_b32_17;
                asm("prmt.b32 %0, %1, %2, 0x1111;" : "=r"(_prmt_b32_17) : "r"(ssw_1[0]), "r"(ssw_1[0]));
                unsigned int sc1_1 = _prmt_b32_17;
                {
                    v8_1[0] = cake_mla_nvfp4_qmul4<5>(kreg_1[0], sc0_2);
                }
                {
                    v8_1[1] = cake_mla_nvfp4_qmul4<6>(kreg_1[0], sc0_2);
                }
                {
                    v8_1[2] = cake_mla_nvfp4_qmul4<5>(kreg_1[1], sc0_2);
                }
                {
                    v8_1[3] = cake_mla_nvfp4_qmul4<6>(kreg_1[1], sc0_2);
                }
                {
                    v8_1[4] = cake_mla_nvfp4_qmul4<5>(kreg_1[2], sc1_1);
                }
                {
                    v8_1[5] = cake_mla_nvfp4_qmul4<6>(kreg_1[2], sc1_1);
                }
                {
                    v8_1[6] = cake_mla_nvfp4_qmul4<5>(kreg_1[3], sc1_1);
                }
                {
                    v8_1[7] = cake_mla_nvfp4_qmul4<6>(kreg_1[3], sc1_1);
                }
                int vrow_1 = smem_v_addr + 32768 + (unsigned int)(my_tok_3 * 128);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(vrow_1 + ((0 ^ swz_1) << 4)), "r"(*reinterpret_cast<uint32_t*>(&v8_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_1[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(vrow_1 + ((1 ^ swz_1) << 4)), "r"(*reinterpret_cast<uint32_t*>(&v8_1[4])), "r"(*reinterpret_cast<uint32_t*>(&v8_1[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_1[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_1[(4) + 3])));
                unsigned int v8_0_1[8];
                uint32_t _prmt_b32_18;
                asm("prmt.b32 %0, %1, %2, 0x2222;" : "=r"(_prmt_b32_18) : "r"(ssw_1[0]), "r"(ssw_1[0]));
                unsigned int sc0_1_1 = _prmt_b32_18;
                uint32_t _prmt_b32_19;
                asm("prmt.b32 %0, %1, %2, 0x3333;" : "=r"(_prmt_b32_19) : "r"(ssw_1[0]), "r"(ssw_1[0]));
                unsigned int sc1_2_1 = _prmt_b32_19;
                {
                    v8_0_1[0] = cake_mla_nvfp4_qmul4<5>(kreg_1[4], sc0_1_1);
                }
                {
                    v8_0_1[1] = cake_mla_nvfp4_qmul4<6>(kreg_1[4], sc0_1_1);
                }
                {
                    v8_0_1[2] = cake_mla_nvfp4_qmul4<5>(kreg_1[5], sc0_1_1);
                }
                {
                    v8_0_1[3] = cake_mla_nvfp4_qmul4<6>(kreg_1[5], sc0_1_1);
                }
                {
                    v8_0_1[4] = cake_mla_nvfp4_qmul4<5>(kreg_1[6], sc1_2_1);
                }
                {
                    v8_0_1[5] = cake_mla_nvfp4_qmul4<6>(kreg_1[6], sc1_2_1);
                }
                {
                    v8_0_1[6] = cake_mla_nvfp4_qmul4<5>(kreg_1[7], sc1_2_1);
                }
                {
                    v8_0_1[7] = cake_mla_nvfp4_qmul4<6>(kreg_1[7], sc1_2_1);
                }
                int vrow_3_1 = smem_v_addr + 32768 + (unsigned int)(my_tok_3 * 128);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(vrow_3_1 + ((2 ^ swz_1) << 4)), "r"(*reinterpret_cast<uint32_t*>(&v8_0_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_0_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_0_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_0_1[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(vrow_3_1 + ((3 ^ swz_1) << 4)), "r"(*reinterpret_cast<uint32_t*>(&v8_0_1[4])), "r"(*reinterpret_cast<uint32_t*>(&v8_0_1[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_0_1[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_0_1[(4) + 3])));
                unsigned int v8_4_1[8];
                uint32_t _prmt_b32_20;
                asm("prmt.b32 %0, %1, %2, 0x0000;" : "=r"(_prmt_b32_20) : "r"(ssw_1[1]), "r"(ssw_1[1]));
                unsigned int sc0_5_1 = _prmt_b32_20;
                uint32_t _prmt_b32_21;
                asm("prmt.b32 %0, %1, %2, 0x1111;" : "=r"(_prmt_b32_21) : "r"(ssw_1[1]), "r"(ssw_1[1]));
                unsigned int sc1_6_1 = _prmt_b32_21;
                {
                    v8_4_1[0] = cake_mla_nvfp4_qmul4<5>(kreg_1[8], sc0_5_1);
                }
                {
                    v8_4_1[1] = cake_mla_nvfp4_qmul4<6>(kreg_1[8], sc0_5_1);
                }
                {
                    v8_4_1[2] = cake_mla_nvfp4_qmul4<5>(kreg_1[9], sc0_5_1);
                }
                {
                    v8_4_1[3] = cake_mla_nvfp4_qmul4<6>(kreg_1[9], sc0_5_1);
                }
                {
                    v8_4_1[4] = cake_mla_nvfp4_qmul4<5>(kreg_1[10], sc1_6_1);
                }
                {
                    v8_4_1[5] = cake_mla_nvfp4_qmul4<6>(kreg_1[10], sc1_6_1);
                }
                {
                    v8_4_1[6] = cake_mla_nvfp4_qmul4<5>(kreg_1[11], sc1_6_1);
                }
                {
                    v8_4_1[7] = cake_mla_nvfp4_qmul4<6>(kreg_1[11], sc1_6_1);
                }
                int vrow_7_1 = smem_v_addr + 32768 + (unsigned int)(my_tok_3 * 128);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(vrow_7_1 + ((4 ^ swz_1) << 4)), "r"(*reinterpret_cast<uint32_t*>(&v8_4_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_4_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_4_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_4_1[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(vrow_7_1 + ((5 ^ swz_1) << 4)), "r"(*reinterpret_cast<uint32_t*>(&v8_4_1[4])), "r"(*reinterpret_cast<uint32_t*>(&v8_4_1[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_4_1[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_4_1[(4) + 3])));
                unsigned int v8_8_1[8];
                uint32_t _prmt_b32_22;
                asm("prmt.b32 %0, %1, %2, 0x2222;" : "=r"(_prmt_b32_22) : "r"(ssw_1[1]), "r"(ssw_1[1]));
                unsigned int sc0_9_1 = _prmt_b32_22;
                uint32_t _prmt_b32_23;
                asm("prmt.b32 %0, %1, %2, 0x3333;" : "=r"(_prmt_b32_23) : "r"(ssw_1[1]), "r"(ssw_1[1]));
                unsigned int sc1_10_1 = _prmt_b32_23;
                {
                    v8_8_1[0] = cake_mla_nvfp4_qmul4<5>(kreg_1[12], sc0_9_1);
                }
                {
                    v8_8_1[1] = cake_mla_nvfp4_qmul4<6>(kreg_1[12], sc0_9_1);
                }
                {
                    v8_8_1[2] = cake_mla_nvfp4_qmul4<5>(kreg_1[13], sc0_9_1);
                }
                {
                    v8_8_1[3] = cake_mla_nvfp4_qmul4<6>(kreg_1[13], sc0_9_1);
                }
                {
                    v8_8_1[4] = cake_mla_nvfp4_qmul4<5>(kreg_1[14], sc1_10_1);
                }
                {
                    v8_8_1[5] = cake_mla_nvfp4_qmul4<6>(kreg_1[14], sc1_10_1);
                }
                {
                    v8_8_1[6] = cake_mla_nvfp4_qmul4<5>(kreg_1[15], sc1_10_1);
                }
                {
                    v8_8_1[7] = cake_mla_nvfp4_qmul4<6>(kreg_1[15], sc1_10_1);
                }
                int vrow_11_1 = smem_v_addr + 32768 + (unsigned int)(my_tok_3 * 128);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(vrow_11_1 + ((6 ^ swz_1) << 4)), "r"(*reinterpret_cast<uint32_t*>(&v8_8_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_8_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_8_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_8_1[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(vrow_11_1 + ((7 ^ swz_1) << 4)), "r"(*reinterpret_cast<uint32_t*>(&v8_8_1[4])), "r"(*reinterpret_cast<uint32_t*>(&v8_8_1[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_8_1[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_8_1[(4) + 3])));
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(v_full_addr + 16);
                if (tile_3 >= 1) {
                    mbarrier_wait(v_empty_addr + 24, tile_3 - 1 & 1);
                }
                unsigned int v8_12_1[8];
                uint32_t _prmt_b32_24;
                asm("prmt.b32 %0, %1, %2, 0x0000;" : "=r"(_prmt_b32_24) : "r"(ssw_1[2]), "r"(ssw_1[2]));
                unsigned int sc0_13_1 = _prmt_b32_24;
                uint32_t _prmt_b32_25;
                asm("prmt.b32 %0, %1, %2, 0x1111;" : "=r"(_prmt_b32_25) : "r"(ssw_1[2]), "r"(ssw_1[2]));
                unsigned int sc1_14_1 = _prmt_b32_25;
                {
                    v8_12_1[0] = cake_mla_nvfp4_qmul4<5>(kreg_1[16], sc0_13_1);
                }
                {
                    v8_12_1[1] = cake_mla_nvfp4_qmul4<6>(kreg_1[16], sc0_13_1);
                }
                {
                    v8_12_1[2] = cake_mla_nvfp4_qmul4<5>(kreg_1[17], sc0_13_1);
                }
                {
                    v8_12_1[3] = cake_mla_nvfp4_qmul4<6>(kreg_1[17], sc0_13_1);
                }
                {
                    v8_12_1[4] = cake_mla_nvfp4_qmul4<5>(kreg_1[18], sc1_14_1);
                }
                {
                    v8_12_1[5] = cake_mla_nvfp4_qmul4<6>(kreg_1[18], sc1_14_1);
                }
                {
                    v8_12_1[6] = cake_mla_nvfp4_qmul4<5>(kreg_1[19], sc1_14_1);
                }
                {
                    v8_12_1[7] = cake_mla_nvfp4_qmul4<6>(kreg_1[19], sc1_14_1);
                }
                int vrow_15_1 = smem_v_addr + 49152 + (unsigned int)(my_tok_3 * 128);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(vrow_15_1 + ((0 ^ swz_1) << 4)), "r"(*reinterpret_cast<uint32_t*>(&v8_12_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_12_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_12_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_12_1[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(vrow_15_1 + ((1 ^ swz_1) << 4)), "r"(*reinterpret_cast<uint32_t*>(&v8_12_1[4])), "r"(*reinterpret_cast<uint32_t*>(&v8_12_1[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_12_1[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_12_1[(4) + 3])));
                unsigned int v8_16_1[8];
                uint32_t _prmt_b32_26;
                asm("prmt.b32 %0, %1, %2, 0x2222;" : "=r"(_prmt_b32_26) : "r"(ssw_1[2]), "r"(ssw_1[2]));
                unsigned int sc0_17_1 = _prmt_b32_26;
                uint32_t _prmt_b32_27;
                asm("prmt.b32 %0, %1, %2, 0x3333;" : "=r"(_prmt_b32_27) : "r"(ssw_1[2]), "r"(ssw_1[2]));
                unsigned int sc1_18_1 = _prmt_b32_27;
                {
                    v8_16_1[0] = cake_mla_nvfp4_qmul4<5>(kreg_1[20], sc0_17_1);
                }
                {
                    v8_16_1[1] = cake_mla_nvfp4_qmul4<6>(kreg_1[20], sc0_17_1);
                }
                {
                    v8_16_1[2] = cake_mla_nvfp4_qmul4<5>(kreg_1[21], sc0_17_1);
                }
                {
                    v8_16_1[3] = cake_mla_nvfp4_qmul4<6>(kreg_1[21], sc0_17_1);
                }
                {
                    v8_16_1[4] = cake_mla_nvfp4_qmul4<5>(kreg_1[22], sc1_18_1);
                }
                {
                    v8_16_1[5] = cake_mla_nvfp4_qmul4<6>(kreg_1[22], sc1_18_1);
                }
                {
                    v8_16_1[6] = cake_mla_nvfp4_qmul4<5>(kreg_1[23], sc1_18_1);
                }
                {
                    v8_16_1[7] = cake_mla_nvfp4_qmul4<6>(kreg_1[23], sc1_18_1);
                }
                int vrow_19_1 = smem_v_addr + 49152 + (unsigned int)(my_tok_3 * 128);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(vrow_19_1 + ((2 ^ swz_1) << 4)), "r"(*reinterpret_cast<uint32_t*>(&v8_16_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_16_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_16_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_16_1[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(vrow_19_1 + ((3 ^ swz_1) << 4)), "r"(*reinterpret_cast<uint32_t*>(&v8_16_1[4])), "r"(*reinterpret_cast<uint32_t*>(&v8_16_1[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_16_1[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_16_1[(4) + 3])));
                unsigned int v8_20_1[8];
                uint32_t _prmt_b32_28;
                asm("prmt.b32 %0, %1, %2, 0x0000;" : "=r"(_prmt_b32_28) : "r"(ssw_1[3]), "r"(ssw_1[3]));
                unsigned int sc0_21_1 = _prmt_b32_28;
                uint32_t _prmt_b32_29;
                asm("prmt.b32 %0, %1, %2, 0x1111;" : "=r"(_prmt_b32_29) : "r"(ssw_1[3]), "r"(ssw_1[3]));
                unsigned int sc1_22_1 = _prmt_b32_29;
                {
                    v8_20_1[0] = cake_mla_nvfp4_qmul4<5>(kreg_1[24], sc0_21_1);
                }
                {
                    v8_20_1[1] = cake_mla_nvfp4_qmul4<6>(kreg_1[24], sc0_21_1);
                }
                {
                    v8_20_1[2] = cake_mla_nvfp4_qmul4<5>(kreg_1[25], sc0_21_1);
                }
                {
                    v8_20_1[3] = cake_mla_nvfp4_qmul4<6>(kreg_1[25], sc0_21_1);
                }
                {
                    v8_20_1[4] = cake_mla_nvfp4_qmul4<5>(kreg_1[26], sc1_22_1);
                }
                {
                    v8_20_1[5] = cake_mla_nvfp4_qmul4<6>(kreg_1[26], sc1_22_1);
                }
                {
                    v8_20_1[6] = cake_mla_nvfp4_qmul4<5>(kreg_1[27], sc1_22_1);
                }
                {
                    v8_20_1[7] = cake_mla_nvfp4_qmul4<6>(kreg_1[27], sc1_22_1);
                }
                int vrow_23_1 = smem_v_addr + 49152 + (unsigned int)(my_tok_3 * 128);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(vrow_23_1 + ((4 ^ swz_1) << 4)), "r"(*reinterpret_cast<uint32_t*>(&v8_20_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_20_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_20_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_20_1[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(vrow_23_1 + ((5 ^ swz_1) << 4)), "r"(*reinterpret_cast<uint32_t*>(&v8_20_1[4])), "r"(*reinterpret_cast<uint32_t*>(&v8_20_1[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_20_1[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_20_1[(4) + 3])));
                unsigned int v8_24_1[8];
                uint32_t _prmt_b32_30;
                asm("prmt.b32 %0, %1, %2, 0x2222;" : "=r"(_prmt_b32_30) : "r"(ssw_1[3]), "r"(ssw_1[3]));
                unsigned int sc0_25_1 = _prmt_b32_30;
                uint32_t _prmt_b32_31;
                asm("prmt.b32 %0, %1, %2, 0x3333;" : "=r"(_prmt_b32_31) : "r"(ssw_1[3]), "r"(ssw_1[3]));
                unsigned int sc1_26_1 = _prmt_b32_31;
                {
                    v8_24_1[0] = cake_mla_nvfp4_qmul4<5>(kreg_1[28], sc0_25_1);
                }
                {
                    v8_24_1[1] = cake_mla_nvfp4_qmul4<6>(kreg_1[28], sc0_25_1);
                }
                {
                    v8_24_1[2] = cake_mla_nvfp4_qmul4<5>(kreg_1[29], sc0_25_1);
                }
                {
                    v8_24_1[3] = cake_mla_nvfp4_qmul4<6>(kreg_1[29], sc0_25_1);
                }
                {
                    v8_24_1[4] = cake_mla_nvfp4_qmul4<5>(kreg_1[30], sc1_26_1);
                }
                {
                    v8_24_1[5] = cake_mla_nvfp4_qmul4<6>(kreg_1[30], sc1_26_1);
                }
                {
                    v8_24_1[6] = cake_mla_nvfp4_qmul4<5>(kreg_1[31], sc1_26_1);
                }
                {
                    v8_24_1[7] = cake_mla_nvfp4_qmul4<6>(kreg_1[31], sc1_26_1);
                }
                int vrow_27_1 = smem_v_addr + 49152 + (unsigned int)(my_tok_3 * 128);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(vrow_27_1 + ((6 ^ swz_1) << 4)), "r"(*reinterpret_cast<uint32_t*>(&v8_24_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_24_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_24_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_24_1[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(vrow_27_1 + ((7 ^ swz_1) << 4)), "r"(*reinterpret_cast<uint32_t*>(&v8_24_1[4])), "r"(*reinterpret_cast<uint32_t*>(&v8_24_1[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_24_1[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_24_1[(4) + 3])));
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(v_full_addr + 24);
                #pragma unroll
                for (int j_57 = 0; j_57 < 8; j_57++) {
                    sfw_1[j_57] = sfn_1[j_57];
                }
                #pragma unroll
                for (int j_58 = 0; j_58 < 32; j_58++) {
                    kreg_1[j_58] = kregn_1[j_58];
                }
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 16) {
        { // mma_warp_main
            int split_idx_4 = blockIdx.x;
            int m_tile_4 = gridDim.y - 1 - blockIdx.y;
            int b_4 = blockIdx.z;
            int q_start_4 = cum_seq_lens_q[b_4];
            int q_len_b_4 = cum_seq_lens_q[b_4 + 1] - q_start_4;
            int kv_len_4 = seq_lens[b_4];
            int g_len_4 = kv_len_global[b_4];
            int rows_b_4 = q_len_b_4 * num_heads;
            int row0_4 = m_tile_4 * 16;
            int rows_left_4 = rows_b_4 - row0_4;
            int rows_pos_4 = ((rows_left_4 < 0) ? 0 : rows_left_4);
            int rows_valid_4 = ((rows_pos_4 > 16) ? 16 : rows_pos_4);
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
            int my_start_4 = split_idx_4 * tiles_per_split_4;
            int my_end_raw_4 = my_start_4 + tiles_per_split_4;
            int my_end_4 = ((my_end_raw_4 > n_tiles_total_4) ? n_tiles_total_4 : my_end_raw_4);
            int my_n_raw_4 = my_end_4 - my_start_4;
            int my_n_tiles_4 = ((my_n_raw_4 < 0) ? 0 : my_n_raw_4);
            int pt_base_4 = b_4 * max_pages_per_seq;
            unsigned int _phase_q_full_0_1 = 0;
            mbarrier_wait(q_full_addr, _phase_q_full_0_1);
            _phase_q_full_0_1 ^= 1;
            unsigned int _phase_sfb_full_0 = 0;
            mbarrier_wait(sfb_full_addr, _phase_sfb_full_0);
            _phase_sfb_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (elect_sync()) {
                tcgen05_cp_32x128b_warpx4(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo512((((smem_sfb_addr) >> 4))));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo512((((smem_sfb_addr) >> 4) + 8)));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb + 8), make_sf_cp_desc_lo_sbo512((((smem_sfb_addr) >> 4) + 16)));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb + 12), make_sf_cp_desc_lo_sbo512((((smem_sfb_addr) >> 4) + 24)));
            }
            if (elect_sync()) {
                tcgen05_cp_32x128b_warpx4(tmem_tmem_sfb + 16, make_sf_cp_desc_lo_sbo512((((smem_sfb_addr) >> 4) + 128)));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb + 16 + 4), make_sf_cp_desc_lo_sbo512((((smem_sfb_addr) >> 4) + 128 + 8)));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb + 16 + 8), make_sf_cp_desc_lo_sbo512((((smem_sfb_addr) >> 4) + 128 + 16)));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb + 16 + 12), make_sf_cp_desc_lo_sbo512((((smem_sfb_addr) >> 4) + 128 + 24)));
            }
            #pragma unroll 1
            for (int tile_4 = 0; tile_4 < my_n_tiles_4; tile_4++) {
                int st = tile_4 % 2;
                int rnd = tile_4 / 2;
                int sbuf = tile_4 & 1;
                int sphase_2 = tile_4 & 1;
                if (tile_4 >= 2) {
                    mbarrier_wait(s_free_addr + (sphase_2) * 8, (tile_4 >> 1) - 1 & 1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                }
                if (tile_4 >= 2) {
                    int t0_8 = tile_4 - 2;
                    mbarrier_wait(pv_issued_addr + (t0_8 & 1) * 8, t0_8 >> 1 & 1);
                }
                mbarrier_wait(kv_full_addr + (st) * 8, rnd & 1);
                mbarrier_wait(sfa_full_addr + (sbuf) * 8, tile_4 >> 1 & 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_0 = make_warp_uniform((((smem_kr_addr) >> 4) & 0x3FFF) + (st) * 2816);
                int _mma_b_lo_0 = make_warp_uniform(((smem_qr_addr) >> 4) & 0x3FFF);
                {
                    uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x80004020U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                    uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x80004020U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (sphase_2 * 32)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134479888, 0);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (sphase_2 * 32)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134479888, 1);
                    }
                }
                int _mma_a_lo_1 = make_warp_uniform((((smem_v6_addr) >> 4) & 0x3FFF) + (st) * 2816);
                int _mma_b_lo_1 = make_warp_uniform(((smem_v0_addr) >> 4) & 0x3FFF);
                if (elect_sync()) {
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf4nvf4_bs((tmem_tmem + (sphase_2 * 32)), a_desc + 0, b_desc + 0,
                            0x8040480U, tmem_tmem_sfa + sbuf * 32 + 0, tmem_tmem_sfb + 0, 1);
                        tcgen05_mma_mxf4nvf4_bs((tmem_tmem + (sphase_2 * 32)), a_desc + 2, b_desc + 2,
                            0x8040480U, tmem_tmem_sfa + sbuf * 32 + 4, tmem_tmem_sfb + 4, 1);
                        tcgen05_mma_mxf4nvf4_bs((tmem_tmem + (sphase_2 * 32)), a_desc + 4, b_desc + 4,
                            0x8040480U, tmem_tmem_sfa + sbuf * 32 + 8, tmem_tmem_sfb + 8, 1);
                        tcgen05_mma_mxf4nvf4_bs((tmem_tmem + (sphase_2 * 32)), a_desc + 6, b_desc + 6,
                            0x8040480U, tmem_tmem_sfa + sbuf * 32 + 12, tmem_tmem_sfb + 12, 1);
                    }
                }
                int _mma_a_lo_2 = make_warp_uniform((((smem_v7_addr) >> 4) & 0x3FFF) + (st) * 2816);
                int _mma_b_lo_2 = make_warp_uniform(((smem_v1_addr) >> 4) & 0x3FFF);
                if (elect_sync()) {
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf4nvf4_bs((tmem_tmem + (sphase_2 * 32)), a_desc + 0, b_desc + 0,
                            0x8040480U, tmem_tmem_sfa + (sbuf * 32 + 16) + 0, tmem_tmem_sfb + 16 + 0, 1);
                        tcgen05_mma_mxf4nvf4_bs((tmem_tmem + (sphase_2 * 32)), a_desc + 2, b_desc + 2,
                            0x8040480U, tmem_tmem_sfa + (sbuf * 32 + 16) + 4, tmem_tmem_sfb + 16 + 4, 1);
                        tcgen05_mma_mxf4nvf4_bs((tmem_tmem + (sphase_2 * 32)), a_desc + 4, b_desc + 4,
                            0x8040480U, tmem_tmem_sfa + (sbuf * 32 + 16) + 8, tmem_tmem_sfb + 16 + 8, 1);
                        tcgen05_mma_mxf4nvf4_bs((tmem_tmem + (sphase_2 * 32)), a_desc + 6, b_desc + 6,
                            0x8040480U, tmem_tmem_sfa + (sbuf * 32 + 16) + 12, tmem_tmem_sfb + 16 + 12, 1);
                    }
                }
                elect_commit(s_full_addr + (sphase_2) * 8);
                elect_commit(kv_empty_qk_addr + (st) * 8);
                elect_commit(sfa_empty_addr + (sbuf) * 8);
            }
            int t_k = my_n_tiles_4 - 2;
            if (t_k >= 0) {
                mbarrier_wait(pv_issued_addr + (t_k & 1) * 8, t_k >> 1 & 1);
            }
            int t_k_0 = my_n_tiles_4 - 2 + 1;
            if (t_k_0 >= 0) {
                mbarrier_wait(pv_issued_addr + (t_k_0 & 1) * 8, t_k_0 >> 1 & 1);
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
    // ---- Role: pv_warp ----
    if (warp == 17) {
        { // pv_warp_main
            int split_idx_5 = blockIdx.x;
            int m_tile_5 = gridDim.y - 1 - blockIdx.y;
            int b_5 = blockIdx.z;
            int q_start_5 = cum_seq_lens_q[b_5];
            int q_len_b_5 = cum_seq_lens_q[b_5 + 1] - q_start_5;
            int kv_len_5 = seq_lens[b_5];
            int g_len_5 = kv_len_global[b_5];
            int rows_b_5 = q_len_b_5 * num_heads;
            int row0_5 = m_tile_5 * 16;
            int rows_left_5 = rows_b_5 - row0_5;
            int rows_pos_5 = ((rows_left_5 < 0) ? 0 : rows_left_5);
            int rows_valid_5 = ((rows_pos_5 > 16) ? 16 : rows_pos_5);
            int row_base_global_5 = q_start_5 * num_heads + row0_5;
            int last_row_5 = row0_5 + rows_valid_5 - 1;
            int t_last_5 = last_row_5 / num_heads;
            int num_5 = g_len_5 - q_len_b_5 + t_last_5 - cp_rank;
            int vis_raw_6 = num_5 / cp_world + 1;
            int vis_cap_6 = ((vis_raw_6 > kv_len_5) ? kv_len_5 : vis_raw_6);
            int vis_out_6 = ((num_5 < 0) ? 0 : vis_cap_6);
            int kv_end_raw_5 = vis_out_6;
            int kv_end_5 = ((rows_valid_5 == 0) ? 0 : kv_end_raw_5);
            int n_tiles_total_5 = (kv_end_5 + 128 - 1) / 128;
            int tiles_per_split_5 = (n_tiles_total_5 + num_split - 1) / num_split;
            int my_start_5 = split_idx_5 * tiles_per_split_5;
            int my_end_raw_5 = my_start_5 + tiles_per_split_5;
            int my_end_5 = ((my_end_raw_5 > n_tiles_total_5) ? n_tiles_total_5 : my_end_raw_5);
            int my_n_raw_5 = my_end_5 - my_start_5;
            int my_n_tiles_5 = ((my_n_raw_5 < 0) ? 0 : my_n_raw_5);
            int pt_base_5 = b_5 * max_pages_per_seq;
            #pragma unroll 1
            for (int tile_5 = 0; tile_5 < my_n_tiles_5; tile_5++) {
                int pv_pp = tile_5 & 1;
                int pv_wait = tile_5 >> 1 & 1;
                int acc = tile_5 & 1;
                int first_pv = ((tile_5 < 2) ? 1 : 0);
                mbarrier_wait(p_full_addr + (pv_pp) * 8, pv_wait);
                asm volatile("tcgen05.fence::after_thread_sync;");
                mbarrier_wait(v_full_addr, tile_5 & 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_3 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (0) * 1024);
                int _mma_b_lo_3 = make_warp_uniform(((((smem_pt_addr) >> 4) & 0x3FFF) | 0x4000000) + (pv_pp) * 1024);
                {
                    uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_3);
                    uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_3);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (64 + acc * 4 * 32)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134578192, ((first_pv) ? 0 : 1));
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 256U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 256U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (64 + acc * 4 * 32)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134578192, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 256U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 256U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (64 + acc * 4 * 32)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134578192, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 256U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 256U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (64 + acc * 4 * 32)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134578192, 1);
                    }
                }
                elect_commit(v_empty_addr);
                mbarrier_wait(v_full_addr + 8, tile_5 & 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_4 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (1) * 1024);
                int _mma_b_lo_4 = make_warp_uniform(((((smem_pt_addr) >> 4) & 0x3FFF) | 0x4000000) + (pv_pp) * 1024);
                {
                    uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_4);
                    uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_4);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (64 + (acc * 4 + 1) * 32)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134578192, ((first_pv) ? 0 : 1));
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 256U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 256U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (64 + (acc * 4 + 1) * 32)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134578192, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 256U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 256U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (64 + (acc * 4 + 1) * 32)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134578192, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 256U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 256U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (64 + (acc * 4 + 1) * 32)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134578192, 1);
                    }
                }
                elect_commit(v_empty_addr + 8);
                mbarrier_wait(v_full_addr + 16, tile_5 & 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_5 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (2) * 1024);
                int _mma_b_lo_5 = make_warp_uniform(((((smem_pt_addr) >> 4) & 0x3FFF) | 0x4000000) + (pv_pp) * 1024);
                {
                    uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_5);
                    uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_5);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (64 + (acc * 4 + 2) * 32)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134578192, ((first_pv) ? 0 : 1));
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 256U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 256U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (64 + (acc * 4 + 2) * 32)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134578192, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 256U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 256U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (64 + (acc * 4 + 2) * 32)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134578192, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 256U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 256U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (64 + (acc * 4 + 2) * 32)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134578192, 1);
                    }
                }
                elect_commit(v_empty_addr + 16);
                mbarrier_wait(v_full_addr + 24, tile_5 & 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_6 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (3) * 1024);
                int _mma_b_lo_6 = make_warp_uniform(((((smem_pt_addr) >> 4) & 0x3FFF) | 0x4000000) + (pv_pp) * 1024);
                {
                    uint64_t _mma_ss_a_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_6);
                    uint64_t _mma_ss_b_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_6);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (64 + (acc * 4 + 3) * 32)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134578192, ((first_pv) ? 0 : 1));
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_3, 256U);
                    incr_smem_desc_lo(_mma_ss_b_desc_3, 256U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (64 + (acc * 4 + 3) * 32)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134578192, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_3, 256U);
                    incr_smem_desc_lo(_mma_ss_b_desc_3, 256U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (64 + (acc * 4 + 3) * 32)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134578192, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_3, 256U);
                    incr_smem_desc_lo(_mma_ss_b_desc_3, 256U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (64 + (acc * 4 + 3) * 32)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134578192, 1);
                    }
                }
                elect_commit(v_empty_addr + 24);
                if (elect_sync()) {
                    mbarrier_arrive(pv_issued_addr + (tile_5 & 1) * 8);
                }
                elect_commit(pt_free_addr + (pv_pp) * 8);
                elect_commit(pv_done_addr + (2 * acc + (tile_5 >> 1 & 1)) * 8);
            }
            if (my_n_tiles_5 > 0) {
                elect_commit(o_done_addr);
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: load_warp ----
    if (warp == 18) {
        { // load_warp_main
            int split_idx_6 = blockIdx.x;
            int m_tile_6 = gridDim.y - 1 - blockIdx.y;
            int b_6 = blockIdx.z;
            int q_start_6 = cum_seq_lens_q[b_6];
            int q_len_b_6 = cum_seq_lens_q[b_6 + 1] - q_start_6;
            int kv_len_6 = seq_lens[b_6];
            int g_len_6 = kv_len_global[b_6];
            int rows_b_6 = q_len_b_6 * num_heads;
            int row0_6 = m_tile_6 * 16;
            int rows_left_6 = rows_b_6 - row0_6;
            int rows_pos_6 = ((rows_left_6 < 0) ? 0 : rows_left_6);
            int rows_valid_6 = ((rows_pos_6 > 16) ? 16 : rows_pos_6);
            int row_base_global_6 = q_start_6 * num_heads + row0_6;
            int last_row_6 = row0_6 + rows_valid_6 - 1;
            int t_last_6 = last_row_6 / num_heads;
            int num_6 = g_len_6 - q_len_b_6 + t_last_6 - cp_rank;
            int vis_raw_7 = num_6 / cp_world + 1;
            int vis_cap_7 = ((vis_raw_7 > kv_len_6) ? kv_len_6 : vis_raw_7);
            int vis_out_7 = ((num_6 < 0) ? 0 : vis_cap_7);
            int kv_end_raw_6 = vis_out_7;
            int kv_end_6 = ((rows_valid_6 == 0) ? 0 : kv_end_raw_6);
            int n_tiles_total_6 = (kv_end_6 + 128 - 1) / 128;
            int tiles_per_split_6 = (n_tiles_total_6 + num_split - 1) / num_split;
            int my_start_6 = split_idx_6 * tiles_per_split_6;
            int my_end_raw_6 = my_start_6 + tiles_per_split_6;
            int my_end_6 = ((my_end_raw_6 > n_tiles_total_6) ? n_tiles_total_6 : my_end_raw_6);
            int my_n_raw_6 = my_end_6 - my_start_6;
            int my_n_tiles_6 = ((my_n_raw_6 < 0) ? 0 : my_n_raw_6);
            int pt_base_6 = b_6 * max_pages_per_seq;
            int pg[4];
            int po[4];
            pg[0] = 0;
            po[0] = 0;
            pg[1] = 0;
            po[1] = 0;
            pg[2] = 0;
            po[2] = 0;
            pg[3] = 0;
            po[3] = 0;
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(q_full_addr, 5632);
                tma_2d_gmem2smem(smem_v0_addr, (&tmap_qn), 0, row_base_global_6, q_full_addr);
                tma_2d_gmem2smem(smem_v1_addr, (&tmap_qn), 128, row_base_global_6, q_full_addr);
                tma_2d_gmem2smem(smem_qs_addr, (&tmap_qs), 0, row_base_global_6, q_full_addr);
                tma_2d_gmem2smem(smem_qr_addr, (&tmap_qr), 0, row_base_global_6, q_full_addr);
            }
            if (my_n_tiles_6 > 0) {
                int tok_raw = my_start_6 * 128;
                int tok = ((tok_raw >= kv_end_6) ? my_start_6 * 128 : tok_raw);
                int pidx = tok >> page_shift;
                int off = tok - (pidx << page_shift);
                int g = page_table[pt_base_6 + pidx];
                pg[0] = g;
                po[0] = off;
                int tok_raw_0 = my_start_6 * 128 + 32;
                int tok_1 = ((tok_raw_0 >= kv_end_6) ? my_start_6 * 128 : tok_raw_0);
                int pidx_2 = tok_1 >> page_shift;
                int off_3 = tok_1 - (pidx_2 << page_shift);
                int g_4 = page_table[pt_base_6 + pidx_2];
                pg[1] = g_4;
                po[1] = off_3;
                int tok_raw_5 = my_start_6 * 128 + 64;
                int tok_6 = ((tok_raw_5 >= kv_end_6) ? my_start_6 * 128 : tok_raw_5);
                int pidx_7 = tok_6 >> page_shift;
                int off_8 = tok_6 - (pidx_7 << page_shift);
                int g_9 = page_table[pt_base_6 + pidx_7];
                pg[2] = g_9;
                po[2] = off_8;
                int tok_raw_10 = my_start_6 * 128 + 96;
                int tok_11 = ((tok_raw_10 >= kv_end_6) ? my_start_6 * 128 : tok_raw_10);
                int pidx_12 = tok_11 >> page_shift;
                int off_13 = tok_11 - (pidx_12 << page_shift);
                int g_14 = page_table[pt_base_6 + pidx_12];
                pg[3] = g_14;
                po[3] = off_13;
            }
            #pragma unroll 1
            for (int tile_6 = 0; tile_6 < my_n_tiles_6; tile_6++) {
                int st_1 = tile_6 % 2;
                int rnd_1 = tile_6 / 2;
                if (tile_6 >= 2) {
                    mbarrier_wait(kv_empty_qk_addr + (st_1) * 8, rnd_1 - 1 & 1);
                    mbarrier_wait(kv_empty_tr_addr + (st_1) * 8, rnd_1 - 1 & 1);
                }
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(kv_full_addr + (st_1) * 8, 45056);
                    tma_3d_gmem2smem(smem_v6_addr + (unsigned int)(st_1 * 45056), (&tmap_k), 0, po[0], pg[0], kv_full_addr + (st_1) * 8);
                    tma_3d_gmem2smem(smem_v7_addr + (unsigned int)(st_1 * 45056), (&tmap_k), 128, po[0], pg[0], kv_full_addr + (st_1) * 8);
                    tma_3d_gmem2smem(smem_ks_addr + (unsigned int)(st_1 * 45056), (&tmap_ks), 0, po[0], pg[0], kv_full_addr + (st_1) * 8);
                    tma_3d_gmem2smem(smem_kr_addr + (unsigned int)(st_1 * 45056), (&tmap_kr), 0, po[0], pg[0], kv_full_addr + (st_1) * 8);
                    tma_3d_gmem2smem(smem_v6_addr + (unsigned int)(st_1 * 45056) + 4096, (&tmap_k), 0, po[1], pg[1], kv_full_addr + (st_1) * 8);
                    tma_3d_gmem2smem(smem_v7_addr + (unsigned int)(st_1 * 45056) + 4096, (&tmap_k), 128, po[1], pg[1], kv_full_addr + (st_1) * 8);
                    tma_3d_gmem2smem(smem_ks_addr + (unsigned int)(st_1 * 45056) + 1024, (&tmap_ks), 0, po[1], pg[1], kv_full_addr + (st_1) * 8);
                    tma_3d_gmem2smem(smem_kr_addr + (unsigned int)(st_1 * 45056) + 2048, (&tmap_kr), 0, po[1], pg[1], kv_full_addr + (st_1) * 8);
                    tma_3d_gmem2smem(smem_v6_addr + (unsigned int)(st_1 * 45056) + 8192, (&tmap_k), 0, po[2], pg[2], kv_full_addr + (st_1) * 8);
                    tma_3d_gmem2smem(smem_v7_addr + (unsigned int)(st_1 * 45056) + 8192, (&tmap_k), 128, po[2], pg[2], kv_full_addr + (st_1) * 8);
                    tma_3d_gmem2smem(smem_ks_addr + (unsigned int)(st_1 * 45056) + 2048, (&tmap_ks), 0, po[2], pg[2], kv_full_addr + (st_1) * 8);
                    tma_3d_gmem2smem(smem_kr_addr + (unsigned int)(st_1 * 45056) + 4096, (&tmap_kr), 0, po[2], pg[2], kv_full_addr + (st_1) * 8);
                    tma_3d_gmem2smem(smem_v6_addr + (unsigned int)(st_1 * 45056) + 12288, (&tmap_k), 0, po[3], pg[3], kv_full_addr + (st_1) * 8);
                    tma_3d_gmem2smem(smem_v7_addr + (unsigned int)(st_1 * 45056) + 12288, (&tmap_k), 128, po[3], pg[3], kv_full_addr + (st_1) * 8);
                    tma_3d_gmem2smem(smem_ks_addr + (unsigned int)(st_1 * 45056) + 3072, (&tmap_ks), 0, po[3], pg[3], kv_full_addr + (st_1) * 8);
                    tma_3d_gmem2smem(smem_kr_addr + (unsigned int)(st_1 * 45056) + 6144, (&tmap_kr), 0, po[3], pg[3], kv_full_addr + (st_1) * 8);
                }
                int nxt_2 = tile_6 + 1;
                if (nxt_2 < my_n_tiles_6) {
                    int tok_raw_1 = (my_start_6 + nxt_2) * 128;
                    int tok_2 = ((tok_raw_1 >= kv_end_6) ? (my_start_6 + nxt_2) * 128 : tok_raw_1);
                    int pidx_1 = tok_2 >> page_shift;
                    int off_1 = tok_2 - (pidx_1 << page_shift);
                    int g_1 = page_table[pt_base_6 + pidx_1];
                    pg[0] = g_1;
                    po[0] = off_1;
                    int tok_raw_0_1 = (my_start_6 + nxt_2) * 128 + 32;
                    int tok_1_1 = ((tok_raw_0_1 >= kv_end_6) ? (my_start_6 + nxt_2) * 128 : tok_raw_0_1);
                    int pidx_2_1 = tok_1_1 >> page_shift;
                    int off_3_1 = tok_1_1 - (pidx_2_1 << page_shift);
                    int g_4_1 = page_table[pt_base_6 + pidx_2_1];
                    pg[1] = g_4_1;
                    po[1] = off_3_1;
                    int tok_raw_5_1 = (my_start_6 + nxt_2) * 128 + 64;
                    int tok_6_1 = ((tok_raw_5_1 >= kv_end_6) ? (my_start_6 + nxt_2) * 128 : tok_raw_5_1);
                    int pidx_7_1 = tok_6_1 >> page_shift;
                    int off_8_1 = tok_6_1 - (pidx_7_1 << page_shift);
                    int g_9_1 = page_table[pt_base_6 + pidx_7_1];
                    pg[2] = g_9_1;
                    po[2] = off_8_1;
                    int tok_raw_10_1 = (my_start_6 + nxt_2) * 128 + 96;
                    int tok_11_1 = ((tok_raw_10_1 >= kv_end_6) ? (my_start_6 + nxt_2) * 128 : tok_raw_10_1);
                    int pidx_12_1 = tok_11_1 >> page_shift;
                    int off_13_1 = tok_11_1 - (pidx_12_1 << page_shift);
                    int g_14_1 = page_table[pt_base_6 + pidx_12_1];
                    pg[3] = g_14_1;
                    po[3] = off_13_1;
                }
                if (nxt_2 < my_n_tiles_6) {
                    if (elect_sync()) {
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&tmap_k))), "r"((int)(0)), "r"((int)(po[0])), "r"((int)(pg[0])) : "memory");
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&tmap_k))), "r"((int)(128)), "r"((int)(po[0])), "r"((int)(pg[0])) : "memory");
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&tmap_ks))), "r"((int)(0)), "r"((int)(po[0])), "r"((int)(pg[0])) : "memory");
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&tmap_kr))), "r"((int)(0)), "r"((int)(po[0])), "r"((int)(pg[0])) : "memory");
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&tmap_k))), "r"((int)(0)), "r"((int)(po[1])), "r"((int)(pg[1])) : "memory");
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&tmap_k))), "r"((int)(128)), "r"((int)(po[1])), "r"((int)(pg[1])) : "memory");
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&tmap_ks))), "r"((int)(0)), "r"((int)(po[1])), "r"((int)(pg[1])) : "memory");
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&tmap_kr))), "r"((int)(0)), "r"((int)(po[1])), "r"((int)(pg[1])) : "memory");
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&tmap_k))), "r"((int)(0)), "r"((int)(po[2])), "r"((int)(pg[2])) : "memory");
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&tmap_k))), "r"((int)(128)), "r"((int)(po[2])), "r"((int)(pg[2])) : "memory");
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&tmap_ks))), "r"((int)(0)), "r"((int)(po[2])), "r"((int)(pg[2])) : "memory");
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&tmap_kr))), "r"((int)(0)), "r"((int)(po[2])), "r"((int)(pg[2])) : "memory");
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&tmap_k))), "r"((int)(0)), "r"((int)(po[3])), "r"((int)(pg[3])) : "memory");
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&tmap_k))), "r"((int)(128)), "r"((int)(po[3])), "r"((int)(pg[3])) : "memory");
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&tmap_ks))), "r"((int)(0)), "r"((int)(po[3])), "r"((int)(pg[3])) : "memory");
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&tmap_kr))), "r"((int)(0)), "r"((int)(po[3])), "r"((int)(pg[3])) : "memory");
                    }
                }
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: idle ----
    if (warp == 19) {
        // idle — no tasks assigned
    }

    // Cleanup
}

} // extern "C"
