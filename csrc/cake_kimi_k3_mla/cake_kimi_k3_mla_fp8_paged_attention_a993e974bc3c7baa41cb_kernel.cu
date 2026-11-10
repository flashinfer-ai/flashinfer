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

typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define LAUNCH_MIN_BLOCKS 1

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
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count) : "memory");
}



// CTA-local pipelines have short, resident producer/consumer edges.  Omitting
// suspendTimeHint keeps a miss on the lightweight TRYWAIT retry path; the
// explicit loop still makes this helper blocking until acquire succeeds.
__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}


__device__ __forceinline__ void tcgen05_mma_f8f6f4(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(enable_input_d)
         : "memory");
}



union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};

__device__ __forceinline__ void incr_smem_desc_lo(uint64_t& smem_desc, uint32_t offset) {
    MmaSmemDesc tmp;
    tmp.u64 = smem_desc;
    tmp.u32[0] += offset;
    smem_desc = tmp.u64;
}


__device__ __forceinline__ void elect_commit(int mbar_addr) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "}\n"
        :: "r"(mbar_addr));
}


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}


__device__ __forceinline__ void tmem_ld_x16(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=f"(dst[0]),  "=f"(dst[1]),  "=f"(dst[2]),  "=f"(dst[3]),
          "=f"(dst[4]),  "=f"(dst[5]),  "=f"(dst[6]),  "=f"(dst[7]),
          "=f"(dst[8]),  "=f"(dst[9]),  "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15])
        : "r"(tmem_addr));
}


__device__ __forceinline__ void tmem_st_x16_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x16.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8,"
        "  %9, %10, %11, %12, %13, %14, %15, %16};"
        :: "r"(tmem_addr),
           "f"(src[0]),  "f"(src[1]),  "f"(src[2]),  "f"(src[3]),
           "f"(src[4]),  "f"(src[5]),  "f"(src[6]),  "f"(src[7]),
           "f"(src[8]),  "f"(src[9]),  "f"(src[10]), "f"(src[11]),
           "f"(src[12]), "f"(src[13]), "f"(src[14]), "f"(src[15]));
}


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}




__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}



__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xFFFFFFFF;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_kimi_k3_mla_fp8_paged_attention_a993e974bc3c7baa41cb(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_qr, const __grid_constant__ CUtensorMap tmap_k, const __grid_constant__ CUtensorMap tmap_kr, __nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_max, float* __restrict__ partial_sum, int* __restrict__ seq_lens, int* __restrict__ cum_seq_lens_q, int* __restrict__ page_table, float softmax_scale_log2, float bmm2_scale, int num_heads, int num_split, int max_pages_per_seq, float* __restrict__ lse, int* __restrict__ causal_seqs, float lse_scale, int causal, int cp_shift, int cp_ishift, int cp_rank_off, int cp_span)
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
    #define kv_full_addr (mbar_base + 8)
    #define kv_empty_addr (mbar_base + 80)
    #define rope_full_addr (mbar_base + 152)
    #define rope_empty_addr (mbar_base + 160)
    #define s_full_addr (mbar_base + 168)
    #define s_free_addr (mbar_base + 176)
    #define p_full_addr (mbar_base + 184)
    #define pt_free_addr (mbar_base + 192)
    #define pv_done_addr (mbar_base + 200)
    #define o_done_addr (mbar_base + 208)
    #define tmem_dealloc_addr (mbar_base + 216)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    uint8_t* smem_q = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    uint8_t* smem_qr = reinterpret_cast<uint8_t*>(smem_raw + 50176);
    const int smem_qr_addr = smem + 50176;
    uint8_t* smem_k = reinterpret_cast<uint8_t*>(smem_raw + 56320);
    const int smem_k_addr = smem + 56320;
    uint8_t* smem_vt = reinterpret_cast<uint8_t*>(smem_raw + 56320);
    const int smem_vt_addr = smem + 56320;
    uint8_t* smem_kr = reinterpret_cast<uint8_t*>(smem_raw + 203776);
    const int smem_kr_addr = smem + 203776;
    uint8_t* smem_pt = reinterpret_cast<uint8_t*>(smem_raw + 211968);
    const int smem_pt_addr = smem + 211968;
    unsigned int* smem_pt_words = reinterpret_cast<unsigned int*>(smem_raw + 211968);
    const int smem_pt_words_addr = smem + 211968;
    float* smem_stats = reinterpret_cast<float*>(smem_raw + 228352);
    const int smem_stats_addr = smem + 228352;
    int* smem_vis = reinterpret_cast<int*>(smem_raw + 231808);
    const int smem_vis_addr = smem + 231808;
    int* smem_flags = reinterpret_cast<int*>(smem_raw + 232192);
    const int smem_flags_addr = smem + 232192;
    unsigned int* smem_stats_w = reinterpret_cast<unsigned int*>(smem_raw + 228352);
    const int smem_stats_w_addr = smem + 228352;
    unsigned int* smem_vis_w = reinterpret_cast<unsigned int*>(smem_raw + 231808);
    const int smem_vis_w_addr = smem + 231808;
    unsigned int* smem_flags_w = reinterpret_cast<unsigned int*>(smem_raw + 232192);
    const int smem_flags_w_addr = smem + 232192;

    // Mbarrier init (28 barriers at smem_raw[0..224))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // --- pipeline 'kv_pipe' ---
            // kv_full: 9 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // kv_empty: 9 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            // --- pipeline 'rope_pipe' ---
            // rope_full: 1 barriers, init_count=1
            mbarrier_init(smem + 152, 1);
            // rope_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            // s_full: 1 barriers, init_count=1
            mbarrier_init(smem + 168, 1);
            // s_free: 1 barriers, init_count=384
            mbarrier_init(smem + 176, 384);
            // p_full: 1 barriers, init_count=384
            mbarrier_init(smem + 184, 384);
            // pt_free: 1 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            // pv_done: 1 barriers, init_count=1
            mbarrier_init(smem + 200, 1);
            // o_done: 1 barriers, init_count=1
            mbarrier_init(smem + 208, 1);
            // tmem_dealloc: 1 barriers, init_count=480
            mbarrier_init(smem + 216, 480);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 480 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 224);
    if (warp == 0) {
        int _tmem_hold = smem + 224;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_tmem = taddr;

    // Kernel post-init ops
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 32;");
    }

    // ---- Role: softmax_wg0 ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 160;");
        { // softmax_wg0_main
            float psum0_0[16];
            float psum1_0[16];
            #pragma unroll
            for (int j_0 = 0; j_0 < 16; j_0++) {
                psum0_0[j_0] = 0.0f;
            }
            #pragma unroll
            for (int j_1 = 0; j_1 < 16; j_1++) {
                psum1_0[j_1] = 0.0f;
            }
            int split_idx = blockIdx.x;
            int m_tile = gridDim.y - 1 - blockIdx.y;
            int b = blockIdx.z;
            int q_start = cum_seq_lens_q[b];
            int q_len_b = cum_seq_lens_q[b + 1] - q_start;
            int kv_len = seq_lens[b];
            int rows_b = q_len_b * num_heads;
            int row0 = m_tile * 96;
            int rows_left = rows_b - row0;
            int rows_pos = ((rows_left < 0) ? 0 : rows_left);
            int rows_valid = ((rows_pos > 96) ? 96 : rows_pos);
            int row_base_global = q_start * num_heads + row0;
            int last_row = row0 + rows_valid - 1;
            int t_last = last_row / num_heads;
            int glen = causal_seqs[b];
            int seen_raw = glen - q_len_b + t_last + 1;
            int seen = ((seen_raw < 0) ? 0 : seen_raw);
            int full = seen >> cp_shift;
            int rem = seen - (full << cp_shift);
            int tail_raw = rem - cp_rank_off;
            int tail_lo = ((tail_raw < 0) ? 0 : tail_raw);
            int tail = ((tail_lo > cp_span) ? cp_span : tail_lo);
            int local = (full << cp_ishift) + tail;
            int local_c = ((local > kv_len) ? kv_len : local);
            int bound = ((causal != 0) ? local_c : kv_len);
            int kv_end = ((rows_valid == 0) ? 0 : bound);
            int n_pages = (kv_end + 64 - 1) / 64;
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
            int red_col = warp_in_wg * 8 + lane;
            int is_reducer = lane < 8;
            int row_stride = num_split * 512;
            int out_base = (row_base_global * num_split + split_idx) * 512 + my_tok;
            if (is_reducer != 0) {
                int q_tok = (row0 + red_col) / num_heads;
                int glen_r = causal_seqs[blockIdx.z];
                int seen_raw_0 = glen_r - q_len_b + q_tok + 1;
                int seen_0 = ((seen_raw_0 < 0) ? 0 : seen_raw_0);
                int full_0 = seen_0 >> cp_shift;
                int rem_0 = seen_0 - (full_0 << cp_shift);
                int tail_raw_0 = rem_0 - cp_rank_off;
                int tail_lo_0 = ((tail_raw_0 < 0) ? 0 : tail_raw_0);
                int tail_0 = ((tail_lo_0 > cp_span) ? cp_span : tail_lo_0);
                int local_0 = (full_0 << cp_ishift) + tail_0;
                int local_c_0 = ((local_0 > kv_len) ? kv_len : local_0);
                int bound_0 = ((causal != 0) ? local_c_0 : kv_len);
                smem_vis[red_col] = bound_0;
                smem_stats[576 + red_col] = 1031.8074f;
                smem_stats[672 + red_col] = -CUDART_INF_F;
            }
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            int vis_min = smem_vis[0];
            #pragma unroll 1
            for (int tile = 0; tile < my_n_tiles; tile++) {
                int pphase = tile & 0;
                int aphase = tile & 1;
                int sphase = 0;
                int _parity_s_full = (int)((unsigned int)tile % 2);
                mbarrier_wait(s_full_addr, _parity_s_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int t_abs = (my_start + tile) * 128 + my_tok;
                float tmem_load[16];
                tmem_ld_x16(&tmem_load[0], taddr + (unsigned int)(sphase * 96) + (unsigned int)(lane_base << 16));
                float tmem_load_0[16];
                tmem_ld_x16(&tmem_load_0[0], taddr + (unsigned int)(sphase * 96) + 16 + (unsigned int)(lane_base << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(s_free_addr + (sphase) * 8);
                int tile_last = (my_start + tile) * 128 + 127;
                if (tile_last >= vis_min) {
                    #pragma unroll
                    for (int k = 0; k < 16; k += 4) {
                        unsigned int smem_regs[4];
                        __int128_t _smem_raw_0;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_0) : "r"(smem_vis_w_addr + (unsigned int)(k * 4)));
                        smem_regs[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_0)[0];
                        smem_regs[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_0)[1];
                        smem_regs[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_0)[2];
                        smem_regs[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_0)[3];
                        #pragma unroll
                        for (int e = 0; e < 4; e++) {
                            int vis_e = 0;
                            unsigned int _bit_cast_src_1 = smem_regs[e];
                            vis_e = reinterpret_cast<int*>(&_bit_cast_src_1)[0];
                            if (t_abs >= vis_e) {
                                tmem_load[k + e] = -CUDART_INF_F;
                            }
                        }
                    }
                    #pragma unroll
                    for (int k_1 = 0; k_1 < 16; k_1 += 4) {
                        unsigned int smem_regs_1[4];
                        __int128_t _smem_raw_2;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_2) : "r"(smem_vis_w_addr + (unsigned int)((16 + k_1) * 4)));
                        smem_regs_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[0];
                        smem_regs_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[1];
                        smem_regs_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[2];
                        smem_regs_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[3];
                        #pragma unroll
                        for (int e_1 = 0; e_1 < 4; e_1++) {
                            int vis_e_1 = 0;
                            unsigned int _bit_cast_src_3 = smem_regs_1[e_1];
                            vis_e_1 = reinterpret_cast<int*>(&_bit_cast_src_3)[0];
                            if (t_abs >= vis_e_1) {
                                tmem_load_0[k_1 + e_1] = -CUDART_INF_F;
                            }
                        }
                    }
                }
                #pragma unroll
                for (int j = 0; j < 16; j++) {
                    float m_j;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xFFFFFFFF;" : "=f"(m_j) : "f"(tmem_load[j]));
                    if (lane == j % 32) {
                        smem_stats[192 + warp_in_wg * 32 + j] = m_j;
                    }
                }
                #pragma unroll
                for (int j_2 = 0; j_2 < 16; j_2++) {
                    float m_j_1;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xFFFFFFFF;" : "=f"(m_j_1) : "f"(tmem_load_0[j_2]));
                    if (lane == (16 + j_2) % 32) {
                        smem_stats[192 + warp_in_wg * 32 + 16 + j_2] = m_j_1;
                    }
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                if (is_reducer != 0) {
                    float r0 = smem_stats[192 + red_col];
                    float r1 = smem_stats[224 + red_col];
                    float r2 = smem_stats[256 + red_col];
                    float r3 = smem_stats[288 + red_col];
                    float tr;
                    asm("max.f32 %0, %1, %2;" : "=f"(tr) : "f"(max_noftz(r0, r1)), "f"(max_noftz(r2, r3)));
                    float cr_old = smem_stats[576 + red_col];
                    float cr_upd = 7.807355f - tr * softmax_scale_log2;
                    float run_old_r = smem_stats[672 + red_col];
                    float t_tile_r = tr * softmax_scale_log2;
                    float run_grown_r;
                    asm("max.f32 %0, %1, %2;" : "=f"(run_grown_r) : "f"(run_old_r), "f"(t_tile_r));
                    float run_new_r = ((tr > -CUDART_INF_F) ? run_grown_r : run_old_r);
                    float cr_floor = 7.807355f - run_new_r + 32.0f;
                    float cr_bounded = ((cr_upd <= cr_floor) ? cr_upd : cr_floor);
                    float cr_new = ((tr > -CUDART_INF_F) ? cr_bounded : cr_old);
                    float alpha_r = approx_exp2(cr_new - cr_old);
                    smem_stats[576 + red_col] = cr_new;
                    smem_stats[672 + red_col] = run_new_r;
                    smem_stats[red_col] = alpha_r;
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                if (tile >= 1) {
                    int _parity_pt_free = (int)((unsigned int)tile % 2) ^ 1;
                    mbarrier_wait(pt_free_addr, _parity_pt_free);
                }
                unsigned int packed[4];
                #pragma unroll
                for (int k_2 = 0; k_2 < 16; k_2 += 4) {
                    unsigned int smem_regs_2[4];
                    __int128_t _smem_raw_4;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_4) : "r"(smem_stats_w_addr + (unsigned int)((576 + k_2) * 4)));
                    smem_regs_2[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[0];
                    smem_regs_2[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[1];
                    smem_regs_2[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[2];
                    smem_regs_2[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[3];
                    unsigned int smem_regs_0[4];
                    __int128_t _smem_raw_5;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_5) : "r"(smem_stats_w_addr + (unsigned int)(k_2 * 4)));
                    smem_regs_0[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[0];
                    smem_regs_0[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[1];
                    smem_regs_0[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[2];
                    smem_regs_0[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[3];
                    #pragma unroll
                    for (int e_2 = 0; e_2 < 4; e_2++) {
                        float _exp2_6 = approx_exp2(tmem_load[k_2 + e_2] * softmax_scale_log2 + __uint_as_float(smem_regs_2[e_2]));
                        tmem_load[k_2 + e_2] = _exp2_6;
                    }
                    #pragma unroll
                    for (int e_3 = 0; e_3 < 4; e_3++) {
                        psum0_0[k_2 + e_3] = psum0_0[k_2 + e_3] * __uint_as_float(smem_regs_0[e_3]) + tmem_load[k_2 + e_3];
                    }
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
                        : "=r"(_packed) : "f"(tmem_load[0]), "f"(tmem_load[1]),
                                           "f"(tmem_load[2]), "f"(tmem_load[3]));
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
                        : "=r"(_packed) : "f"(tmem_load[4]), "f"(tmem_load[5]),
                                           "f"(tmem_load[6]), "f"(tmem_load[7]));
                    packed[1] = _packed;
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
                        : "=r"(_packed) : "f"(tmem_load[8]), "f"(tmem_load[9]),
                                           "f"(tmem_load[10]), "f"(tmem_load[11]));
                    packed[2] = _packed;
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
                        : "=r"(_packed) : "f"(tmem_load[12]), "f"(tmem_load[13]),
                                           "f"(tmem_load[14]), "f"(tmem_load[15]));
                    packed[3] = _packed;
                }
                int row_addr = smem_pt_addr + (unsigned int)(pphase * 16384) + (unsigned int)(my_tok * 128);
                int row_rel = pphase * 16384 + my_tok * 128;
                int dst = row_addr + ((0 ^ my_tok & 7) << 4);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dst), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3]) : "memory");
                unsigned int packed_0[4];
                #pragma unroll
                for (int k_3 = 0; k_3 < 16; k_3 += 4) {
                    unsigned int smem_regs_3[4];
                    __int128_t _smem_raw_7;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_7) : "r"(smem_stats_w_addr + (unsigned int)((592 + k_3) * 4)));
                    smem_regs_3[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[0];
                    smem_regs_3[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[1];
                    smem_regs_3[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[2];
                    smem_regs_3[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[3];
                    unsigned int smem_regs_1_1[4];
                    __int128_t _smem_raw_8;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_8) : "r"(smem_stats_w_addr + (unsigned int)((16 + k_3) * 4)));
                    smem_regs_1_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[0];
                    smem_regs_1_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[1];
                    smem_regs_1_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[2];
                    smem_regs_1_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[3];
                    #pragma unroll
                    for (int e_4 = 0; e_4 < 4; e_4++) {
                        float _exp2_9 = approx_exp2(tmem_load_0[k_3 + e_4] * softmax_scale_log2 + __uint_as_float(smem_regs_3[e_4]));
                        tmem_load_0[k_3 + e_4] = _exp2_9;
                    }
                    #pragma unroll
                    for (int e_5 = 0; e_5 < 4; e_5++) {
                        psum1_0[k_3 + e_5] = psum1_0[k_3 + e_5] * __uint_as_float(smem_regs_1_1[e_5]) + tmem_load_0[k_3 + e_5];
                    }
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
                        : "=r"(_packed) : "f"(tmem_load_0[0]), "f"(tmem_load_0[1]),
                                           "f"(tmem_load_0[2]), "f"(tmem_load_0[3]));
                    packed_0[0] = _packed;
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
                        : "=r"(_packed) : "f"(tmem_load_0[4]), "f"(tmem_load_0[5]),
                                           "f"(tmem_load_0[6]), "f"(tmem_load_0[7]));
                    packed_0[1] = _packed;
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
                        : "=r"(_packed) : "f"(tmem_load_0[8]), "f"(tmem_load_0[9]),
                                           "f"(tmem_load_0[10]), "f"(tmem_load_0[11]));
                    packed_0[2] = _packed;
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
                        : "=r"(_packed) : "f"(tmem_load_0[12]), "f"(tmem_load_0[13]),
                                           "f"(tmem_load_0[14]), "f"(tmem_load_0[15]));
                    packed_0[3] = _packed;
                }
                int row_addr_0 = smem_pt_addr + (unsigned int)(pphase * 16384) + (unsigned int)(my_tok * 128);
                int row_rel_0 = pphase * 16384 + my_tok * 128;
                int dst_0 = row_addr_0 + ((1 ^ my_tok & 7) << 4);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dst_0), "r"(packed_0[0]), "r"(packed_0[1]), "r"(packed_0[2]), "r"(packed_0[3]) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (tile > 0) {
                    int _parity_pv_done = (int)((unsigned int)tile % 2) ^ 1;
                    mbarrier_wait(pv_done_addr, _parity_pv_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll
                    for (int vs = 0; vs < 4; vs++) {
                        int o_a = taddr + 96 + (unsigned int)(vs * 96) + (unsigned int)(lane_base << 16);
                        int o_b = taddr + 96 + (unsigned int)(vs * 96) + 16 + (unsigned int)(lane_base << 16);
                        float _tmem_load_2[16];
                        tmem_ld_x16(&_tmem_load_2[0], o_a);
                        float _tmem_load_3[16];
                        tmem_ld_x16(&_tmem_load_3[0], o_b);
                        float vals[16];
                        #pragma unroll
                        for (int k_4 = 0; k_4 < 16; k_4 += 4) {
                            unsigned int smem_regs_4[4];
                            __int128_t _smem_raw_10;
                            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_10) : "r"(smem_stats_w_addr + (unsigned int)(k_4 * 4)));
                            smem_regs_4[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[0];
                            smem_regs_4[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[1];
                            smem_regs_4[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[2];
                            smem_regs_4[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[3];
                            #pragma unroll
                            for (int e_6 = 0; e_6 < 4; e_6++) {
                                vals[k_4 + e_6] = __uint_as_float(smem_regs_4[e_6]);
                            }
                        }
                        float vals_0[16];
                        #pragma unroll
                        for (int k_5 = 0; k_5 < 16; k_5 += 4) {
                            unsigned int smem_regs_5[4];
                            __int128_t _smem_raw_11;
                            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_11) : "r"(smem_stats_w_addr + (unsigned int)((16 + k_5) * 4)));
                            smem_regs_5[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[0];
                            smem_regs_5[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[1];
                            smem_regs_5[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[2];
                            smem_regs_5[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[3];
                            #pragma unroll
                            for (int e_7 = 0; e_7 < 4; e_7++) {
                                vals_0[k_5 + e_7] = __uint_as_float(smem_regs_5[e_7]);
                            }
                        }
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        #pragma unroll
                        for (int j_3 = 0; j_3 < 16; j_3++) {
                            _tmem_load_2[j_3] = _tmem_load_2[j_3] * vals[j_3];
                        }
                        #pragma unroll
                        for (int j_4 = 0; j_4 < 16; j_4++) {
                            _tmem_load_3[j_4] = _tmem_load_3[j_4] * vals_0[j_4];
                        }
                        tmem_st_x16_f32(o_a, _tmem_load_2);
                        tmem_st_x16_f32(o_b, _tmem_load_3);
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(p_full_addr);
            }
            #pragma unroll
            for (int j_5 = 0; j_5 < 16; j_5++) {
                float wsum_j = psum0_0[j_5];
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    wsum_j += __shfl_xor_sync(0xFFFFFFFF, wsum_j, offset);
                if (lane == j_5 % 32) {
                    smem_stats[192 + warp_in_wg * 32 + j_5] = wsum_j;
                }
            }
            #pragma unroll
            for (int j_6 = 0; j_6 < 16; j_6++) {
                float wsum_j_1 = psum1_0[j_6];
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    wsum_j_1 += __shfl_xor_sync(0xFFFFFFFF, wsum_j_1, offset);
                if (lane == (16 + j_6) % 32) {
                    smem_stats[192 + warp_in_wg * 32 + 16 + j_6] = wsum_j_1;
                }
            }
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (is_reducer != 0) {
                float s0 = smem_stats[192 + red_col];
                float s1 = smem_stats[224 + red_col];
                float s2 = smem_stats[256 + red_col];
                float s3 = smem_stats[288 + red_col];
                float csum = s0 + s1 + (s2 + s3);
                float safe_sum = ((csum > 0.0f) ? csum : 1.0f);
                float out_scale = ((num_split == 1) ? bmm2_scale : 1.0f);
                float _rcp_12 = approx_rcp(safe_sum);
                smem_stats[768 + red_col] = _rcp_12 * out_scale;
                if (red_col < rows_valid) {
                    int stat_off = (row_base_global + red_col) * num_split + split_idx;
                    float c_fin = smem_stats[576 + red_col];
                    float stored_max = 7.807355f - c_fin;
                    *(reinterpret_cast<float*>(partial_max + stat_off) + (0)) = stored_max;
                    *(reinterpret_cast<float*>(partial_sum + stat_off) + (0)) = csum;
                    if (num_split == 1) {
                        float _log2_13;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_13) : "f"(csum));
                        float lse_row = (_log2_13 - c_fin) * lse_scale;
                        *(reinterpret_cast<float*>(lse + (row_base_global + red_col)) + (0)) = lse_row;
                    }
                }
            }
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (my_n_tiles == 0) {
                if (num_split == 1) {
                    #pragma unroll 1
                    for (int c = 0; c < rows_valid; c++) {
                        #pragma unroll
                        for (int vs_1 = 0; vs_1 < 4; vs_1++) {
                            *(reinterpret_cast<__nv_bfloat16*>(partial_O + (out_base + c * row_stride + vs_1 * 128)) + (0)) = 0.0f;
                        }
                    }
                }
            }
            unsigned int _phase_o_done_0 = 0;
            if (my_n_tiles > 0) {
                float vals_1[32];
                #pragma unroll
                for (int k_6 = 0; k_6 < 32; k_6 += 4) {
                    unsigned int smem_regs_6[4];
                    __int128_t _smem_raw_14;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_14) : "r"(smem_stats_w_addr + (unsigned int)((768 + k_6) * 4)));
                    smem_regs_6[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[0];
                    smem_regs_6[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[1];
                    smem_regs_6[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[2];
                    smem_regs_6[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[3];
                    #pragma unroll
                    for (int e_8 = 0; e_8 < 4; e_8++) {
                        vals_1[k_6 + e_8] = __uint_as_float(smem_regs_6[e_8]);
                    }
                }
                mbarrier_wait(o_done_addr, _phase_o_done_0);
                _phase_o_done_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (int vs_2 = 0; vs_2 < 4; vs_2++) {
                    float _tmem_load_4[16];
                    tmem_ld_x16(&_tmem_load_4[0], taddr + 96 + (unsigned int)(vs_2 * 96) + (unsigned int)(lane_base << 16));
                    #pragma unroll
                    for (int j_7 = 0; j_7 < 16; j_7++) {
                        int c_1 = j_7;
                        if (c_1 < rows_valid) {
                            *(reinterpret_cast<__nv_bfloat16*>(partial_O + (out_base + c_1 * row_stride + vs_2 * 128)) + (0)) = (__nv_bfloat16)(_tmem_load_4[j_7] * vals_1[j_7]);
                        }
                    }
                    float tmem_load_1[16];
                    tmem_ld_x16(&tmem_load_1[0], taddr + 96 + (unsigned int)(vs_2 * 96) + 16 + (unsigned int)(lane_base << 16));
                    #pragma unroll
                    for (int j_8 = 0; j_8 < 16; j_8++) {
                        int c_2 = 16 + j_8;
                        if (c_2 < rows_valid) {
                            *(reinterpret_cast<__nv_bfloat16*>(partial_O + (out_base + c_2 * row_stride + vs_2 * 128)) + (0)) = (__nv_bfloat16)(tmem_load_1[j_8] * vals_1[16 + j_8]);
                        }
                    }
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: softmax_wg1 ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 160;");
        { // softmax_wg1_main
            float psum0_1[16];
            float psum1_1[16];
            #pragma unroll
            for (int j_2_1 = 0; j_2_1 < 16; j_2_1++) {
                psum0_1[j_2_1] = 0.0f;
            }
            #pragma unroll
            for (int j_3_1 = 0; j_3_1 < 16; j_3_1++) {
                psum1_1[j_3_1] = 0.0f;
            }
            int split_idx_1 = blockIdx.x;
            int m_tile_1 = gridDim.y - 1 - blockIdx.y;
            int b_1 = blockIdx.z;
            int q_start_1 = cum_seq_lens_q[b_1];
            int q_len_b_1 = cum_seq_lens_q[b_1 + 1] - q_start_1;
            int kv_len_1 = seq_lens[b_1];
            int rows_b_1 = q_len_b_1 * num_heads;
            int row0_1 = m_tile_1 * 96;
            int rows_left_1 = rows_b_1 - row0_1;
            int rows_pos_1 = ((rows_left_1 < 0) ? 0 : rows_left_1);
            int rows_valid_1 = ((rows_pos_1 > 96) ? 96 : rows_pos_1);
            int row_base_global_1 = q_start_1 * num_heads + row0_1;
            int last_row_1 = row0_1 + rows_valid_1 - 1;
            int t_last_1 = last_row_1 / num_heads;
            int glen_1 = causal_seqs[b_1];
            int seen_raw_1 = glen_1 - q_len_b_1 + t_last_1 + 1;
            int seen_1 = ((seen_raw_1 < 0) ? 0 : seen_raw_1);
            int full_1 = seen_1 >> cp_shift;
            int rem_1 = seen_1 - (full_1 << cp_shift);
            int tail_raw_1 = rem_1 - cp_rank_off;
            int tail_lo_1 = ((tail_raw_1 < 0) ? 0 : tail_raw_1);
            int tail_1 = ((tail_lo_1 > cp_span) ? cp_span : tail_lo_1);
            int local_1 = (full_1 << cp_ishift) + tail_1;
            int local_c_1 = ((local_1 > kv_len_1) ? kv_len_1 : local_1);
            int bound_1 = ((causal != 0) ? local_c_1 : kv_len_1);
            int kv_end_1 = ((rows_valid_1 == 0) ? 0 : bound_1);
            int n_pages_1 = (kv_end_1 + 64 - 1) / 64;
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
            int red_col_1 = 32 + warp_in_wg_1 * 8 + lane;
            int is_reducer_1 = lane < 8;
            int row_stride_1 = num_split * 512;
            int out_base_1 = (row_base_global_1 * num_split + split_idx_1) * 512 + my_tok_1;
            if (is_reducer_1 != 0) {
                int q_tok_1 = (row0_1 + red_col_1) / num_heads;
                int glen_r_1 = causal_seqs[blockIdx.z];
                int seen_raw_1_1 = glen_r_1 - q_len_b_1 + q_tok_1 + 1;
                int seen_1_1 = ((seen_raw_1_1 < 0) ? 0 : seen_raw_1_1);
                int full_1_1 = seen_1_1 >> cp_shift;
                int rem_1_1 = seen_1_1 - (full_1_1 << cp_shift);
                int tail_raw_1_1 = rem_1_1 - cp_rank_off;
                int tail_lo_1_1 = ((tail_raw_1_1 < 0) ? 0 : tail_raw_1_1);
                int tail_1_1 = ((tail_lo_1_1 > cp_span) ? cp_span : tail_lo_1_1);
                int local_1_1 = (full_1_1 << cp_ishift) + tail_1_1;
                int local_c_1_1 = ((local_1_1 > kv_len_1) ? kv_len_1 : local_1_1);
                int bound_1_1 = ((causal != 0) ? local_c_1_1 : kv_len_1);
                smem_vis[red_col_1] = bound_1_1;
                smem_stats[576 + red_col_1] = 1031.8074f;
                smem_stats[672 + red_col_1] = -CUDART_INF_F;
            }
            asm volatile("barrier.sync 2, 128;" ::: "memory");
            int vis_min_1 = smem_vis[32];
            #pragma unroll 1
            for (int tile_1 = 0; tile_1 < my_n_tiles_1; tile_1++) {
                int pphase_1 = tile_1 & 0;
                int aphase_1 = tile_1 & 1;
                int sphase_1 = 0;
                int _parity_s_full_1 = (int)((unsigned int)tile_1 % 2);
                mbarrier_wait(s_full_addr, _parity_s_full_1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int t_abs_1 = (my_start_1 + tile_1) * 128 + my_tok_1;
                float tmem_load_2[16];
                tmem_ld_x16(&tmem_load_2[0], taddr + (unsigned int)(sphase_1 * 96) + 32 + (unsigned int)(lane_base_1 << 16));
                float tmem_load_2_1[16];
                tmem_ld_x16(&tmem_load_2_1[0], taddr + (unsigned int)(sphase_1 * 96) + 32 + 16 + (unsigned int)(lane_base_1 << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(s_free_addr + (sphase_1) * 8);
                int tile_last_1 = (my_start_1 + tile_1) * 128 + 127;
                if (tile_last_1 >= vis_min_1) {
                    #pragma unroll
                    for (int k_7 = 0; k_7 < 16; k_7 += 4) {
                        unsigned int smem_regs_7[4];
                        __int128_t _smem_raw_0;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_0) : "r"(smem_vis_w_addr + (unsigned int)((32 + k_7) * 4)));
                        smem_regs_7[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_0)[0];
                        smem_regs_7[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_0)[1];
                        smem_regs_7[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_0)[2];
                        smem_regs_7[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_0)[3];
                        #pragma unroll
                        for (int e_9 = 0; e_9 < 4; e_9++) {
                            int vis_e_2 = 0;
                            unsigned int _bit_cast_src_1 = smem_regs_7[e_9];
                            vis_e_2 = reinterpret_cast<int*>(&_bit_cast_src_1)[0];
                            if (t_abs_1 >= vis_e_2) {
                                tmem_load_2[k_7 + e_9] = -CUDART_INF_F;
                            }
                        }
                    }
                    #pragma unroll
                    for (int k_8 = 0; k_8 < 16; k_8 += 4) {
                        unsigned int smem_regs_8[4];
                        __int128_t _smem_raw_2;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_2) : "r"(smem_vis_w_addr + (unsigned int)((48 + k_8) * 4)));
                        smem_regs_8[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[0];
                        smem_regs_8[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[1];
                        smem_regs_8[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[2];
                        smem_regs_8[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[3];
                        #pragma unroll
                        for (int e_10 = 0; e_10 < 4; e_10++) {
                            int vis_e_3 = 0;
                            unsigned int _bit_cast_src_3 = smem_regs_8[e_10];
                            vis_e_3 = reinterpret_cast<int*>(&_bit_cast_src_3)[0];
                            if (t_abs_1 >= vis_e_3) {
                                tmem_load_2_1[k_8 + e_10] = -CUDART_INF_F;
                            }
                        }
                    }
                }
                #pragma unroll
                for (int j_9 = 0; j_9 < 16; j_9++) {
                    float m_j_2;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xFFFFFFFF;" : "=f"(m_j_2) : "f"(tmem_load_2[j_9]));
                    if (lane == j_9 % 32) {
                        smem_stats[320 + warp_in_wg_1 * 32 + j_9] = m_j_2;
                    }
                }
                #pragma unroll
                for (int j_10 = 0; j_10 < 16; j_10++) {
                    float m_j_3;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xFFFFFFFF;" : "=f"(m_j_3) : "f"(tmem_load_2_1[j_10]));
                    if (lane == (16 + j_10) % 32) {
                        smem_stats[320 + warp_in_wg_1 * 32 + 16 + j_10] = m_j_3;
                    }
                }
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                if (is_reducer_1 != 0) {
                    float r0_1 = smem_stats[320 + (red_col_1 - 32)];
                    float r1_1 = smem_stats[352 + (red_col_1 - 32)];
                    float r2_1 = smem_stats[384 + (red_col_1 - 32)];
                    float r3_1 = smem_stats[416 + (red_col_1 - 32)];
                    float tr_1;
                    asm("max.f32 %0, %1, %2;" : "=f"(tr_1) : "f"(max_noftz(r0_1, r1_1)), "f"(max_noftz(r2_1, r3_1)));
                    float cr_old_1 = smem_stats[576 + red_col_1];
                    float cr_upd_1 = 7.807355f - tr_1 * softmax_scale_log2;
                    float run_old_r_1 = smem_stats[672 + red_col_1];
                    float t_tile_r_1 = tr_1 * softmax_scale_log2;
                    float run_grown_r_1;
                    asm("max.f32 %0, %1, %2;" : "=f"(run_grown_r_1) : "f"(run_old_r_1), "f"(t_tile_r_1));
                    float run_new_r_1 = ((tr_1 > -CUDART_INF_F) ? run_grown_r_1 : run_old_r_1);
                    float cr_floor_1 = 7.807355f - run_new_r_1 + 32.0f;
                    float cr_bounded_1 = ((cr_upd_1 <= cr_floor_1) ? cr_upd_1 : cr_floor_1);
                    float cr_new_1 = ((tr_1 > -CUDART_INF_F) ? cr_bounded_1 : cr_old_1);
                    float alpha_r_1 = approx_exp2(cr_new_1 - cr_old_1);
                    smem_stats[576 + red_col_1] = cr_new_1;
                    smem_stats[672 + red_col_1] = run_new_r_1;
                    smem_stats[red_col_1] = alpha_r_1;
                }
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                if (tile_1 >= 1) {
                    int _parity_pt_free_1 = (int)((unsigned int)tile_1 % 2) ^ 1;
                    mbarrier_wait(pt_free_addr, _parity_pt_free_1);
                }
                unsigned int packed_1[4];
                #pragma unroll
                for (int k_9 = 0; k_9 < 16; k_9 += 4) {
                    unsigned int smem_regs_9[4];
                    __int128_t _smem_raw_4;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_4) : "r"(smem_stats_w_addr + (unsigned int)((608 + k_9) * 4)));
                    smem_regs_9[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[0];
                    smem_regs_9[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[1];
                    smem_regs_9[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[2];
                    smem_regs_9[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[3];
                    unsigned int smem_regs_2_1[4];
                    __int128_t _smem_raw_5;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_5) : "r"(smem_stats_w_addr + (unsigned int)((32 + k_9) * 4)));
                    smem_regs_2_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[0];
                    smem_regs_2_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[1];
                    smem_regs_2_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[2];
                    smem_regs_2_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[3];
                    #pragma unroll
                    for (int e_11 = 0; e_11 < 4; e_11++) {
                        float _exp2_6_1 = approx_exp2(tmem_load_2[k_9 + e_11] * softmax_scale_log2 + __uint_as_float(smem_regs_9[e_11]));
                        tmem_load_2[k_9 + e_11] = _exp2_6_1;
                    }
                    #pragma unroll
                    for (int e_12 = 0; e_12 < 4; e_12++) {
                        psum0_1[k_9 + e_12] = psum0_1[k_9 + e_12] * __uint_as_float(smem_regs_2_1[e_12]) + tmem_load_2[k_9 + e_12];
                    }
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
                        : "=r"(_packed) : "f"(tmem_load_2[0]), "f"(tmem_load_2[1]),
                                           "f"(tmem_load_2[2]), "f"(tmem_load_2[3]));
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
                        : "=r"(_packed) : "f"(tmem_load_2[4]), "f"(tmem_load_2[5]),
                                           "f"(tmem_load_2[6]), "f"(tmem_load_2[7]));
                    packed_1[1] = _packed;
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
                        : "=r"(_packed) : "f"(tmem_load_2[8]), "f"(tmem_load_2[9]),
                                           "f"(tmem_load_2[10]), "f"(tmem_load_2[11]));
                    packed_1[2] = _packed;
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
                        : "=r"(_packed) : "f"(tmem_load_2[12]), "f"(tmem_load_2[13]),
                                           "f"(tmem_load_2[14]), "f"(tmem_load_2[15]));
                    packed_1[3] = _packed;
                }
                int row_addr_1 = smem_pt_addr + (unsigned int)(pphase_1 * 16384) + (unsigned int)(my_tok_1 * 128);
                int row_rel_1 = pphase_1 * 16384 + my_tok_1 * 128;
                int dst_1 = row_addr_1 + ((2 ^ my_tok_1 & 7) << 4);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dst_1), "r"(packed_1[0]), "r"(packed_1[1]), "r"(packed_1[2]), "r"(packed_1[3]) : "memory");
                unsigned int packed_1_1[4];
                #pragma unroll
                for (int k_10 = 0; k_10 < 16; k_10 += 4) {
                    unsigned int smem_regs_10[4];
                    __int128_t _smem_raw_7;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_7) : "r"(smem_stats_w_addr + (unsigned int)((624 + k_10) * 4)));
                    smem_regs_10[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[0];
                    smem_regs_10[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[1];
                    smem_regs_10[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[2];
                    smem_regs_10[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[3];
                    unsigned int smem_regs_3_1[4];
                    __int128_t _smem_raw_8;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_8) : "r"(smem_stats_w_addr + (unsigned int)((48 + k_10) * 4)));
                    smem_regs_3_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[0];
                    smem_regs_3_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[1];
                    smem_regs_3_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[2];
                    smem_regs_3_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[3];
                    #pragma unroll
                    for (int e_13 = 0; e_13 < 4; e_13++) {
                        float _exp2_9_1 = approx_exp2(tmem_load_2_1[k_10 + e_13] * softmax_scale_log2 + __uint_as_float(smem_regs_10[e_13]));
                        tmem_load_2_1[k_10 + e_13] = _exp2_9_1;
                    }
                    #pragma unroll
                    for (int e_14 = 0; e_14 < 4; e_14++) {
                        psum1_1[k_10 + e_14] = psum1_1[k_10 + e_14] * __uint_as_float(smem_regs_3_1[e_14]) + tmem_load_2_1[k_10 + e_14];
                    }
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
                        : "=r"(_packed) : "f"(tmem_load_2_1[0]), "f"(tmem_load_2_1[1]),
                                           "f"(tmem_load_2_1[2]), "f"(tmem_load_2_1[3]));
                    packed_1_1[0] = _packed;
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
                        : "=r"(_packed) : "f"(tmem_load_2_1[4]), "f"(tmem_load_2_1[5]),
                                           "f"(tmem_load_2_1[6]), "f"(tmem_load_2_1[7]));
                    packed_1_1[1] = _packed;
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
                        : "=r"(_packed) : "f"(tmem_load_2_1[8]), "f"(tmem_load_2_1[9]),
                                           "f"(tmem_load_2_1[10]), "f"(tmem_load_2_1[11]));
                    packed_1_1[2] = _packed;
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
                        : "=r"(_packed) : "f"(tmem_load_2_1[12]), "f"(tmem_load_2_1[13]),
                                           "f"(tmem_load_2_1[14]), "f"(tmem_load_2_1[15]));
                    packed_1_1[3] = _packed;
                }
                int row_addr_1_1 = smem_pt_addr + (unsigned int)(pphase_1 * 16384) + (unsigned int)(my_tok_1 * 128);
                int row_rel_1_1 = pphase_1 * 16384 + my_tok_1 * 128;
                int dst_1_1 = row_addr_1_1 + ((3 ^ my_tok_1 & 7) << 4);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dst_1_1), "r"(packed_1_1[0]), "r"(packed_1_1[1]), "r"(packed_1_1[2]), "r"(packed_1_1[3]) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (tile_1 > 0) {
                    int _parity_pv_done_1 = (int)((unsigned int)tile_1 % 2) ^ 1;
                    mbarrier_wait(pv_done_addr, _parity_pv_done_1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll
                    for (int vs_3 = 0; vs_3 < 4; vs_3++) {
                        int o_a_1 = taddr + 96 + (unsigned int)(vs_3 * 96) + 32 + (unsigned int)(lane_base_1 << 16);
                        int o_b_1 = taddr + 96 + (unsigned int)(vs_3 * 96) + 32 + 16 + (unsigned int)(lane_base_1 << 16);
                        float tmem_load_3[16];
                        tmem_ld_x16(&tmem_load_3[0], o_a_1);
                        float tmem_load_1_1[16];
                        tmem_ld_x16(&tmem_load_1_1[0], o_b_1);
                        float vals_2[16];
                        #pragma unroll
                        for (int k_11 = 0; k_11 < 16; k_11 += 4) {
                            unsigned int smem_regs_11[4];
                            __int128_t _smem_raw_10;
                            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_10) : "r"(smem_stats_w_addr + (unsigned int)((32 + k_11) * 4)));
                            smem_regs_11[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[0];
                            smem_regs_11[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[1];
                            smem_regs_11[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[2];
                            smem_regs_11[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[3];
                            #pragma unroll
                            for (int e_15 = 0; e_15 < 4; e_15++) {
                                vals_2[k_11 + e_15] = __uint_as_float(smem_regs_11[e_15]);
                            }
                        }
                        float vals_1_1[16];
                        #pragma unroll
                        for (int k_12 = 0; k_12 < 16; k_12 += 4) {
                            unsigned int smem_regs_12[4];
                            __int128_t _smem_raw_11;
                            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_11) : "r"(smem_stats_w_addr + (unsigned int)((48 + k_12) * 4)));
                            smem_regs_12[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[0];
                            smem_regs_12[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[1];
                            smem_regs_12[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[2];
                            smem_regs_12[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[3];
                            #pragma unroll
                            for (int e_16 = 0; e_16 < 4; e_16++) {
                                vals_1_1[k_12 + e_16] = __uint_as_float(smem_regs_12[e_16]);
                            }
                        }
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        #pragma unroll
                        for (int j_11 = 0; j_11 < 16; j_11++) {
                            tmem_load_3[j_11] = tmem_load_3[j_11] * vals_2[j_11];
                        }
                        #pragma unroll
                        for (int j_12 = 0; j_12 < 16; j_12++) {
                            tmem_load_1_1[j_12] = tmem_load_1_1[j_12] * vals_1_1[j_12];
                        }
                        tmem_st_x16_f32(o_a_1, tmem_load_3);
                        tmem_st_x16_f32(o_b_1, tmem_load_1_1);
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(p_full_addr);
            }
            #pragma unroll
            for (int j_13 = 0; j_13 < 16; j_13++) {
                float wsum_j_2 = psum0_1[j_13];
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    wsum_j_2 += __shfl_xor_sync(0xFFFFFFFF, wsum_j_2, offset);
                if (lane == j_13 % 32) {
                    smem_stats[320 + warp_in_wg_1 * 32 + j_13] = wsum_j_2;
                }
            }
            #pragma unroll
            for (int j_14 = 0; j_14 < 16; j_14++) {
                float wsum_j_3 = psum1_1[j_14];
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    wsum_j_3 += __shfl_xor_sync(0xFFFFFFFF, wsum_j_3, offset);
                if (lane == (16 + j_14) % 32) {
                    smem_stats[320 + warp_in_wg_1 * 32 + 16 + j_14] = wsum_j_3;
                }
            }
            asm volatile("barrier.sync 2, 128;" ::: "memory");
            if (is_reducer_1 != 0) {
                float s0_1 = smem_stats[320 + (red_col_1 - 32)];
                float s1_1 = smem_stats[352 + (red_col_1 - 32)];
                float s2_1 = smem_stats[384 + (red_col_1 - 32)];
                float s3_1 = smem_stats[416 + (red_col_1 - 32)];
                float csum_1 = s0_1 + s1_1 + (s2_1 + s3_1);
                float safe_sum_1 = ((csum_1 > 0.0f) ? csum_1 : 1.0f);
                float out_scale_1 = ((num_split == 1) ? bmm2_scale : 1.0f);
                float _rcp_12_1 = approx_rcp(safe_sum_1);
                smem_stats[768 + red_col_1] = _rcp_12_1 * out_scale_1;
                if (red_col_1 < rows_valid_1) {
                    int stat_off_1 = (row_base_global_1 + red_col_1) * num_split + split_idx_1;
                    float c_fin_1 = smem_stats[576 + red_col_1];
                    float stored_max_1 = 7.807355f - c_fin_1;
                    *(reinterpret_cast<float*>(partial_max + stat_off_1) + (0)) = stored_max_1;
                    *(reinterpret_cast<float*>(partial_sum + stat_off_1) + (0)) = csum_1;
                    if (num_split == 1) {
                        float _log2_13_1;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_13_1) : "f"(csum_1));
                        float lse_row_1 = (_log2_13_1 - c_fin_1) * lse_scale;
                        *(reinterpret_cast<float*>(lse + (row_base_global_1 + red_col_1)) + (0)) = lse_row_1;
                    }
                }
            }
            asm volatile("barrier.sync 2, 128;" ::: "memory");
            unsigned int _phase_o_done_0_1 = 0;
            if (my_n_tiles_1 > 0) {
                float vals_3[32];
                #pragma unroll
                for (int k_13 = 0; k_13 < 32; k_13 += 4) {
                    unsigned int smem_regs_13[4];
                    __int128_t _smem_raw_14;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_14) : "r"(smem_stats_w_addr + (unsigned int)((800 + k_13) * 4)));
                    smem_regs_13[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[0];
                    smem_regs_13[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[1];
                    smem_regs_13[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[2];
                    smem_regs_13[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[3];
                    #pragma unroll
                    for (int e_17 = 0; e_17 < 4; e_17++) {
                        vals_3[k_13 + e_17] = __uint_as_float(smem_regs_13[e_17]);
                    }
                }
                mbarrier_wait(o_done_addr, _phase_o_done_0_1);
                _phase_o_done_0_1 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (int vs_4 = 0; vs_4 < 4; vs_4++) {
                    float tmem_load_4[16];
                    tmem_ld_x16(&tmem_load_4[0], taddr + 96 + (unsigned int)(vs_4 * 96) + 32 + (unsigned int)(lane_base_1 << 16));
                    #pragma unroll
                    for (int j_15 = 0; j_15 < 16; j_15++) {
                        int c_3 = 32 + j_15;
                        if (c_3 < rows_valid_1) {
                            *(reinterpret_cast<__nv_bfloat16*>(partial_O + (out_base_1 + c_3 * row_stride_1 + vs_4 * 128)) + (0)) = (__nv_bfloat16)(tmem_load_4[j_15] * vals_3[j_15]);
                        }
                    }
                    float tmem_load_4_1[16];
                    tmem_ld_x16(&tmem_load_4_1[0], taddr + 96 + (unsigned int)(vs_4 * 96) + 32 + 16 + (unsigned int)(lane_base_1 << 16));
                    #pragma unroll
                    for (int j_16 = 0; j_16 < 16; j_16++) {
                        int c_4 = 48 + j_16;
                        if (c_4 < rows_valid_1) {
                            *(reinterpret_cast<__nv_bfloat16*>(partial_O + (out_base_1 + c_4 * row_stride_1 + vs_4 * 128)) + (0)) = (__nv_bfloat16)(tmem_load_4_1[j_16] * vals_3[16 + j_16]);
                        }
                    }
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: softmax_wg2 ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 160;");
        { // softmax_wg2_main
            float psum0_2[16];
            float psum1_2[16];
            #pragma unroll
            for (int j_4_1 = 0; j_4_1 < 16; j_4_1++) {
                psum0_2[j_4_1] = 0.0f;
            }
            #pragma unroll
            for (int j_5_1 = 0; j_5_1 < 16; j_5_1++) {
                psum1_2[j_5_1] = 0.0f;
            }
            int split_idx_2 = blockIdx.x;
            int m_tile_2 = gridDim.y - 1 - blockIdx.y;
            int b_2 = blockIdx.z;
            int q_start_2 = cum_seq_lens_q[b_2];
            int q_len_b_2 = cum_seq_lens_q[b_2 + 1] - q_start_2;
            int kv_len_2 = seq_lens[b_2];
            int rows_b_2 = q_len_b_2 * num_heads;
            int row0_2 = m_tile_2 * 96;
            int rows_left_2 = rows_b_2 - row0_2;
            int rows_pos_2 = ((rows_left_2 < 0) ? 0 : rows_left_2);
            int rows_valid_2 = ((rows_pos_2 > 96) ? 96 : rows_pos_2);
            int row_base_global_2 = q_start_2 * num_heads + row0_2;
            int last_row_2 = row0_2 + rows_valid_2 - 1;
            int t_last_2 = last_row_2 / num_heads;
            int glen_2 = causal_seqs[b_2];
            int seen_raw_2 = glen_2 - q_len_b_2 + t_last_2 + 1;
            int seen_2 = ((seen_raw_2 < 0) ? 0 : seen_raw_2);
            int full_2 = seen_2 >> cp_shift;
            int rem_2 = seen_2 - (full_2 << cp_shift);
            int tail_raw_2 = rem_2 - cp_rank_off;
            int tail_lo_2 = ((tail_raw_2 < 0) ? 0 : tail_raw_2);
            int tail_2 = ((tail_lo_2 > cp_span) ? cp_span : tail_lo_2);
            int local_2 = (full_2 << cp_ishift) + tail_2;
            int local_c_2 = ((local_2 > kv_len_2) ? kv_len_2 : local_2);
            int bound_2 = ((causal != 0) ? local_c_2 : kv_len_2);
            int kv_end_2 = ((rows_valid_2 == 0) ? 0 : bound_2);
            int n_pages_2 = (kv_end_2 + 64 - 1) / 64;
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
            int red_col_2 = 64 + warp_in_wg_2 * 8 + lane;
            int is_reducer_2 = lane < 8;
            int row_stride_2 = num_split * 512;
            int out_base_2 = (row_base_global_2 * num_split + split_idx_2) * 512 + my_tok_2;
            if (is_reducer_2 != 0) {
                int q_tok_2 = (row0_2 + red_col_2) / num_heads;
                int glen_r_2 = causal_seqs[blockIdx.z];
                int seen_raw_2_1 = glen_r_2 - q_len_b_2 + q_tok_2 + 1;
                int seen_2_1 = ((seen_raw_2_1 < 0) ? 0 : seen_raw_2_1);
                int full_2_1 = seen_2_1 >> cp_shift;
                int rem_2_1 = seen_2_1 - (full_2_1 << cp_shift);
                int tail_raw_2_1 = rem_2_1 - cp_rank_off;
                int tail_lo_2_1 = ((tail_raw_2_1 < 0) ? 0 : tail_raw_2_1);
                int tail_2_1 = ((tail_lo_2_1 > cp_span) ? cp_span : tail_lo_2_1);
                int local_2_1 = (full_2_1 << cp_ishift) + tail_2_1;
                int local_c_2_1 = ((local_2_1 > kv_len_2) ? kv_len_2 : local_2_1);
                int bound_2_1 = ((causal != 0) ? local_c_2_1 : kv_len_2);
                smem_vis[red_col_2] = bound_2_1;
                smem_stats[576 + red_col_2] = 1031.8074f;
                smem_stats[672 + red_col_2] = -CUDART_INF_F;
            }
            asm volatile("barrier.sync 3, 128;" ::: "memory");
            int vis_min_2 = smem_vis[64];
            #pragma unroll 1
            for (int tile_2 = 0; tile_2 < my_n_tiles_2; tile_2++) {
                int pphase_2 = tile_2 & 0;
                int aphase_2 = tile_2 & 1;
                int sphase_2 = 0;
                int _parity_s_full_2 = (int)((unsigned int)tile_2 % 2);
                mbarrier_wait(s_full_addr, _parity_s_full_2);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int t_abs_2 = (my_start_2 + tile_2) * 128 + my_tok_2;
                float tmem_load_5[16];
                tmem_ld_x16(&tmem_load_5[0], taddr + (unsigned int)(sphase_2 * 96) + 64 + (unsigned int)(lane_base_2 << 16));
                float tmem_load_5_1[16];
                tmem_ld_x16(&tmem_load_5_1[0], taddr + (unsigned int)(sphase_2 * 96) + 64 + 16 + (unsigned int)(lane_base_2 << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(s_free_addr + (sphase_2) * 8);
                int tile_last_2 = (my_start_2 + tile_2) * 128 + 127;
                if (tile_last_2 >= vis_min_2) {
                    #pragma unroll
                    for (int k_14 = 0; k_14 < 16; k_14 += 4) {
                        unsigned int smem_regs_14[4];
                        __int128_t _smem_raw_0;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_0) : "r"(smem_vis_w_addr + (unsigned int)((64 + k_14) * 4)));
                        smem_regs_14[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_0)[0];
                        smem_regs_14[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_0)[1];
                        smem_regs_14[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_0)[2];
                        smem_regs_14[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_0)[3];
                        #pragma unroll
                        for (int e_18 = 0; e_18 < 4; e_18++) {
                            int vis_e_4 = 0;
                            unsigned int _bit_cast_src_1 = smem_regs_14[e_18];
                            vis_e_4 = reinterpret_cast<int*>(&_bit_cast_src_1)[0];
                            if (t_abs_2 >= vis_e_4) {
                                tmem_load_5[k_14 + e_18] = -CUDART_INF_F;
                            }
                        }
                    }
                    #pragma unroll
                    for (int k_15 = 0; k_15 < 16; k_15 += 4) {
                        unsigned int smem_regs_15[4];
                        __int128_t _smem_raw_2;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_2) : "r"(smem_vis_w_addr + (unsigned int)((80 + k_15) * 4)));
                        smem_regs_15[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[0];
                        smem_regs_15[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[1];
                        smem_regs_15[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[2];
                        smem_regs_15[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[3];
                        #pragma unroll
                        for (int e_19 = 0; e_19 < 4; e_19++) {
                            int vis_e_5 = 0;
                            unsigned int _bit_cast_src_3 = smem_regs_15[e_19];
                            vis_e_5 = reinterpret_cast<int*>(&_bit_cast_src_3)[0];
                            if (t_abs_2 >= vis_e_5) {
                                tmem_load_5_1[k_15 + e_19] = -CUDART_INF_F;
                            }
                        }
                    }
                }
                #pragma unroll
                for (int j_17 = 0; j_17 < 16; j_17++) {
                    float m_j_4;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xFFFFFFFF;" : "=f"(m_j_4) : "f"(tmem_load_5[j_17]));
                    if (lane == j_17 % 32) {
                        smem_stats[448 + warp_in_wg_2 * 32 + j_17] = m_j_4;
                    }
                }
                #pragma unroll
                for (int j_18 = 0; j_18 < 16; j_18++) {
                    float m_j_5;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xFFFFFFFF;" : "=f"(m_j_5) : "f"(tmem_load_5_1[j_18]));
                    if (lane == (16 + j_18) % 32) {
                        smem_stats[448 + warp_in_wg_2 * 32 + 16 + j_18] = m_j_5;
                    }
                }
                asm volatile("barrier.sync 3, 128;" ::: "memory");
                if (is_reducer_2 != 0) {
                    float r0_2 = smem_stats[448 + (red_col_2 - 64)];
                    float r1_2 = smem_stats[480 + (red_col_2 - 64)];
                    float r2_2 = smem_stats[512 + (red_col_2 - 64)];
                    float r3_2 = smem_stats[544 + (red_col_2 - 64)];
                    float tr_2;
                    asm("max.f32 %0, %1, %2;" : "=f"(tr_2) : "f"(max_noftz(r0_2, r1_2)), "f"(max_noftz(r2_2, r3_2)));
                    float cr_old_2 = smem_stats[576 + red_col_2];
                    float cr_upd_2 = 7.807355f - tr_2 * softmax_scale_log2;
                    float run_old_r_2 = smem_stats[672 + red_col_2];
                    float t_tile_r_2 = tr_2 * softmax_scale_log2;
                    float run_grown_r_2;
                    asm("max.f32 %0, %1, %2;" : "=f"(run_grown_r_2) : "f"(run_old_r_2), "f"(t_tile_r_2));
                    float run_new_r_2 = ((tr_2 > -CUDART_INF_F) ? run_grown_r_2 : run_old_r_2);
                    float cr_floor_2 = 7.807355f - run_new_r_2 + 32.0f;
                    float cr_bounded_2 = ((cr_upd_2 <= cr_floor_2) ? cr_upd_2 : cr_floor_2);
                    float cr_new_2 = ((tr_2 > -CUDART_INF_F) ? cr_bounded_2 : cr_old_2);
                    float alpha_r_2 = approx_exp2(cr_new_2 - cr_old_2);
                    smem_stats[576 + red_col_2] = cr_new_2;
                    smem_stats[672 + red_col_2] = run_new_r_2;
                    smem_stats[red_col_2] = alpha_r_2;
                }
                asm volatile("barrier.sync 3, 128;" ::: "memory");
                if (tile_2 >= 1) {
                    int _parity_pt_free_2 = (int)((unsigned int)tile_2 % 2) ^ 1;
                    mbarrier_wait(pt_free_addr, _parity_pt_free_2);
                }
                unsigned int packed_2[4];
                #pragma unroll
                for (int k_16 = 0; k_16 < 16; k_16 += 4) {
                    unsigned int smem_regs_16[4];
                    __int128_t _smem_raw_4;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_4) : "r"(smem_stats_w_addr + (unsigned int)((640 + k_16) * 4)));
                    smem_regs_16[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[0];
                    smem_regs_16[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[1];
                    smem_regs_16[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[2];
                    smem_regs_16[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[3];
                    unsigned int smem_regs_4_1[4];
                    __int128_t _smem_raw_5;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_5) : "r"(smem_stats_w_addr + (unsigned int)((64 + k_16) * 4)));
                    smem_regs_4_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[0];
                    smem_regs_4_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[1];
                    smem_regs_4_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[2];
                    smem_regs_4_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[3];
                    #pragma unroll
                    for (int e_20 = 0; e_20 < 4; e_20++) {
                        float _exp2_6_2 = approx_exp2(tmem_load_5[k_16 + e_20] * softmax_scale_log2 + __uint_as_float(smem_regs_16[e_20]));
                        tmem_load_5[k_16 + e_20] = _exp2_6_2;
                    }
                    #pragma unroll
                    for (int e_21 = 0; e_21 < 4; e_21++) {
                        psum0_2[k_16 + e_21] = psum0_2[k_16 + e_21] * __uint_as_float(smem_regs_4_1[e_21]) + tmem_load_5[k_16 + e_21];
                    }
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
                        : "=r"(_packed) : "f"(tmem_load_5[0]), "f"(tmem_load_5[1]),
                                           "f"(tmem_load_5[2]), "f"(tmem_load_5[3]));
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
                        : "=r"(_packed) : "f"(tmem_load_5[4]), "f"(tmem_load_5[5]),
                                           "f"(tmem_load_5[6]), "f"(tmem_load_5[7]));
                    packed_2[1] = _packed;
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
                        : "=r"(_packed) : "f"(tmem_load_5[8]), "f"(tmem_load_5[9]),
                                           "f"(tmem_load_5[10]), "f"(tmem_load_5[11]));
                    packed_2[2] = _packed;
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
                        : "=r"(_packed) : "f"(tmem_load_5[12]), "f"(tmem_load_5[13]),
                                           "f"(tmem_load_5[14]), "f"(tmem_load_5[15]));
                    packed_2[3] = _packed;
                }
                int row_addr_2 = smem_pt_addr + (unsigned int)(pphase_2 * 16384) + (unsigned int)(my_tok_2 * 128);
                int row_rel_2 = pphase_2 * 16384 + my_tok_2 * 128;
                int dst_2 = row_addr_2 + ((4 ^ my_tok_2 & 7) << 4);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dst_2), "r"(packed_2[0]), "r"(packed_2[1]), "r"(packed_2[2]), "r"(packed_2[3]) : "memory");
                unsigned int packed_2_1[4];
                #pragma unroll
                for (int k_17 = 0; k_17 < 16; k_17 += 4) {
                    unsigned int smem_regs_17[4];
                    __int128_t _smem_raw_7;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_7) : "r"(smem_stats_w_addr + (unsigned int)((656 + k_17) * 4)));
                    smem_regs_17[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[0];
                    smem_regs_17[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[1];
                    smem_regs_17[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[2];
                    smem_regs_17[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[3];
                    unsigned int smem_regs_5_1[4];
                    __int128_t _smem_raw_8;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_8) : "r"(smem_stats_w_addr + (unsigned int)((80 + k_17) * 4)));
                    smem_regs_5_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[0];
                    smem_regs_5_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[1];
                    smem_regs_5_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[2];
                    smem_regs_5_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[3];
                    #pragma unroll
                    for (int e_22 = 0; e_22 < 4; e_22++) {
                        float _exp2_9_2 = approx_exp2(tmem_load_5_1[k_17 + e_22] * softmax_scale_log2 + __uint_as_float(smem_regs_17[e_22]));
                        tmem_load_5_1[k_17 + e_22] = _exp2_9_2;
                    }
                    #pragma unroll
                    for (int e_23 = 0; e_23 < 4; e_23++) {
                        psum1_2[k_17 + e_23] = psum1_2[k_17 + e_23] * __uint_as_float(smem_regs_5_1[e_23]) + tmem_load_5_1[k_17 + e_23];
                    }
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
                        : "=r"(_packed) : "f"(tmem_load_5_1[0]), "f"(tmem_load_5_1[1]),
                                           "f"(tmem_load_5_1[2]), "f"(tmem_load_5_1[3]));
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
                        : "=r"(_packed) : "f"(tmem_load_5_1[4]), "f"(tmem_load_5_1[5]),
                                           "f"(tmem_load_5_1[6]), "f"(tmem_load_5_1[7]));
                    packed_2_1[1] = _packed;
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
                        : "=r"(_packed) : "f"(tmem_load_5_1[8]), "f"(tmem_load_5_1[9]),
                                           "f"(tmem_load_5_1[10]), "f"(tmem_load_5_1[11]));
                    packed_2_1[2] = _packed;
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
                        : "=r"(_packed) : "f"(tmem_load_5_1[12]), "f"(tmem_load_5_1[13]),
                                           "f"(tmem_load_5_1[14]), "f"(tmem_load_5_1[15]));
                    packed_2_1[3] = _packed;
                }
                int row_addr_2_1 = smem_pt_addr + (unsigned int)(pphase_2 * 16384) + (unsigned int)(my_tok_2 * 128);
                int row_rel_2_1 = pphase_2 * 16384 + my_tok_2 * 128;
                int dst_2_1 = row_addr_2_1 + ((5 ^ my_tok_2 & 7) << 4);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dst_2_1), "r"(packed_2_1[0]), "r"(packed_2_1[1]), "r"(packed_2_1[2]), "r"(packed_2_1[3]) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (tile_2 > 0) {
                    int _parity_pv_done_2 = (int)((unsigned int)tile_2 % 2) ^ 1;
                    mbarrier_wait(pv_done_addr, _parity_pv_done_2);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll
                    for (int vs_5 = 0; vs_5 < 4; vs_5++) {
                        int o_a_2 = taddr + 96 + (unsigned int)(vs_5 * 96) + 64 + (unsigned int)(lane_base_2 << 16);
                        int o_b_2 = taddr + 96 + (unsigned int)(vs_5 * 96) + 64 + 16 + (unsigned int)(lane_base_2 << 16);
                        float tmem_load_6[16];
                        tmem_ld_x16(&tmem_load_6[0], o_a_2);
                        float tmem_load_1_2[16];
                        tmem_ld_x16(&tmem_load_1_2[0], o_b_2);
                        float vals_4[16];
                        #pragma unroll
                        for (int k_18 = 0; k_18 < 16; k_18 += 4) {
                            unsigned int smem_regs_18[4];
                            __int128_t _smem_raw_10;
                            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_10) : "r"(smem_stats_w_addr + (unsigned int)((64 + k_18) * 4)));
                            smem_regs_18[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[0];
                            smem_regs_18[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[1];
                            smem_regs_18[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[2];
                            smem_regs_18[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[3];
                            #pragma unroll
                            for (int e_24 = 0; e_24 < 4; e_24++) {
                                vals_4[k_18 + e_24] = __uint_as_float(smem_regs_18[e_24]);
                            }
                        }
                        float vals_2_1[16];
                        #pragma unroll
                        for (int k_19 = 0; k_19 < 16; k_19 += 4) {
                            unsigned int smem_regs_19[4];
                            __int128_t _smem_raw_11;
                            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_11) : "r"(smem_stats_w_addr + (unsigned int)((80 + k_19) * 4)));
                            smem_regs_19[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[0];
                            smem_regs_19[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[1];
                            smem_regs_19[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[2];
                            smem_regs_19[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[3];
                            #pragma unroll
                            for (int e_25 = 0; e_25 < 4; e_25++) {
                                vals_2_1[k_19 + e_25] = __uint_as_float(smem_regs_19[e_25]);
                            }
                        }
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        #pragma unroll
                        for (int j_19 = 0; j_19 < 16; j_19++) {
                            tmem_load_6[j_19] = tmem_load_6[j_19] * vals_4[j_19];
                        }
                        #pragma unroll
                        for (int j_20 = 0; j_20 < 16; j_20++) {
                            tmem_load_1_2[j_20] = tmem_load_1_2[j_20] * vals_2_1[j_20];
                        }
                        tmem_st_x16_f32(o_a_2, tmem_load_6);
                        tmem_st_x16_f32(o_b_2, tmem_load_1_2);
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(p_full_addr);
            }
            #pragma unroll
            for (int j_21 = 0; j_21 < 16; j_21++) {
                float wsum_j_4 = psum0_2[j_21];
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    wsum_j_4 += __shfl_xor_sync(0xFFFFFFFF, wsum_j_4, offset);
                if (lane == j_21 % 32) {
                    smem_stats[448 + warp_in_wg_2 * 32 + j_21] = wsum_j_4;
                }
            }
            #pragma unroll
            for (int j_22 = 0; j_22 < 16; j_22++) {
                float wsum_j_5 = psum1_2[j_22];
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    wsum_j_5 += __shfl_xor_sync(0xFFFFFFFF, wsum_j_5, offset);
                if (lane == (16 + j_22) % 32) {
                    smem_stats[448 + warp_in_wg_2 * 32 + 16 + j_22] = wsum_j_5;
                }
            }
            asm volatile("barrier.sync 3, 128;" ::: "memory");
            if (is_reducer_2 != 0) {
                float s0_2 = smem_stats[448 + (red_col_2 - 64)];
                float s1_2 = smem_stats[480 + (red_col_2 - 64)];
                float s2_2 = smem_stats[512 + (red_col_2 - 64)];
                float s3_2 = smem_stats[544 + (red_col_2 - 64)];
                float csum_2 = s0_2 + s1_2 + (s2_2 + s3_2);
                float safe_sum_2 = ((csum_2 > 0.0f) ? csum_2 : 1.0f);
                float out_scale_2 = ((num_split == 1) ? bmm2_scale : 1.0f);
                float _rcp_12_2 = approx_rcp(safe_sum_2);
                smem_stats[768 + red_col_2] = _rcp_12_2 * out_scale_2;
                if (red_col_2 < rows_valid_2) {
                    int stat_off_2 = (row_base_global_2 + red_col_2) * num_split + split_idx_2;
                    float c_fin_2 = smem_stats[576 + red_col_2];
                    float stored_max_2 = 7.807355f - c_fin_2;
                    *(reinterpret_cast<float*>(partial_max + stat_off_2) + (0)) = stored_max_2;
                    *(reinterpret_cast<float*>(partial_sum + stat_off_2) + (0)) = csum_2;
                    if (num_split == 1) {
                        float _log2_13_2;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_13_2) : "f"(csum_2));
                        float lse_row_2 = (_log2_13_2 - c_fin_2) * lse_scale;
                        *(reinterpret_cast<float*>(lse + (row_base_global_2 + red_col_2)) + (0)) = lse_row_2;
                    }
                }
            }
            asm volatile("barrier.sync 3, 128;" ::: "memory");
            unsigned int _phase_o_done_0_2 = 0;
            if (my_n_tiles_2 > 0) {
                float vals_5[32];
                #pragma unroll
                for (int k_20 = 0; k_20 < 32; k_20 += 4) {
                    unsigned int smem_regs_20[4];
                    __int128_t _smem_raw_14;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_14) : "r"(smem_stats_w_addr + (unsigned int)((832 + k_20) * 4)));
                    smem_regs_20[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[0];
                    smem_regs_20[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[1];
                    smem_regs_20[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[2];
                    smem_regs_20[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[3];
                    #pragma unroll
                    for (int e_26 = 0; e_26 < 4; e_26++) {
                        vals_5[k_20 + e_26] = __uint_as_float(smem_regs_20[e_26]);
                    }
                }
                mbarrier_wait(o_done_addr, _phase_o_done_0_2);
                _phase_o_done_0_2 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (int vs_6 = 0; vs_6 < 4; vs_6++) {
                    float tmem_load_7[16];
                    tmem_ld_x16(&tmem_load_7[0], taddr + 96 + (unsigned int)(vs_6 * 96) + 64 + (unsigned int)(lane_base_2 << 16));
                    #pragma unroll
                    for (int j_23 = 0; j_23 < 16; j_23++) {
                        int c_5 = 64 + j_23;
                        if (c_5 < rows_valid_2) {
                            *(reinterpret_cast<__nv_bfloat16*>(partial_O + (out_base_2 + c_5 * row_stride_2 + vs_6 * 128)) + (0)) = (__nv_bfloat16)(tmem_load_7[j_23] * vals_5[j_23]);
                        }
                    }
                    float tmem_load_7_1[16];
                    tmem_ld_x16(&tmem_load_7_1[0], taddr + 96 + (unsigned int)(vs_6 * 96) + 64 + 16 + (unsigned int)(lane_base_2 << 16));
                    #pragma unroll
                    for (int j_24 = 0; j_24 < 16; j_24++) {
                        int c_6 = 80 + j_24;
                        if (c_6 < rows_valid_2) {
                            *(reinterpret_cast<__nv_bfloat16*>(partial_O + (out_base_2 + c_6 * row_stride_2 + vs_6 * 128)) + (0)) = (__nv_bfloat16)(tmem_load_7_1[j_24] * vals_5[16 + j_24]);
                        }
                    }
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 12) {
        { // mma_warp_main
            int split_idx_3 = blockIdx.x;
            int m_tile_3 = gridDim.y - 1 - blockIdx.y;
            int b_3 = blockIdx.z;
            int q_start_3 = cum_seq_lens_q[b_3];
            int q_len_b_3 = cum_seq_lens_q[b_3 + 1] - q_start_3;
            int kv_len_3 = seq_lens[b_3];
            int rows_b_3 = q_len_b_3 * num_heads;
            int row0_3 = m_tile_3 * 96;
            int rows_left_3 = rows_b_3 - row0_3;
            int rows_pos_3 = ((rows_left_3 < 0) ? 0 : rows_left_3);
            int rows_valid_3 = ((rows_pos_3 > 96) ? 96 : rows_pos_3);
            int row_base_global_3 = q_start_3 * num_heads + row0_3;
            int last_row_3 = row0_3 + rows_valid_3 - 1;
            int t_last_3 = last_row_3 / num_heads;
            int glen_3 = causal_seqs[b_3];
            int seen_raw_3 = glen_3 - q_len_b_3 + t_last_3 + 1;
            int seen_3 = ((seen_raw_3 < 0) ? 0 : seen_raw_3);
            int full_3 = seen_3 >> cp_shift;
            int rem_3 = seen_3 - (full_3 << cp_shift);
            int tail_raw_3 = rem_3 - cp_rank_off;
            int tail_lo_3 = ((tail_raw_3 < 0) ? 0 : tail_raw_3);
            int tail_3 = ((tail_lo_3 > cp_span) ? cp_span : tail_lo_3);
            int local_3 = (full_3 << cp_ishift) + tail_3;
            int local_c_3 = ((local_3 > kv_len_3) ? kv_len_3 : local_3);
            int bound_3 = ((causal != 0) ? local_c_3 : kv_len_3);
            int kv_end_3 = ((rows_valid_3 == 0) ? 0 : bound_3);
            int n_pages_3 = (kv_end_3 + 64 - 1) / 64;
            int n_tiles_total_3 = (kv_end_3 + 128 - 1) / 128;
            int tiles_per_split_3 = (n_tiles_total_3 + num_split - 1) / num_split;
            int my_start_3 = split_idx_3 * tiles_per_split_3;
            int my_end_raw_3 = my_start_3 + tiles_per_split_3;
            int my_end_3 = ((my_end_raw_3 > n_tiles_total_3) ? n_tiles_total_3 : my_end_raw_3);
            int my_n_raw_3 = my_end_3 - my_start_3;
            int my_n_tiles_3 = ((my_n_raw_3 < 0) ? 0 : my_n_raw_3);
            int pt_base_3 = b_3 * max_pages_per_seq;
            unsigned int mma_stage = 0;
            unsigned int _phase_q_full_0 = 0;
            mbarrier_wait(q_full_addr, _phase_q_full_0);
            _phase_q_full_0 ^= 1;
            unsigned int _phase_kv_full = 0;
            unsigned int _phase_rope_full = 0;
            #pragma unroll 1
            for (int tile_3 = 0; tile_3 < my_n_tiles_3; tile_3++) {
                int sphase_3 = 0;
                if (tile_3 >= 1) {
                    int _parity_s_free = (int)((unsigned int)tile_3 % 2) ^ 1;
                    mbarrier_wait(s_free_addr, _parity_s_free);
                }
                #pragma unroll
                for (int st = 0; st < 4; st++) {
                    mbarrier_wait(kv_full_addr + (mma_stage) * 8, _phase_kv_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_0 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (mma_stage) * 1024);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (st) * 768);
                    {
                        uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                        uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem + (sphase_3 * 96)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135790608, ((st == 0) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem + (sphase_3 * 96)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135790608, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem + (sphase_3 * 96)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135790608, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem + (sphase_3 * 96)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135790608, 1);
                        }
                    }
                    mma_stage += 1;
                    if (mma_stage == 9) { mma_stage = 0; _phase_kv_full ^= 1; }
                }
                mbarrier_wait(rope_full_addr, _phase_rope_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_1 = make_warp_uniform((((smem_kr_addr) >> 4) & 0x3FFF) + (0) * 512);
                int _mma_b_lo_1 = make_warp_uniform(((smem_qr_addr) >> 4) & 0x3FFF);
                {
                    uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x80004020U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                    uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x80004020U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (sphase_3 * 96)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135790608, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (sphase_3 * 96)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135790608, 1);
                    }
                }
                elect_commit(s_full_addr);
                elect_commit(rope_empty_addr);
                _phase_rope_full ^= 1;
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_dealloc_addr);
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait(tmem_dealloc_addr, _phase_tmem_dealloc_0);
            _phase_tmem_dealloc_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
        }
    }
    // ---- Role: pv_warp ----
    if (warp == 13) {
        { // pv_warp_main
            int split_idx_4 = blockIdx.x;
            int m_tile_4 = gridDim.y - 1 - blockIdx.y;
            int b_4 = blockIdx.z;
            int q_start_4 = cum_seq_lens_q[b_4];
            int q_len_b_4 = cum_seq_lens_q[b_4 + 1] - q_start_4;
            int kv_len_4 = seq_lens[b_4];
            int rows_b_4 = q_len_b_4 * num_heads;
            int row0_4 = m_tile_4 * 96;
            int rows_left_4 = rows_b_4 - row0_4;
            int rows_pos_4 = ((rows_left_4 < 0) ? 0 : rows_left_4);
            int rows_valid_4 = ((rows_pos_4 > 96) ? 96 : rows_pos_4);
            int row_base_global_4 = q_start_4 * num_heads + row0_4;
            int last_row_4 = row0_4 + rows_valid_4 - 1;
            int t_last_4 = last_row_4 / num_heads;
            int glen_4 = causal_seqs[b_4];
            int seen_raw_4 = glen_4 - q_len_b_4 + t_last_4 + 1;
            int seen_4 = ((seen_raw_4 < 0) ? 0 : seen_raw_4);
            int full_4 = seen_4 >> cp_shift;
            int rem_4 = seen_4 - (full_4 << cp_shift);
            int tail_raw_4 = rem_4 - cp_rank_off;
            int tail_lo_4 = ((tail_raw_4 < 0) ? 0 : tail_raw_4);
            int tail_4 = ((tail_lo_4 > cp_span) ? cp_span : tail_lo_4);
            int local_4 = (full_4 << cp_ishift) + tail_4;
            int local_c_4 = ((local_4 > kv_len_4) ? kv_len_4 : local_4);
            int bound_4 = ((causal != 0) ? local_c_4 : kv_len_4);
            int kv_end_4 = ((rows_valid_4 == 0) ? 0 : bound_4);
            int n_pages_4 = (kv_end_4 + 64 - 1) / 64;
            int n_tiles_total_4 = (kv_end_4 + 128 - 1) / 128;
            int tiles_per_split_4 = (n_tiles_total_4 + num_split - 1) / num_split;
            int my_start_4 = split_idx_4 * tiles_per_split_4;
            int my_end_raw_4 = my_start_4 + tiles_per_split_4;
            int my_end_4 = ((my_end_raw_4 > n_tiles_total_4) ? n_tiles_total_4 : my_end_raw_4);
            int my_n_raw_4 = my_end_4 - my_start_4;
            int my_n_tiles_4 = ((my_n_raw_4 < 0) ? 0 : my_n_raw_4);
            int pt_base_4 = b_4 * max_pages_per_seq;
            int first_pv = 1;
            unsigned int pv_stage = 0;
            #pragma unroll 1
            for (int tile_4 = 0; tile_4 < my_n_tiles_4; tile_4++) {
                int pv_pp = 0;
                int _parity_p_full = (int)((unsigned int)tile_4 % 2);
                mbarrier_wait(p_full_addr, _parity_p_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (int vs_7 = 0; vs_7 < 4; vs_7++) {
                    int _mma_a_lo_2 = make_warp_uniform(((((smem_vt_addr) >> 4) & 0x3FFF) | 0x4000000) + (pv_stage) * 1024);
                    int _mma_b_lo_2 = make_warp_uniform(((((smem_pt_addr) >> 4) & 0x3FFF) | 0x4000000) + (pv_pp) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                        uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem + (96 + vs_7 * 96)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135888912, ((first_pv) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 256U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 256U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem + (96 + vs_7 * 96)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135888912, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 256U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 256U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem + (96 + vs_7 * 96)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135888912, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 256U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 256U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem + (96 + vs_7 * 96)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135888912, 1);
                        }
                    }
                    elect_commit(kv_empty_addr + (pv_stage) * 8);
                    pv_stage += 1;
                    if (pv_stage == 9) { pv_stage = 0; }
                }
                elect_commit(pt_free_addr);
                elect_commit(pv_done_addr);
                first_pv = 0;
            }
            if (my_n_tiles_4 > 0) {
                elect_commit(o_done_addr);
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: load_warp ----
    if (warp == 14) {
        { // load_warp_main
            int split_idx_5 = blockIdx.x;
            int m_tile_5 = gridDim.y - 1 - blockIdx.y;
            int b_5 = blockIdx.z;
            int q_start_5 = cum_seq_lens_q[b_5];
            int q_len_b_5 = cum_seq_lens_q[b_5 + 1] - q_start_5;
            int kv_len_5 = seq_lens[b_5];
            int rows_b_5 = q_len_b_5 * num_heads;
            int row0_5 = m_tile_5 * 96;
            int rows_left_5 = rows_b_5 - row0_5;
            int rows_pos_5 = ((rows_left_5 < 0) ? 0 : rows_left_5);
            int rows_valid_5 = ((rows_pos_5 > 96) ? 96 : rows_pos_5);
            int row_base_global_5 = q_start_5 * num_heads + row0_5;
            int last_row_5 = row0_5 + rows_valid_5 - 1;
            int t_last_5 = last_row_5 / num_heads;
            int glen_5 = causal_seqs[b_5];
            int seen_raw_5 = glen_5 - q_len_b_5 + t_last_5 + 1;
            int seen_5 = ((seen_raw_5 < 0) ? 0 : seen_raw_5);
            int full_5 = seen_5 >> cp_shift;
            int rem_5 = seen_5 - (full_5 << cp_shift);
            int tail_raw_5 = rem_5 - cp_rank_off;
            int tail_lo_5 = ((tail_raw_5 < 0) ? 0 : tail_raw_5);
            int tail_5 = ((tail_lo_5 > cp_span) ? cp_span : tail_lo_5);
            int local_5 = (full_5 << cp_ishift) + tail_5;
            int local_c_5 = ((local_5 > kv_len_5) ? kv_len_5 : local_5);
            int bound_5 = ((causal != 0) ? local_c_5 : kv_len_5);
            int kv_end_5 = ((rows_valid_5 == 0) ? 0 : bound_5);
            int n_pages_5 = (kv_end_5 + 64 - 1) / 64;
            int n_tiles_total_5 = (kv_end_5 + 128 - 1) / 128;
            int tiles_per_split_5 = (n_tiles_total_5 + num_split - 1) / num_split;
            int my_start_5 = split_idx_5 * tiles_per_split_5;
            int my_end_raw_5 = my_start_5 + tiles_per_split_5;
            int my_end_5 = ((my_end_raw_5 > n_tiles_total_5) ? n_tiles_total_5 : my_end_raw_5);
            int my_n_raw_5 = my_end_5 - my_start_5;
            int my_n_tiles_5 = ((my_n_raw_5 < 0) ? 0 : my_n_raw_5);
            int pt_base_5 = b_5 * max_pages_per_seq;
            unsigned int load_stage = 0;
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(q_full_addr, 55296);
                #pragma unroll
                for (int st_1 = 0; st_1 < 4; st_1++) {
                    tma_2d_gmem2smem(smem_q_addr + (unsigned int)(st_1 * 12288), (&tmap_q), st_1 * 128, row_base_global_5, q_full_addr);
                }
                tma_2d_gmem2smem(smem_qr_addr, (&tmap_qr), 512, row_base_global_5, q_full_addr);
            }
            unsigned int _phase_kv_empty = 1;
            unsigned int _phase_rope_empty = 1;
            #pragma unroll 1
            for (int tile_5 = 0; tile_5 < my_n_tiles_5; tile_5++) {
                int p0 = (my_start_5 + tile_5) * 2;
                int p1_raw = p0 + 1;
                int p1 = ((p1_raw >= n_pages_5) ? p0 : p1_raw);
                int g0 = page_table[pt_base_5 + p0];
                int g1 = page_table[pt_base_5 + p1];
                #pragma unroll
                for (int st_2 = 0; st_2 < 4; st_2++) {
                    mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 16384);
                        tma_2d_gmem2smem(smem_k_addr + load_stage * 16384, (&tmap_k), st_2 * 128, g0 * 64, kv_full_addr + (load_stage) * 8);
                        tma_2d_gmem2smem(smem_k_addr + load_stage * 16384 + 8192, (&tmap_k), st_2 * 128, g1 * 64, kv_full_addr + (load_stage) * 8);
                    }
                    load_stage += 1;
                    if (load_stage == 9) { load_stage = 0; _phase_kv_empty ^= 1; }
                }
                mbarrier_wait(rope_empty_addr, _phase_rope_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(rope_full_addr, 8192);
                    tma_2d_gmem2smem(smem_kr_addr + 0 * 8192, (&tmap_kr), 512, g0 * 64, rope_full_addr);
                    tma_2d_gmem2smem(smem_kr_addr + 0 * 8192 + 4096, (&tmap_kr), 512, g1 * 64, rope_full_addr);
                }
                _phase_rope_empty ^= 1;
                int pf_tile = tile_5;
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: idle ----
    if (warp == 15) {
        // idle — no tasks assigned
    }

    // Cleanup
}

} // extern "C"
