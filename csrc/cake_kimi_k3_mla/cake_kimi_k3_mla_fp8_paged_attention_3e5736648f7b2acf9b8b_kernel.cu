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


__device__ __forceinline__ void tmem_ld_x4(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x4.b32"
        " {%0, %1, %2, %3}, [%4];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])
        : "r"(tmem_addr));
}


__device__ __forceinline__ void tmem_st_x4_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x4.b32"
        " [%0], {%1, %2, %3, %4};"
        :: "r"(tmem_addr),
           "f"(src[0]), "f"(src[1]), "f"(src[2]), "f"(src[3]));
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

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_kimi_k3_mla_fp8_paged_attention_3e5736648f7b2acf9b8b(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_qr, const __grid_constant__ CUtensorMap tmap_k, const __grid_constant__ CUtensorMap tmap_kr, __nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_max, float* __restrict__ partial_sum, int* __restrict__ seq_lens, int* __restrict__ cum_seq_lens_q, int* __restrict__ page_table, float softmax_scale_log2, float bmm2_scale, int num_heads, int num_split, int max_pages_per_seq, float* __restrict__ lse, int* __restrict__ causal_seqs, float lse_scale, int causal, int cp_shift, int cp_ishift, int cp_rank_off, int cp_span)
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
    #define kv_empty_addr (mbar_base + 88)
    #define rope_full_addr (mbar_base + 168)
    #define rope_empty_addr (mbar_base + 184)
    #define s_full_addr (mbar_base + 200)
    #define s_free_addr (mbar_base + 216)
    #define p_full_addr (mbar_base + 232)
    #define pt_free_addr (mbar_base + 248)
    #define pv_done_addr (mbar_base + 264)
    #define o_done_addr (mbar_base + 272)
    #define tmem_dealloc_addr (mbar_base + 280)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    uint8_t* smem_q = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    uint8_t* smem_qr = reinterpret_cast<uint8_t*>(smem_raw + 9216);
    const int smem_qr_addr = smem + 9216;
    uint8_t* smem_k = reinterpret_cast<uint8_t*>(smem_raw + 10240);
    const int smem_k_addr = smem + 10240;
    uint8_t* smem_vt = reinterpret_cast<uint8_t*>(smem_raw + 10240);
    const int smem_vt_addr = smem + 10240;
    uint8_t* smem_kr = reinterpret_cast<uint8_t*>(smem_raw + 174080);
    const int smem_kr_addr = smem + 174080;
    uint8_t* smem_pt = reinterpret_cast<uint8_t*>(smem_raw + 190464);
    const int smem_pt_addr = smem + 190464;
    unsigned int* smem_pt_words = reinterpret_cast<unsigned int*>(smem_raw + 190464);
    const int smem_pt_words_addr = smem + 190464;
    float* smem_stats = reinterpret_cast<float*>(smem_raw + 223232);
    const int smem_stats_addr = smem + 223232;
    int* smem_vis = reinterpret_cast<int*>(smem_raw + 223808);
    const int smem_vis_addr = smem + 223808;
    int* smem_flags = reinterpret_cast<int*>(smem_raw + 223872);
    const int smem_flags_addr = smem + 223872;
    unsigned int* smem_stats_w = reinterpret_cast<unsigned int*>(smem_raw + 223232);
    const int smem_stats_w_addr = smem + 223232;
    unsigned int* smem_vis_w = reinterpret_cast<unsigned int*>(smem_raw + 223808);
    const int smem_vis_w_addr = smem + 223808;
    unsigned int* smem_flags_w = reinterpret_cast<unsigned int*>(smem_raw + 223872);
    const int smem_flags_w_addr = smem + 223872;

    // Mbarrier init (36 barriers at smem_raw[0..288))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // --- pipeline 'kv_pipe' ---
            // kv_full: 10 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            // kv_empty: 10 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            // --- pipeline 'rope_pipe' ---
            // rope_full: 2 barriers, init_count=1
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            // rope_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            // s_free: 2 barriers, init_count=256
            mbarrier_init(smem + 216, 256);
            mbarrier_init(smem + 224, 256);
            // p_full: 2 barriers, init_count=256
            mbarrier_init(smem + 232, 256);
            mbarrier_init(smem + 240, 256);
            // pt_free: 2 barriers, init_count=1
            mbarrier_init(smem + 248, 1);
            mbarrier_init(smem + 256, 1);
            // pv_done: 1 barriers, init_count=1
            mbarrier_init(smem + 264, 1);
            // o_done: 1 barriers, init_count=1
            mbarrier_init(smem + 272, 1);
            // tmem_dealloc: 1 barriers, init_count=352
            mbarrier_init(smem + 280, 352);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 192 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 288);
    if (warp == 0) {
        int _tmem_hold = smem + 288;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_tmem = taddr;

    // Kernel post-init ops
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    }

    // ---- Role: softmax_wg0 ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 184;");
        { // softmax_wg0_main
            float psum0_0[4];
            float psum1_0[4];
            #pragma unroll
            for (int j_0 = 0; j_0 < 4; j_0++) {
                psum0_0[j_0] = 0.0f;
            }
            #pragma unroll
            for (int j_1 = 0; j_1 < 4; j_1++) {
                psum1_0[j_1] = 0.0f;
            }
            int split_idx = blockIdx.x;
            int m_tile = gridDim.y - 1 - blockIdx.y;
            int b = blockIdx.z;
            int q_start = cum_seq_lens_q[b];
            int q_len_b = cum_seq_lens_q[b + 1] - q_start;
            int kv_len = seq_lens[b];
            int rows_b = q_len_b * num_heads;
            int row0 = m_tile * 16;
            int rows_left = rows_b - row0;
            int rows_pos = ((rows_left < 0) ? 0 : rows_left);
            int rows_valid = ((rows_pos > 16) ? 16 : rows_pos);
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
            int red_col = warp_in_wg * 2 + lane;
            int is_reducer = lane < 2;
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
                smem_stats[96 + red_col] = 1031.8074f;
                smem_stats[112 + red_col] = -CUDART_INF_F;
            }
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            int vis_min = smem_vis[0];
            #pragma unroll 1
            for (int tile = 0; tile < my_n_tiles; tile++) {
                int pphase = tile & 1;
                int aphase = tile & 1;
                int sphase = tile % 2;
                int _stage_s_full = (int)((unsigned int)tile % 2);
                int _parity_s_full = (int)((unsigned int)tile / 2 % 2);
                mbarrier_wait(s_full_addr + (_stage_s_full) * 8, _parity_s_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int t_abs = (my_start + tile) * 128 + my_tok;
                float tmem_load[4];
                tmem_ld_x4(&tmem_load[0], taddr + (unsigned int)(sphase * 32) + (unsigned int)(lane_base << 16));
                float tmem_load_0[4];
                tmem_ld_x4(&tmem_load_0[0], taddr + (unsigned int)(sphase * 32) + 4 + (unsigned int)(lane_base << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(s_free_addr + ((int)((unsigned int)tile % 2)) * 8);
                int tile_last = (my_start + tile) * 128 + 127;
                if (tile_last >= vis_min) {
                    #pragma unroll
                    for (int k = 0; k < 4; k += 4) {
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
                    for (int k_1 = 0; k_1 < 4; k_1 += 4) {
                        unsigned int smem_regs_1[4];
                        __int128_t _smem_raw_2;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_2) : "r"(smem_vis_w_addr + (unsigned int)((4 + k_1) * 4)));
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
                float pm = -CUDART_INF_F;
                #pragma unroll
                for (int k_2 = 0; k_2 < 4; k_2 += 4) {
                    unsigned int smem_regs_2[4];
                    __int128_t _smem_raw_4;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_4) : "r"(smem_stats_w_addr + (unsigned int)((96 + k_2) * 4)));
                    smem_regs_2[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[0];
                    smem_regs_2[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[1];
                    smem_regs_2[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[2];
                    smem_regs_2[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[3];
                    #pragma unroll
                    for (int e_2 = 0; e_2 < 4; e_2++) {
                        tmem_load[k_2 + e_2] = tmem_load[k_2 + e_2] * softmax_scale_log2 + __uint_as_float(smem_regs_2[e_2]);
                        pm = max_noftz(pm, tmem_load[k_2 + e_2]);
                    }
                }
                float pm_2 = -CUDART_INF_F;
                #pragma unroll
                for (int k_3 = 0; k_3 < 4; k_3 += 4) {
                    unsigned int smem_regs_3[4];
                    __int128_t _smem_raw_5;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_5) : "r"(smem_stats_w_addr + (unsigned int)((100 + k_3) * 4)));
                    smem_regs_3[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[0];
                    smem_regs_3[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[1];
                    smem_regs_3[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[2];
                    smem_regs_3[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[3];
                    #pragma unroll
                    for (int e_3 = 0; e_3 < 4; e_3++) {
                        tmem_load_0[k_3 + e_3] = tmem_load_0[k_3 + e_3] * softmax_scale_log2 + __uint_as_float(smem_regs_3[e_3]);
                        pm_2 = max_noftz(pm_2, tmem_load_0[k_3 + e_3]);
                    }
                }
                float pm_1;
                asm("max.f32 %0, %1, %2;" : "=f"(pm_1) : "f"(pm), "f"(pm_2));
                int warp_exceed = __any_sync(0xFFFFFFFF, pm_1 > 8.807355f);
                if (lane == 0) {
                    smem_flags[aphase * 8 + warp_in_wg] = warp_exceed;
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                unsigned int smem_regs_4[4];
                __int128_t _smem_raw_6;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_6) : "r"(smem_flags_w_addr + (unsigned int)(aphase * 8 * 4)));
                smem_regs_4[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[0];
                smem_regs_4[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[1];
                smem_regs_4[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[2];
                smem_regs_4[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[3];
                unsigned int f_or = smem_regs_4[0] | smem_regs_4[1] | (smem_regs_4[2] | smem_regs_4[3]);
                int wg_exceed = 0;
                wg_exceed = reinterpret_cast<int*>(&f_or)[0];
                if (wg_exceed != 0) {
                    float red[4];
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        red[j] = tmem_load[j];
                    }
                    int upper = (lane & 16) != 0;
                    #pragma unroll
                    for (int j_2 = 0; j_2 < 2; j_2++) {
                        float send = ((upper != 0) ? red[j_2] : red[j_2 + 2]);
                        float keep = ((upper != 0) ? red[j_2 + 2] : red[j_2]);
                        float recv = __shfl_xor_sync(0xFFFFFFFF, send, 16);
                        red[j_2] = max_noftz(keep, recv);
                    }
                    int upper_1 = (lane & 8) != 0;
                    #pragma unroll
                    for (int j_3 = 0; j_3 < 1; j_3++) {
                        float send_1 = ((upper_1 != 0) ? red[j_3] : red[j_3 + 1]);
                        float keep_1 = ((upper_1 != 0) ? red[j_3 + 1] : red[j_3]);
                        float recv_1 = __shfl_xor_sync(0xFFFFFFFF, send_1, 8);
                        red[j_3] = max_noftz(keep_1, recv_1);
                    }
                    float other_0 = __shfl_xor_sync(0xFFFFFFFF, red[0], 4);
                    red[0] = max_noftz(red[0], other_0);
                    float other_1 = __shfl_xor_sync(0xFFFFFFFF, red[0], 2);
                    red[0] = max_noftz(red[0], other_1);
                    float other_2 = __shfl_xor_sync(0xFFFFFFFF, red[0], 1);
                    red[0] = max_noftz(red[0], other_2);
                    if ((lane & 7) == 0) {
                        smem_stats[32 + warp_in_wg * 8 + (lane >> 3)] = red[0];
                    }
                    float red_0[4];
                    #pragma unroll
                    for (int j_4 = 0; j_4 < 4; j_4++) {
                        red_0[j_4] = tmem_load_0[j_4];
                    }
                    int upper_0 = (lane & 16) != 0;
                    #pragma unroll
                    for (int j_5 = 0; j_5 < 2; j_5++) {
                        float send_2 = ((upper_0 != 0) ? red_0[j_5] : red_0[j_5 + 2]);
                        float keep_2 = ((upper_0 != 0) ? red_0[j_5 + 2] : red_0[j_5]);
                        float recv_2 = __shfl_xor_sync(0xFFFFFFFF, send_2, 16);
                        red_0[j_5] = max_noftz(keep_2, recv_2);
                    }
                    int upper_1_0 = (lane & 8) != 0;
                    #pragma unroll
                    for (int j_6 = 0; j_6 < 1; j_6++) {
                        float send_3 = ((upper_1_0 != 0) ? red_0[j_6] : red_0[j_6 + 1]);
                        float keep_3 = ((upper_1_0 != 0) ? red_0[j_6 + 1] : red_0[j_6]);
                        float recv_3 = __shfl_xor_sync(0xFFFFFFFF, send_3, 8);
                        red_0[j_6] = max_noftz(keep_3, recv_3);
                    }
                    float other_3 = __shfl_xor_sync(0xFFFFFFFF, red_0[0], 4);
                    red_0[0] = max_noftz(red_0[0], other_3);
                    float other_4 = __shfl_xor_sync(0xFFFFFFFF, red_0[0], 2);
                    red_0[0] = max_noftz(red_0[0], other_4);
                    float other_2_0 = __shfl_xor_sync(0xFFFFFFFF, red_0[0], 1);
                    red_0[0] = max_noftz(red_0[0], other_2_0);
                    if ((lane & 7) == 0) {
                        smem_stats[32 + warp_in_wg * 8 + 4 + (lane >> 3)] = red_0[0];
                    }
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                    if (is_reducer != 0) {
                        float m0 = smem_stats[32 + red_col];
                        float m1 = smem_stats[40 + red_col];
                        float m2 = smem_stats[48 + red_col];
                        float m3 = smem_stats[56 + red_col];
                        float tm;
                        asm("max.f32 %0, %1, %2;" : "=f"(tm) : "f"(max_noftz(m0, m1)), "f"(max_noftz(m2, m3)));
                        float c_old = smem_stats[96 + red_col];
                        float c_upd = 7.807355f - tm + c_old;
                        float run_old = smem_stats[112 + red_col];
                        float t_tile = 7.807355f - c_upd;
                        float run_grown;
                        asm("max.f32 %0, %1, %2;" : "=f"(run_grown) : "f"(run_old), "f"(t_tile));
                        float run_new = ((tm > -CUDART_INF_F) ? run_grown : run_old);
                        float c_floor = 7.807355f - run_new + 32.0f;
                        float c_bounded = ((c_upd <= c_floor) ? c_upd : c_floor);
                        float c_new = ((tm > -CUDART_INF_F) ? c_bounded : c_old);
                        float delta = c_new - c_old;
                        float alpha = approx_exp2(delta);
                        smem_stats[96 + red_col] = c_new;
                        smem_stats[112 + red_col] = run_new;
                        smem_stats[16 + red_col] = delta;
                        smem_stats[red_col] = alpha;
                    }
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                    #pragma unroll
                    for (int k_4 = 0; k_4 < 4; k_4 += 4) {
                        unsigned int smem_regs_0[4];
                        __int128_t _smem_raw_7;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_7) : "r"(smem_stats_w_addr + (unsigned int)((16 + k_4) * 4)));
                        smem_regs_0[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[0];
                        smem_regs_0[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[1];
                        smem_regs_0[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[2];
                        smem_regs_0[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[3];
                        unsigned int smem_regs_1_1[4];
                        __int128_t _smem_raw_8;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_8) : "r"(smem_stats_w_addr + (unsigned int)(k_4 * 4)));
                        smem_regs_1_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[0];
                        smem_regs_1_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[1];
                        smem_regs_1_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[2];
                        smem_regs_1_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[3];
                        #pragma unroll
                        for (int e_4 = 0; e_4 < 4; e_4++) {
                            tmem_load[k_4 + e_4] = tmem_load[k_4 + e_4] + __uint_as_float(smem_regs_0[e_4]);
                            psum0_0[k_4 + e_4] = psum0_0[k_4 + e_4] * __uint_as_float(smem_regs_1_1[e_4]);
                        }
                    }
                    #pragma unroll
                    for (int k_5 = 0; k_5 < 4; k_5 += 4) {
                        unsigned int smem_regs_2_1[4];
                        __int128_t _smem_raw_9;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_9) : "r"(smem_stats_w_addr + (unsigned int)((20 + k_5) * 4)));
                        smem_regs_2_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[0];
                        smem_regs_2_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[1];
                        smem_regs_2_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[2];
                        smem_regs_2_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[3];
                        unsigned int smem_regs_3_1[4];
                        __int128_t _smem_raw_10;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_10) : "r"(smem_stats_w_addr + (unsigned int)((4 + k_5) * 4)));
                        smem_regs_3_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[0];
                        smem_regs_3_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[1];
                        smem_regs_3_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[2];
                        smem_regs_3_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[3];
                        #pragma unroll
                        for (int e_5 = 0; e_5 < 4; e_5++) {
                            tmem_load_0[k_5 + e_5] = tmem_load_0[k_5 + e_5] + __uint_as_float(smem_regs_2_1[e_5]);
                            psum1_0[k_5 + e_5] = psum1_0[k_5 + e_5] * __uint_as_float(smem_regs_3_1[e_5]);
                        }
                    }
                }
                if (tile >= 2) {
                    int _stage_pt_free = (int)((unsigned int)(tile - 2) % 2);
                    int _parity_pt_free = (int)((unsigned int)(tile - 2) / 2 % 2);
                    mbarrier_wait(pt_free_addr + (_stage_pt_free) * 8, _parity_pt_free);
                }
                #pragma unroll
                for (int j_7 = 0; j_7 < 4; j_7++) {
                    float _exp2_11 = approx_exp2(tmem_load[j_7]);
                    tmem_load[j_7] = _exp2_11;
                    psum0_0[j_7] = psum0_0[j_7] + tmem_load[j_7];
                }
                unsigned int packed[1];
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
                int row_addr = smem_pt_addr + (unsigned int)(pphase * 16384) + (unsigned int)(my_tok * 128);
                int row_rel = pphase * 16384 + my_tok * 128;
                int swz_rel = row_rel + ((0 ^ my_tok & 7) << 4);
                smem_pt_words[swz_rel >> 2] = packed[0];
                #pragma unroll
                for (int j_8 = 0; j_8 < 4; j_8++) {
                    float _exp2_12 = approx_exp2(tmem_load_0[j_8]);
                    tmem_load_0[j_8] = _exp2_12;
                    psum1_0[j_8] = psum1_0[j_8] + tmem_load_0[j_8];
                }
                unsigned int packed_0[1];
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
                int row_addr_0 = smem_pt_addr + (unsigned int)(pphase * 16384) + (unsigned int)(my_tok * 128);
                int row_rel_0 = pphase * 16384 + my_tok * 128;
                int swz_rel_0 = row_rel_0 + ((0 ^ my_tok & 7) << 4) + 4;
                smem_pt_words[swz_rel_0 >> 2] = packed_0[0];
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (tile > 0) {
                    int _parity_pv_done = (int)((unsigned int)tile % 2) ^ 1;
                    mbarrier_wait(pv_done_addr, _parity_pv_done);
                }
                if (wg_exceed != 0) {
                    if (tile > 0) {
                        int _parity_pv_done_1 = (int)((unsigned int)tile % 2) ^ 1;
                        mbarrier_wait(pv_done_addr, _parity_pv_done_1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        #pragma unroll
                        for (int vs = 0; vs < 4; vs++) {
                            int o_a = taddr + 64 + (unsigned int)(vs * 32) + (unsigned int)(lane_base << 16);
                            int o_b = taddr + 64 + (unsigned int)(vs * 32) + 4 + (unsigned int)(lane_base << 16);
                            float _tmem_load_2[4];
                            tmem_ld_x4(&_tmem_load_2[0], o_a);
                            float _tmem_load_3[4];
                            tmem_ld_x4(&_tmem_load_3[0], o_b);
                            float vals[4];
                            #pragma unroll
                            for (int k_6 = 0; k_6 < 4; k_6 += 4) {
                                unsigned int smem_regs_4_1[4];
                                __int128_t _smem_raw_13;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_13) : "r"(smem_stats_w_addr + (unsigned int)(k_6 * 4)));
                                smem_regs_4_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_13)[0];
                                smem_regs_4_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_13)[1];
                                smem_regs_4_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_13)[2];
                                smem_regs_4_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_13)[3];
                                #pragma unroll
                                for (int e_6 = 0; e_6 < 4; e_6++) {
                                    vals[k_6 + e_6] = __uint_as_float(smem_regs_4_1[e_6]);
                                }
                            }
                            float vals_0[4];
                            #pragma unroll
                            for (int k_7 = 0; k_7 < 4; k_7 += 4) {
                                unsigned int smem_regs_5[4];
                                __int128_t _smem_raw_14;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_14) : "r"(smem_stats_w_addr + (unsigned int)((4 + k_7) * 4)));
                                smem_regs_5[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[0];
                                smem_regs_5[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[1];
                                smem_regs_5[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[2];
                                smem_regs_5[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[3];
                                #pragma unroll
                                for (int e_7 = 0; e_7 < 4; e_7++) {
                                    vals_0[k_7 + e_7] = __uint_as_float(smem_regs_5[e_7]);
                                }
                            }
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            #pragma unroll
                            for (int j_9 = 0; j_9 < 4; j_9++) {
                                _tmem_load_2[j_9] = _tmem_load_2[j_9] * vals[j_9];
                            }
                            #pragma unroll
                            for (int j_10 = 0; j_10 < 4; j_10++) {
                                _tmem_load_3[j_10] = _tmem_load_3[j_10] * vals_0[j_10];
                            }
                            tmem_st_x4_f32(o_a, _tmem_load_2);
                            tmem_st_x4_f32(o_b, _tmem_load_3);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(p_full_addr + ((int)((unsigned int)tile % 2)) * 8);
            }
            #pragma unroll
            for (int j_11 = 0; j_11 < 4; j_11++) {
                float wsum_j = psum0_0[j_11];
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    wsum_j += __shfl_xor_sync(0xFFFFFFFF, wsum_j, offset);
                if (lane == j_11 % 32) {
                    smem_stats[32 + warp_in_wg * 8 + j_11] = wsum_j;
                }
            }
            #pragma unroll
            for (int j_12 = 0; j_12 < 4; j_12++) {
                float wsum_j_1 = psum1_0[j_12];
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    wsum_j_1 += __shfl_xor_sync(0xFFFFFFFF, wsum_j_1, offset);
                if (lane == (4 + j_12) % 32) {
                    smem_stats[32 + warp_in_wg * 8 + 4 + j_12] = wsum_j_1;
                }
            }
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (is_reducer != 0) {
                float s0 = smem_stats[32 + red_col];
                float s1 = smem_stats[40 + red_col];
                float s2 = smem_stats[48 + red_col];
                float s3 = smem_stats[56 + red_col];
                float csum = s0 + s1 + (s2 + s3);
                float safe_sum = ((csum > 0.0f) ? csum : 1.0f);
                float out_scale = ((num_split == 1) ? bmm2_scale : 1.0f);
                float _rcp_15 = approx_rcp(safe_sum);
                smem_stats[128 + red_col] = _rcp_15 * out_scale;
                if (red_col < rows_valid) {
                    int stat_off = (row_base_global + red_col) * num_split + split_idx;
                    float c_fin = smem_stats[96 + red_col];
                    float stored_max = 7.807355f - c_fin;
                    *(reinterpret_cast<float*>(partial_max + stat_off) + (0)) = stored_max;
                    *(reinterpret_cast<float*>(partial_sum + stat_off) + (0)) = csum;
                    if (num_split == 1) {
                        float _log2_16;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_16) : "f"(csum));
                        float lse_row = (_log2_16 - c_fin) * lse_scale;
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
                float vals_1[8];
                #pragma unroll
                for (int k_8 = 0; k_8 < 8; k_8 += 4) {
                    unsigned int smem_regs_6[4];
                    __int128_t _smem_raw_17;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_17) : "r"(smem_stats_w_addr + (unsigned int)((128 + k_8) * 4)));
                    smem_regs_6[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_17)[0];
                    smem_regs_6[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_17)[1];
                    smem_regs_6[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_17)[2];
                    smem_regs_6[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_17)[3];
                    #pragma unroll
                    for (int e_8 = 0; e_8 < 4; e_8++) {
                        vals_1[k_8 + e_8] = __uint_as_float(smem_regs_6[e_8]);
                    }
                }
                mbarrier_wait(o_done_addr, _phase_o_done_0);
                _phase_o_done_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (int vs_2 = 0; vs_2 < 4; vs_2++) {
                    float _tmem_load_4[4];
                    tmem_ld_x4(&_tmem_load_4[0], taddr + 64 + (unsigned int)(vs_2 * 32) + (unsigned int)(lane_base << 16));
                    #pragma unroll
                    for (int j_13 = 0; j_13 < 4; j_13++) {
                        int c_1 = j_13;
                        if (c_1 < rows_valid) {
                            *(reinterpret_cast<__nv_bfloat16*>(partial_O + (out_base + c_1 * row_stride + vs_2 * 128)) + (0)) = (__nv_bfloat16)(_tmem_load_4[j_13] * vals_1[j_13]);
                        }
                    }
                    float tmem_load_1[4];
                    tmem_ld_x4(&tmem_load_1[0], taddr + 64 + (unsigned int)(vs_2 * 32) + 4 + (unsigned int)(lane_base << 16));
                    #pragma unroll
                    for (int j_14 = 0; j_14 < 4; j_14++) {
                        int c_2 = 4 + j_14;
                        if (c_2 < rows_valid) {
                            *(reinterpret_cast<__nv_bfloat16*>(partial_O + (out_base + c_2 * row_stride + vs_2 * 128)) + (0)) = (__nv_bfloat16)(tmem_load_1[j_14] * vals_1[4 + j_14]);
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
        asm volatile("setmaxnreg.inc.sync.aligned.u32 184;");
        { // softmax_wg1_main
            float psum0_1[4];
            float psum1_1[4];
            #pragma unroll
            for (int j_2_1 = 0; j_2_1 < 4; j_2_1++) {
                psum0_1[j_2_1] = 0.0f;
            }
            #pragma unroll
            for (int j_3_1 = 0; j_3_1 < 4; j_3_1++) {
                psum1_1[j_3_1] = 0.0f;
            }
            int split_idx_1 = blockIdx.x;
            int m_tile_1 = gridDim.y - 1 - blockIdx.y;
            int b_1 = blockIdx.z;
            int q_start_1 = cum_seq_lens_q[b_1];
            int q_len_b_1 = cum_seq_lens_q[b_1 + 1] - q_start_1;
            int kv_len_1 = seq_lens[b_1];
            int rows_b_1 = q_len_b_1 * num_heads;
            int row0_1 = m_tile_1 * 16;
            int rows_left_1 = rows_b_1 - row0_1;
            int rows_pos_1 = ((rows_left_1 < 0) ? 0 : rows_left_1);
            int rows_valid_1 = ((rows_pos_1 > 16) ? 16 : rows_pos_1);
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
            int red_col_1 = 8 + warp_in_wg_1 * 2 + lane;
            int is_reducer_1 = lane < 2;
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
                smem_stats[96 + red_col_1] = 1031.8074f;
                smem_stats[112 + red_col_1] = -CUDART_INF_F;
            }
            asm volatile("barrier.sync 2, 128;" ::: "memory");
            int vis_min_1 = smem_vis[8];
            #pragma unroll 1
            for (int tile_1 = 0; tile_1 < my_n_tiles_1; tile_1++) {
                int pphase_1 = tile_1 & 1;
                int aphase_1 = tile_1 & 1;
                int sphase_1 = tile_1 % 2;
                int _stage_s_full_1 = (int)((unsigned int)tile_1 % 2);
                int _parity_s_full_1 = (int)((unsigned int)tile_1 / 2 % 2);
                mbarrier_wait(s_full_addr + (_stage_s_full_1) * 8, _parity_s_full_1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int t_abs_1 = (my_start_1 + tile_1) * 128 + my_tok_1;
                float tmem_load_2[4];
                tmem_ld_x4(&tmem_load_2[0], taddr + (unsigned int)(sphase_1 * 32) + 8 + (unsigned int)(lane_base_1 << 16));
                float tmem_load_2_1[4];
                tmem_ld_x4(&tmem_load_2_1[0], taddr + (unsigned int)(sphase_1 * 32) + 8 + 4 + (unsigned int)(lane_base_1 << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(s_free_addr + ((int)((unsigned int)tile_1 % 2)) * 8);
                int tile_last_1 = (my_start_1 + tile_1) * 128 + 127;
                if (tile_last_1 >= vis_min_1) {
                    #pragma unroll
                    for (int k_9 = 0; k_9 < 4; k_9 += 4) {
                        unsigned int smem_regs_7[4];
                        __int128_t _smem_raw_0;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_0) : "r"(smem_vis_w_addr + (unsigned int)((8 + k_9) * 4)));
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
                                tmem_load_2[k_9 + e_9] = -CUDART_INF_F;
                            }
                        }
                    }
                    #pragma unroll
                    for (int k_10 = 0; k_10 < 4; k_10 += 4) {
                        unsigned int smem_regs_8[4];
                        __int128_t _smem_raw_2;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_2) : "r"(smem_vis_w_addr + (unsigned int)((12 + k_10) * 4)));
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
                                tmem_load_2_1[k_10 + e_10] = -CUDART_INF_F;
                            }
                        }
                    }
                }
                float pm_3 = -CUDART_INF_F;
                #pragma unroll
                for (int k_11 = 0; k_11 < 4; k_11 += 4) {
                    unsigned int smem_regs_9[4];
                    __int128_t _smem_raw_4;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_4) : "r"(smem_stats_w_addr + (unsigned int)((104 + k_11) * 4)));
                    smem_regs_9[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[0];
                    smem_regs_9[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[1];
                    smem_regs_9[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[2];
                    smem_regs_9[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[3];
                    #pragma unroll
                    for (int e_11 = 0; e_11 < 4; e_11++) {
                        tmem_load_2[k_11 + e_11] = tmem_load_2[k_11 + e_11] * softmax_scale_log2 + __uint_as_float(smem_regs_9[e_11]);
                        pm_3 = max_noftz(pm_3, tmem_load_2[k_11 + e_11]);
                    }
                }
                float pm_3_1 = -CUDART_INF_F;
                #pragma unroll
                for (int k_12 = 0; k_12 < 4; k_12 += 4) {
                    unsigned int smem_regs_10[4];
                    __int128_t _smem_raw_5;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_5) : "r"(smem_stats_w_addr + (unsigned int)((108 + k_12) * 4)));
                    smem_regs_10[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[0];
                    smem_regs_10[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[1];
                    smem_regs_10[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[2];
                    smem_regs_10[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[3];
                    #pragma unroll
                    for (int e_12 = 0; e_12 < 4; e_12++) {
                        tmem_load_2_1[k_12 + e_12] = tmem_load_2_1[k_12 + e_12] * softmax_scale_log2 + __uint_as_float(smem_regs_10[e_12]);
                        pm_3_1 = max_noftz(pm_3_1, tmem_load_2_1[k_12 + e_12]);
                    }
                }
                float pm_1_1;
                asm("max.f32 %0, %1, %2;" : "=f"(pm_1_1) : "f"(pm_3), "f"(pm_3_1));
                int warp_exceed_1 = __any_sync(0xFFFFFFFF, pm_1_1 > 8.807355f);
                if (lane == 0) {
                    smem_flags[aphase_1 * 8 + 4 + warp_in_wg_1] = warp_exceed_1;
                }
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                unsigned int smem_regs_11[4];
                __int128_t _smem_raw_6;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_6) : "r"(smem_flags_w_addr + (unsigned int)((aphase_1 * 8 + 4) * 4)));
                smem_regs_11[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[0];
                smem_regs_11[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[1];
                smem_regs_11[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[2];
                smem_regs_11[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[3];
                unsigned int f_or_1 = smem_regs_11[0] | smem_regs_11[1] | (smem_regs_11[2] | smem_regs_11[3]);
                int wg_exceed_1 = 0;
                wg_exceed_1 = reinterpret_cast<int*>(&f_or_1)[0];
                if (wg_exceed_1 != 0) {
                    float red_1[4];
                    #pragma unroll
                    for (int j_15 = 0; j_15 < 4; j_15++) {
                        red_1[j_15] = tmem_load_2[j_15];
                    }
                    int upper_2 = (lane & 16) != 0;
                    #pragma unroll
                    for (int j_16 = 0; j_16 < 2; j_16++) {
                        float send_4 = ((upper_2 != 0) ? red_1[j_16] : red_1[j_16 + 2]);
                        float keep_4 = ((upper_2 != 0) ? red_1[j_16 + 2] : red_1[j_16]);
                        float recv_4 = __shfl_xor_sync(0xFFFFFFFF, send_4, 16);
                        red_1[j_16] = max_noftz(keep_4, recv_4);
                    }
                    int upper_1_1 = (lane & 8) != 0;
                    #pragma unroll
                    for (int j_17 = 0; j_17 < 1; j_17++) {
                        float send_5 = ((upper_1_1 != 0) ? red_1[j_17] : red_1[j_17 + 1]);
                        float keep_5 = ((upper_1_1 != 0) ? red_1[j_17 + 1] : red_1[j_17]);
                        float recv_5 = __shfl_xor_sync(0xFFFFFFFF, send_5, 8);
                        red_1[j_17] = max_noftz(keep_5, recv_5);
                    }
                    float other_5 = __shfl_xor_sync(0xFFFFFFFF, red_1[0], 4);
                    red_1[0] = max_noftz(red_1[0], other_5);
                    float other_6 = __shfl_xor_sync(0xFFFFFFFF, red_1[0], 2);
                    red_1[0] = max_noftz(red_1[0], other_6);
                    float other_2_1 = __shfl_xor_sync(0xFFFFFFFF, red_1[0], 1);
                    red_1[0] = max_noftz(red_1[0], other_2_1);
                    if ((lane & 7) == 0) {
                        smem_stats[64 + warp_in_wg_1 * 8 + (lane >> 3)] = red_1[0];
                    }
                    float red_1_1[4];
                    #pragma unroll
                    for (int j_18 = 0; j_18 < 4; j_18++) {
                        red_1_1[j_18] = tmem_load_2_1[j_18];
                    }
                    int upper_2_1 = (lane & 16) != 0;
                    #pragma unroll
                    for (int j_19 = 0; j_19 < 2; j_19++) {
                        float send_6 = ((upper_2_1 != 0) ? red_1_1[j_19] : red_1_1[j_19 + 2]);
                        float keep_6 = ((upper_2_1 != 0) ? red_1_1[j_19 + 2] : red_1_1[j_19]);
                        float recv_6 = __shfl_xor_sync(0xFFFFFFFF, send_6, 16);
                        red_1_1[j_19] = max_noftz(keep_6, recv_6);
                    }
                    int upper_1_1_1 = (lane & 8) != 0;
                    #pragma unroll
                    for (int j_20 = 0; j_20 < 1; j_20++) {
                        float send_7 = ((upper_1_1_1 != 0) ? red_1_1[j_20] : red_1_1[j_20 + 1]);
                        float keep_7 = ((upper_1_1_1 != 0) ? red_1_1[j_20 + 1] : red_1_1[j_20]);
                        float recv_7 = __shfl_xor_sync(0xFFFFFFFF, send_7, 8);
                        red_1_1[j_20] = max_noftz(keep_7, recv_7);
                    }
                    float other_7 = __shfl_xor_sync(0xFFFFFFFF, red_1_1[0], 4);
                    red_1_1[0] = max_noftz(red_1_1[0], other_7);
                    float other_8 = __shfl_xor_sync(0xFFFFFFFF, red_1_1[0], 2);
                    red_1_1[0] = max_noftz(red_1_1[0], other_8);
                    float other_2_1_1 = __shfl_xor_sync(0xFFFFFFFF, red_1_1[0], 1);
                    red_1_1[0] = max_noftz(red_1_1[0], other_2_1_1);
                    if ((lane & 7) == 0) {
                        smem_stats[64 + warp_in_wg_1 * 8 + 4 + (lane >> 3)] = red_1_1[0];
                    }
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    if (is_reducer_1 != 0) {
                        float m0_1 = smem_stats[64 + (red_col_1 - 8)];
                        float m1_1 = smem_stats[72 + (red_col_1 - 8)];
                        float m2_1 = smem_stats[80 + (red_col_1 - 8)];
                        float m3_1 = smem_stats[88 + (red_col_1 - 8)];
                        float tm_1;
                        asm("max.f32 %0, %1, %2;" : "=f"(tm_1) : "f"(max_noftz(m0_1, m1_1)), "f"(max_noftz(m2_1, m3_1)));
                        float c_old_1 = smem_stats[96 + red_col_1];
                        float c_upd_1 = 7.807355f - tm_1 + c_old_1;
                        float run_old_1 = smem_stats[112 + red_col_1];
                        float t_tile_1 = 7.807355f - c_upd_1;
                        float run_grown_1;
                        asm("max.f32 %0, %1, %2;" : "=f"(run_grown_1) : "f"(run_old_1), "f"(t_tile_1));
                        float run_new_1 = ((tm_1 > -CUDART_INF_F) ? run_grown_1 : run_old_1);
                        float c_floor_1 = 7.807355f - run_new_1 + 32.0f;
                        float c_bounded_1 = ((c_upd_1 <= c_floor_1) ? c_upd_1 : c_floor_1);
                        float c_new_1 = ((tm_1 > -CUDART_INF_F) ? c_bounded_1 : c_old_1);
                        float delta_1 = c_new_1 - c_old_1;
                        float alpha_1 = approx_exp2(delta_1);
                        smem_stats[96 + red_col_1] = c_new_1;
                        smem_stats[112 + red_col_1] = run_new_1;
                        smem_stats[16 + red_col_1] = delta_1;
                        smem_stats[red_col_1] = alpha_1;
                    }
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    #pragma unroll
                    for (int k_13 = 0; k_13 < 4; k_13 += 4) {
                        unsigned int smem_regs_6_1[4];
                        __int128_t _smem_raw_7;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_7) : "r"(smem_stats_w_addr + (unsigned int)((24 + k_13) * 4)));
                        smem_regs_6_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[0];
                        smem_regs_6_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[1];
                        smem_regs_6_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[2];
                        smem_regs_6_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[3];
                        unsigned int smem_regs_7_1[4];
                        __int128_t _smem_raw_8;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_8) : "r"(smem_stats_w_addr + (unsigned int)((8 + k_13) * 4)));
                        smem_regs_7_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[0];
                        smem_regs_7_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[1];
                        smem_regs_7_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[2];
                        smem_regs_7_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[3];
                        #pragma unroll
                        for (int e_13 = 0; e_13 < 4; e_13++) {
                            tmem_load_2[k_13 + e_13] = tmem_load_2[k_13 + e_13] + __uint_as_float(smem_regs_6_1[e_13]);
                            psum0_1[k_13 + e_13] = psum0_1[k_13 + e_13] * __uint_as_float(smem_regs_7_1[e_13]);
                        }
                    }
                    #pragma unroll
                    for (int k_14 = 0; k_14 < 4; k_14 += 4) {
                        unsigned int smem_regs_8_1[4];
                        __int128_t _smem_raw_9;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_9) : "r"(smem_stats_w_addr + (unsigned int)((28 + k_14) * 4)));
                        smem_regs_8_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[0];
                        smem_regs_8_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[1];
                        smem_regs_8_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[2];
                        smem_regs_8_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[3];
                        unsigned int smem_regs_9_1[4];
                        __int128_t _smem_raw_10;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_10) : "r"(smem_stats_w_addr + (unsigned int)((12 + k_14) * 4)));
                        smem_regs_9_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[0];
                        smem_regs_9_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[1];
                        smem_regs_9_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[2];
                        smem_regs_9_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[3];
                        #pragma unroll
                        for (int e_14 = 0; e_14 < 4; e_14++) {
                            tmem_load_2_1[k_14 + e_14] = tmem_load_2_1[k_14 + e_14] + __uint_as_float(smem_regs_8_1[e_14]);
                            psum1_1[k_14 + e_14] = psum1_1[k_14 + e_14] * __uint_as_float(smem_regs_9_1[e_14]);
                        }
                    }
                }
                if (tile_1 >= 2) {
                    int _stage_pt_free_1 = (int)((unsigned int)(tile_1 - 2) % 2);
                    int _parity_pt_free_1 = (int)((unsigned int)(tile_1 - 2) / 2 % 2);
                    mbarrier_wait(pt_free_addr + (_stage_pt_free_1) * 8, _parity_pt_free_1);
                }
                #pragma unroll
                for (int j_21 = 0; j_21 < 4; j_21++) {
                    float _exp2_11_1 = approx_exp2(tmem_load_2[j_21]);
                    tmem_load_2[j_21] = _exp2_11_1;
                    psum0_1[j_21] = psum0_1[j_21] + tmem_load_2[j_21];
                }
                unsigned int packed_1[1];
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
                int row_addr_1 = smem_pt_addr + (unsigned int)(pphase_1 * 16384) + (unsigned int)(my_tok_1 * 128);
                int row_rel_1 = pphase_1 * 16384 + my_tok_1 * 128;
                int swz_rel_1 = row_rel_1 + ((0 ^ my_tok_1 & 7) << 4) + 8;
                smem_pt_words[swz_rel_1 >> 2] = packed_1[0];
                #pragma unroll
                for (int j_22 = 0; j_22 < 4; j_22++) {
                    float _exp2_12_1 = approx_exp2(tmem_load_2_1[j_22]);
                    tmem_load_2_1[j_22] = _exp2_12_1;
                    psum1_1[j_22] = psum1_1[j_22] + tmem_load_2_1[j_22];
                }
                unsigned int packed_1_1[1];
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
                int row_addr_1_1 = smem_pt_addr + (unsigned int)(pphase_1 * 16384) + (unsigned int)(my_tok_1 * 128);
                int row_rel_1_1 = pphase_1 * 16384 + my_tok_1 * 128;
                int swz_rel_1_1 = row_rel_1_1 + ((0 ^ my_tok_1 & 7) << 4) + 12;
                smem_pt_words[swz_rel_1_1 >> 2] = packed_1_1[0];
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (tile_1 > 0) {
                    int _parity_pv_done_2 = (int)((unsigned int)tile_1 % 2) ^ 1;
                    mbarrier_wait(pv_done_addr, _parity_pv_done_2);
                }
                if (wg_exceed_1 != 0) {
                    if (tile_1 > 0) {
                        int _parity_pv_done_3 = (int)((unsigned int)tile_1 % 2) ^ 1;
                        mbarrier_wait(pv_done_addr, _parity_pv_done_3);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        #pragma unroll
                        for (int vs_3 = 0; vs_3 < 4; vs_3++) {
                            int o_a_1 = taddr + 64 + (unsigned int)(vs_3 * 32) + 8 + (unsigned int)(lane_base_1 << 16);
                            int o_b_1 = taddr + 64 + (unsigned int)(vs_3 * 32) + 8 + 4 + (unsigned int)(lane_base_1 << 16);
                            float tmem_load_3[4];
                            tmem_ld_x4(&tmem_load_3[0], o_a_1);
                            float tmem_load_1_1[4];
                            tmem_ld_x4(&tmem_load_1_1[0], o_b_1);
                            float vals_2[4];
                            #pragma unroll
                            for (int k_15 = 0; k_15 < 4; k_15 += 4) {
                                unsigned int smem_regs_10_1[4];
                                __int128_t _smem_raw_13;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_13) : "r"(smem_stats_w_addr + (unsigned int)((8 + k_15) * 4)));
                                smem_regs_10_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_13)[0];
                                smem_regs_10_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_13)[1];
                                smem_regs_10_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_13)[2];
                                smem_regs_10_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_13)[3];
                                #pragma unroll
                                for (int e_15 = 0; e_15 < 4; e_15++) {
                                    vals_2[k_15 + e_15] = __uint_as_float(smem_regs_10_1[e_15]);
                                }
                            }
                            float vals_1_1[4];
                            #pragma unroll
                            for (int k_16 = 0; k_16 < 4; k_16 += 4) {
                                unsigned int smem_regs_11_1[4];
                                __int128_t _smem_raw_14;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_14) : "r"(smem_stats_w_addr + (unsigned int)((12 + k_16) * 4)));
                                smem_regs_11_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[0];
                                smem_regs_11_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[1];
                                smem_regs_11_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[2];
                                smem_regs_11_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[3];
                                #pragma unroll
                                for (int e_16 = 0; e_16 < 4; e_16++) {
                                    vals_1_1[k_16 + e_16] = __uint_as_float(smem_regs_11_1[e_16]);
                                }
                            }
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            #pragma unroll
                            for (int j_23 = 0; j_23 < 4; j_23++) {
                                tmem_load_3[j_23] = tmem_load_3[j_23] * vals_2[j_23];
                            }
                            #pragma unroll
                            for (int j_24 = 0; j_24 < 4; j_24++) {
                                tmem_load_1_1[j_24] = tmem_load_1_1[j_24] * vals_1_1[j_24];
                            }
                            tmem_st_x4_f32(o_a_1, tmem_load_3);
                            tmem_st_x4_f32(o_b_1, tmem_load_1_1);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(p_full_addr + ((int)((unsigned int)tile_1 % 2)) * 8);
            }
            #pragma unroll
            for (int j_25 = 0; j_25 < 4; j_25++) {
                float wsum_j_2 = psum0_1[j_25];
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    wsum_j_2 += __shfl_xor_sync(0xFFFFFFFF, wsum_j_2, offset);
                if (lane == j_25 % 32) {
                    smem_stats[64 + warp_in_wg_1 * 8 + j_25] = wsum_j_2;
                }
            }
            #pragma unroll
            for (int j_26 = 0; j_26 < 4; j_26++) {
                float wsum_j_3 = psum1_1[j_26];
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    wsum_j_3 += __shfl_xor_sync(0xFFFFFFFF, wsum_j_3, offset);
                if (lane == (4 + j_26) % 32) {
                    smem_stats[64 + warp_in_wg_1 * 8 + 4 + j_26] = wsum_j_3;
                }
            }
            asm volatile("barrier.sync 2, 128;" ::: "memory");
            if (is_reducer_1 != 0) {
                float s0_1 = smem_stats[64 + (red_col_1 - 8)];
                float s1_1 = smem_stats[72 + (red_col_1 - 8)];
                float s2_1 = smem_stats[80 + (red_col_1 - 8)];
                float s3_1 = smem_stats[88 + (red_col_1 - 8)];
                float csum_1 = s0_1 + s1_1 + (s2_1 + s3_1);
                float safe_sum_1 = ((csum_1 > 0.0f) ? csum_1 : 1.0f);
                float out_scale_1 = ((num_split == 1) ? bmm2_scale : 1.0f);
                float _rcp_15_1 = approx_rcp(safe_sum_1);
                smem_stats[128 + red_col_1] = _rcp_15_1 * out_scale_1;
                if (red_col_1 < rows_valid_1) {
                    int stat_off_1 = (row_base_global_1 + red_col_1) * num_split + split_idx_1;
                    float c_fin_1 = smem_stats[96 + red_col_1];
                    float stored_max_1 = 7.807355f - c_fin_1;
                    *(reinterpret_cast<float*>(partial_max + stat_off_1) + (0)) = stored_max_1;
                    *(reinterpret_cast<float*>(partial_sum + stat_off_1) + (0)) = csum_1;
                    if (num_split == 1) {
                        float _log2_16_1;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_16_1) : "f"(csum_1));
                        float lse_row_1 = (_log2_16_1 - c_fin_1) * lse_scale;
                        *(reinterpret_cast<float*>(lse + (row_base_global_1 + red_col_1)) + (0)) = lse_row_1;
                    }
                }
            }
            asm volatile("barrier.sync 2, 128;" ::: "memory");
            unsigned int _phase_o_done_0_1 = 0;
            if (my_n_tiles_1 > 0) {
                float vals_3[8];
                #pragma unroll
                for (int k_17 = 0; k_17 < 8; k_17 += 4) {
                    unsigned int smem_regs_12[4];
                    __int128_t _smem_raw_17;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_17) : "r"(smem_stats_w_addr + (unsigned int)((136 + k_17) * 4)));
                    smem_regs_12[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_17)[0];
                    smem_regs_12[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_17)[1];
                    smem_regs_12[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_17)[2];
                    smem_regs_12[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_17)[3];
                    #pragma unroll
                    for (int e_17 = 0; e_17 < 4; e_17++) {
                        vals_3[k_17 + e_17] = __uint_as_float(smem_regs_12[e_17]);
                    }
                }
                mbarrier_wait(o_done_addr, _phase_o_done_0_1);
                _phase_o_done_0_1 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (int vs_4 = 0; vs_4 < 4; vs_4++) {
                    float tmem_load_4[4];
                    tmem_ld_x4(&tmem_load_4[0], taddr + 64 + (unsigned int)(vs_4 * 32) + 8 + (unsigned int)(lane_base_1 << 16));
                    #pragma unroll
                    for (int j_27 = 0; j_27 < 4; j_27++) {
                        int c_3 = 8 + j_27;
                        if (c_3 < rows_valid_1) {
                            *(reinterpret_cast<__nv_bfloat16*>(partial_O + (out_base_1 + c_3 * row_stride_1 + vs_4 * 128)) + (0)) = (__nv_bfloat16)(tmem_load_4[j_27] * vals_3[j_27]);
                        }
                    }
                    float tmem_load_4_1[4];
                    tmem_ld_x4(&tmem_load_4_1[0], taddr + 64 + (unsigned int)(vs_4 * 32) + 8 + 4 + (unsigned int)(lane_base_1 << 16));
                    #pragma unroll
                    for (int j_28 = 0; j_28 < 4; j_28++) {
                        int c_4 = 12 + j_28;
                        if (c_4 < rows_valid_1) {
                            *(reinterpret_cast<__nv_bfloat16*>(partial_O + (out_base_1 + c_4 * row_stride_1 + vs_4 * 128)) + (0)) = (__nv_bfloat16)(tmem_load_4_1[j_28] * vals_3[4 + j_28]);
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
    if (warp == 8) {
        { // mma_warp_main
            int split_idx_2 = blockIdx.x;
            int m_tile_2 = gridDim.y - 1 - blockIdx.y;
            int b_2 = blockIdx.z;
            int q_start_2 = cum_seq_lens_q[b_2];
            int q_len_b_2 = cum_seq_lens_q[b_2 + 1] - q_start_2;
            int kv_len_2 = seq_lens[b_2];
            int rows_b_2 = q_len_b_2 * num_heads;
            int row0_2 = m_tile_2 * 16;
            int rows_left_2 = rows_b_2 - row0_2;
            int rows_pos_2 = ((rows_left_2 < 0) ? 0 : rows_left_2);
            int rows_valid_2 = ((rows_pos_2 > 16) ? 16 : rows_pos_2);
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
            unsigned int mma_stage = 0;
            unsigned int rmma_stage = 0;
            unsigned int _phase_q_full_0 = 0;
            mbarrier_wait(q_full_addr, _phase_q_full_0);
            _phase_q_full_0 ^= 1;
            unsigned int _phase_kv_full = 0;
            unsigned int _phase_rope_full = 0;
            #pragma unroll 1
            for (int tile_2 = 0; tile_2 < my_n_tiles_2; tile_2++) {
                int sphase_2 = tile_2 & 1;
                if (tile_2 >= 2) {
                    int _stage_s_free = (int)((unsigned int)(tile_2 - 2) % 2);
                    int _parity_s_free = (int)((unsigned int)(tile_2 - 2) / 2 % 2);
                    mbarrier_wait(s_free_addr + (_stage_s_free) * 8, _parity_s_free);
                }
                #pragma unroll
                for (int st = 0; st < 4; st++) {
                    mbarrier_wait(kv_full_addr + (mma_stage) * 8, _phase_kv_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_0 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (mma_stage) * 1024);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (st) * 128);
                    {
                        uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                        uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem + (sphase_2 * 32)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134479888, ((st == 0) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem + (sphase_2 * 32)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134479888, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem + (sphase_2 * 32)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134479888, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem + (sphase_2 * 32)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134479888, 1);
                        }
                    }
                    mma_stage += 1;
                    if (mma_stage == 10) { mma_stage = 0; _phase_kv_full ^= 1; }
                }
                mbarrier_wait(rope_full_addr + (rmma_stage) * 8, _phase_rope_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_1 = make_warp_uniform((((smem_kr_addr) >> 4) & 0x3FFF) + (rmma_stage) * 512);
                int _mma_b_lo_1 = make_warp_uniform(((smem_qr_addr) >> 4) & 0x3FFF);
                {
                    uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x80004020U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                    uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x80004020U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (sphase_2 * 32)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134479888, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (sphase_2 * 32)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134479888, 1);
                    }
                }
                elect_commit(s_full_addr + ((int)((unsigned int)tile_2 % 2)) * 8);
                elect_commit(rope_empty_addr + (rmma_stage) * 8);
                rmma_stage += 1;
                if (rmma_stage == 2) { rmma_stage = 0; _phase_rope_full ^= 1; }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_dealloc_addr);
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait(tmem_dealloc_addr, _phase_tmem_dealloc_0);
            _phase_tmem_dealloc_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
        }
    }
    // ---- Role: pv_warp ----
    if (warp == 9) {
        { // pv_warp_main
            int split_idx_3 = blockIdx.x;
            int m_tile_3 = gridDim.y - 1 - blockIdx.y;
            int b_3 = blockIdx.z;
            int q_start_3 = cum_seq_lens_q[b_3];
            int q_len_b_3 = cum_seq_lens_q[b_3 + 1] - q_start_3;
            int kv_len_3 = seq_lens[b_3];
            int rows_b_3 = q_len_b_3 * num_heads;
            int row0_3 = m_tile_3 * 16;
            int rows_left_3 = rows_b_3 - row0_3;
            int rows_pos_3 = ((rows_left_3 < 0) ? 0 : rows_left_3);
            int rows_valid_3 = ((rows_pos_3 > 16) ? 16 : rows_pos_3);
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
            int first_pv = 1;
            unsigned int pv_stage = 0;
            #pragma unroll 1
            for (int tile_3 = 0; tile_3 < my_n_tiles_3; tile_3++) {
                int pv_pp = tile_3 % 2;
                int _stage_p_full = (int)((unsigned int)tile_3 % 2);
                int _parity_p_full = (int)((unsigned int)tile_3 / 2 % 2);
                mbarrier_wait(p_full_addr + (_stage_p_full) * 8, _parity_p_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (int vs_5 = 0; vs_5 < 4; vs_5++) {
                    int _mma_a_lo_2 = make_warp_uniform(((((smem_vt_addr) >> 4) & 0x3FFF) | 0x4000000) + (pv_stage) * 1024);
                    int _mma_b_lo_2 = make_warp_uniform(((((smem_pt_addr) >> 4) & 0x3FFF) | 0x4000000) + (pv_pp) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                        uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem + (64 + vs_5 * 32)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134578192, ((first_pv) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 256U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 256U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem + (64 + vs_5 * 32)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134578192, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 256U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 256U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem + (64 + vs_5 * 32)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134578192, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 256U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 256U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem + (64 + vs_5 * 32)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134578192, 1);
                        }
                    }
                    elect_commit(kv_empty_addr + (pv_stage) * 8);
                    pv_stage += 1;
                    if (pv_stage == 10) { pv_stage = 0; }
                }
                elect_commit(pt_free_addr + ((int)((unsigned int)tile_3 % 2)) * 8);
                elect_commit(pv_done_addr);
                first_pv = 0;
            }
            if (my_n_tiles_3 > 0) {
                elect_commit(o_done_addr);
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: load_warp ----
    if (warp == 10) {
        { // load_warp_main
            int split_idx_4 = blockIdx.x;
            int m_tile_4 = gridDim.y - 1 - blockIdx.y;
            int b_4 = blockIdx.z;
            int q_start_4 = cum_seq_lens_q[b_4];
            int q_len_b_4 = cum_seq_lens_q[b_4 + 1] - q_start_4;
            int kv_len_4 = seq_lens[b_4];
            int rows_b_4 = q_len_b_4 * num_heads;
            int row0_4 = m_tile_4 * 16;
            int rows_left_4 = rows_b_4 - row0_4;
            int rows_pos_4 = ((rows_left_4 < 0) ? 0 : rows_left_4);
            int rows_valid_4 = ((rows_pos_4 > 16) ? 16 : rows_pos_4);
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
            unsigned int load_stage = 0;
            unsigned int rope_stage = 0;
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(q_full_addr, 9216);
                #pragma unroll
                for (int st_1 = 0; st_1 < 4; st_1++) {
                    tma_2d_gmem2smem(smem_q_addr + (unsigned int)(st_1 * 2048), (&tmap_q), st_1 * 128, row_base_global_4, q_full_addr);
                }
                tma_2d_gmem2smem(smem_qr_addr, (&tmap_qr), 512, row_base_global_4, q_full_addr);
            }
            if (my_n_tiles_4 > 0) {
                if (elect_sync()) {
                    int p0 = my_start_4 * 2;
                    int p1_raw = p0 + 1;
                    int p1 = ((p1_raw >= n_pages_4) ? p0 : p1_raw);
                    int g0 = page_table[pt_base_4 + p0];
                    int g1 = page_table[pt_base_4 + p1];
                    #pragma unroll
                    for (int st_2 = 0; st_2 < 4; st_2++) {
                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&tmap_k))), "r"((int)(st_2 * 128)), "r"((int)(g0 * 64)) : "memory");
                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&tmap_k))), "r"((int)(st_2 * 128)), "r"((int)(g1 * 64)) : "memory");
                    }
                    asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&tmap_kr))), "r"((int)(512)), "r"((int)(g0 * 64)) : "memory");
                    asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&tmap_kr))), "r"((int)(512)), "r"((int)(g1 * 64)) : "memory");
                }
            }
            unsigned int _phase_kv_empty = 1;
            unsigned int _phase_rope_empty = 1;
            #pragma unroll 1
            for (int tile_4 = 0; tile_4 < my_n_tiles_4; tile_4++) {
                int p0_1 = (my_start_4 + tile_4) * 2;
                int p1_raw_1 = p0_1 + 1;
                int p1_1 = ((p1_raw_1 >= n_pages_4) ? p0_1 : p1_raw_1);
                int g0_1 = page_table[pt_base_4 + p0_1];
                int g1_1 = page_table[pt_base_4 + p1_1];
                #pragma unroll
                for (int st_3 = 0; st_3 < 4; st_3++) {
                    mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 16384);
                        tma_2d_gmem2smem(smem_k_addr + load_stage * 16384, (&tmap_k), st_3 * 128, g0_1 * 64, kv_full_addr + (load_stage) * 8);
                        tma_2d_gmem2smem(smem_k_addr + load_stage * 16384 + 8192, (&tmap_k), st_3 * 128, g1_1 * 64, kv_full_addr + (load_stage) * 8);
                    }
                    load_stage += 1;
                    if (load_stage == 10) { load_stage = 0; _phase_kv_empty ^= 1; }
                }
                mbarrier_wait(rope_empty_addr + (rope_stage) * 8, _phase_rope_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(rope_full_addr + (rope_stage) * 8, 8192);
                    tma_2d_gmem2smem(smem_kr_addr + rope_stage * 8192, (&tmap_kr), 512, g0_1 * 64, rope_full_addr + (rope_stage) * 8);
                    tma_2d_gmem2smem(smem_kr_addr + rope_stage * 8192 + 4096, (&tmap_kr), 512, g1_1 * 64, rope_full_addr + (rope_stage) * 8);
                }
                rope_stage += 1;
                if (rope_stage == 2) { rope_stage = 0; _phase_rope_empty ^= 1; }
                int pf_tile = tile_4 + 1;
                if (pf_tile < my_n_tiles_4) {
                    if (elect_sync()) {
                        int p0_0 = (my_start_4 + pf_tile) * 2;
                        int p1_raw_0 = p0_0 + 1;
                        int p1_0 = ((p1_raw_0 >= n_pages_4) ? p0_0 : p1_raw_0);
                        int g0_0 = page_table[pt_base_4 + p0_0];
                        int g1_0 = page_table[pt_base_4 + p1_0];
                        #pragma unroll
                        for (int st_4 = 0; st_4 < 4; st_4++) {
                            asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&tmap_k))), "r"((int)(st_4 * 128)), "r"((int)(g0_0 * 64)) : "memory");
                            asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&tmap_k))), "r"((int)(st_4 * 128)), "r"((int)(g1_0 * 64)) : "memory");
                        }
                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&tmap_kr))), "r"((int)(512)), "r"((int)(g0_0 * 64)) : "memory");
                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&tmap_kr))), "r"((int)(512)), "r"((int)(g1_0 * 64)) : "memory");
                    }
                }
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: idle ----
    if (warp == 11) {
        // idle — no tasks assigned
    }

    // Cleanup
}

} // extern "C"
