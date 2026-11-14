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

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define LAUNCH_MIN_BLOCKS 2

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




union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};



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




__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}

extern "C" {

__global__ __launch_bounds__(192, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdn_det_train_1dbb3715625ef542df16_kcp(const __grid_constant__ CUtensorMap k_tma, const __grid_constant__ CUtensorMap w_tma, __nv_bfloat16* __restrict__ u, float* __restrict__ rowscale, float* __restrict__ gamma, int num_heads, int num_qk_heads, int* __restrict__ chunk_bos, int* __restrict__ seq_chunk_start, int num_seqs, int* __restrict__ walk_seg, float* __restrict__ h0, int* __restrict__ h0_index, int h0_hg, int h0_n, int tile_stride, const __grid_constant__ CUtensorMap q_tma, const __grid_constant__ CUtensorMap aqk_tma, float* __restrict__ qscale, int* __restrict__ chunk_len, __nv_bfloat16* __restrict__ o_out, const __grid_constant__ CUtensorMap u_tma)
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
    #define w_full_addr (mbar_base + 0)
    #define w_empty_addr (mbar_base + 8)
    #define k_full_addr (mbar_base + 16)
    #define k_empty_addr (mbar_base + 24)
    #define q_full_addr (mbar_base + 32)
    #define q_empty_addr (mbar_base + 40)
    #define aqk_full_addr (mbar_base + 48)
    #define aqk_empty_addr (mbar_base + 56)
    #define hs_ready_addr (mbar_base + 64)
    #define vp_ready_addr (mbar_base + 72)
    #define bv_ready_addr (mbar_base + 80)
    #define o_done_addr (mbar_base + 88)
    #define h_ready_addr (mbar_base + 96)
    #define u_full_addr (mbar_base + 104)
    #define u_empty_addr (mbar_base + 120)
    #define o_free_addr (mbar_base + 136)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    __nv_bfloat16* w_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int w_panel_addr = smem + 1024;
    __nv_bfloat16* k_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int k_panel_addr = smem + 33792;
    __nv_bfloat16* u_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int u_panel_addr = smem + 17408;
    __nv_bfloat16* w_op = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int w_op_addr = smem + 1024;
    __nv_bfloat16* k_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int k_mn_addr = smem + 33792;
    __nv_bfloat16* hs_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int hs_mn_addr = smem + 66560;
    __nv_bfloat16* vn_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 50176);
    const int vn_mn_addr = smem + 50176;
    __nv_bfloat16* bv_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 82944);
    const int bv_mn_addr = smem + 82944;
    __nv_bfloat16* hs = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int hs_addr = smem + 66560;
    __nv_bfloat16* vn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 50176);
    const int vn_addr = smem + 50176;
    __nv_bfloat16* bv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 82944);
    const int bv_addr = smem + 82944;
    __nv_bfloat16* q_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 91136);
    const int q_panel_addr = smem + 91136;
    __nv_bfloat16* q_op = reinterpret_cast<__nv_bfloat16*>(smem_raw + 91136);
    const int q_op_addr = smem + 91136;
    __nv_bfloat16* aqk_op = reinterpret_cast<__nv_bfloat16*>(smem_raw + 107520);
    const int aqk_op_addr = smem + 107520;

    // Mbarrier init (18 barriers at smem_raw[0..144))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'one' ---
            // w_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // w_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // k_full: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            // k_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            // aqk_full: 1 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            // aqk_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            // hs_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 64, 128);
            // vp_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            // bv_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 80, 128);
            // o_done: 1 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            // h_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            // --- pipeline 'ring' ---
            // u_full: 2 barriers, init_count=1
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            // u_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 120, 128);
            mbarrier_init(smem + 128, 128);
            // --- pipeline 'one' ---
            // o_free: 1 barriers, init_count=128
            mbarrier_init(smem + 136, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 144);
    if (warp == 0) {
        int _tmem_hold = smem + 144;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_acc_h = taddr;
    const unsigned int tmem_acc_vp = taddr + 64;
    const unsigned int tmem_acc_qh = taddr + 128;
    const unsigned int tmem_acc_av = taddr + 192;

    // ---- Role: compute ----
    if (warp <= 3) {
        { // compute_main
            int hb = blockIdx.x / 2;
            int vbase = blockIdx.x % 2 * 64;
            int head = hb % num_heads;
            int walk = hb / num_heads;
            int seg = walk_seg[walk];
            int c0 = seq_chunk_start[seg];
            int nc = seq_chunk_start[seg + 1] - c0;
            int row0 = c0 * 64;
            int warp_in_wg = warp % 4;
            int k_row = warp_in_wg * 32 + lane;
            int h_row_base = warp_in_wg * 32 << 16;
            int lane_half = lane >> 4;
            int t_row = warp_in_wg * 16 + (lane & 15);
            int vp_row_base = warp_in_wg * 32 << 16;
            float init_row[16];
            int h0_slot = h0_index[seg];
            if (h0_slot >= 0) {
                int h0_tile = (head / h0_hg * h0_n + h0_slot) * h0_hg + head % h0_hg;
                long long h0_base = (long long)h0_tile * (long long)tile_stride * 16384ll + (long long)(k_row * 128 + vbase);
                #pragma unroll
                for (int blk = 0; blk < 4; blk++) {
                    #pragma unroll
                    for (int q = 0; q < 4; q++) {
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(h0 + h0_base + (long long)(blk * 16 + q * 4));
                            init_row[q * 4 + 0] = _v4.x;
                            init_row[q * 4 + 1] = _v4.y;
                            init_row[q * 4 + 2] = _v4.z;
                            init_row[q * 4 + 3] = _v4.w;
                        }
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)h_row_base + (unsigned int)(blk * 16), init_row);
                }
            } else {
                init_row[0] = 0.0f;
                init_row[1] = 0.0f;
                init_row[2] = 0.0f;
                init_row[3] = 0.0f;
                init_row[4] = 0.0f;
                init_row[5] = 0.0f;
                init_row[6] = 0.0f;
                init_row[7] = 0.0f;
                init_row[8] = 0.0f;
                init_row[9] = 0.0f;
                init_row[10] = 0.0f;
                init_row[11] = 0.0f;
                init_row[12] = 0.0f;
                init_row[13] = 0.0f;
                init_row[14] = 0.0f;
                init_row[15] = 0.0f;
                #pragma unroll
                for (int blk_1 = 0; blk_1 < 4; blk_1++) {
                    tmem_st_x16_f32(taddr + (unsigned int)h_row_base + (unsigned int)(blk_1 * 16), init_row);
                }
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            float decay = 0.0f;
            float rs = 0.0f;
            float qs_next = 0.0f;
            int bos_next = 0;
            int len_next = 0;
            long long g_next = (long long)c0 * (long long)num_heads + (long long)head;
            long long rs_next = (long long)(row0 + t_row) * (long long)num_heads + (long long)head;
            if (nc > 0) {
                decay = gamma[g_next];
                rs = rowscale[rs_next];
                qs_next = qscale[rs_next];
                bos_next = chunk_bos[c0];
                len_next = chunk_len[c0];
            }
            unsigned int u_cursor = 0;
            unsigned int u_pk[16];
            float next_f[3];
            int next_i[2];
            float qs_prev = 0.0f;
            int bos_prev = 0;
            int len_prev = 0;
            unsigned int _phase_u_full = 0;
            unsigned int _phase_vp_ready_0 = 0;
            unsigned int _phase_h_ready_0 = 0;
            #pragma unroll 1
            for (int c = 0; c < nc; c++) {
                float h_i_0[16];
                tmem_ld_x16(&h_i_0[0], taddr + (unsigned int)h_row_base);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                unsigned int h_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_i_0[_lp*2 + 0], h_i_0[_lp*2+1 + 0]));
                    h_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_1 = 0; q_1 < 2; q_1++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)(q_1 * 8 / 64 * 16384 + k_row * 128 + q_1 * 8 % 64 * 2 ^ (q_1 * 8 / 64 * 16384 + k_row * 128 + q_1 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(h_bf16[q_1 * 4]), "r"(h_bf16[q_1 * 4 + 1]), "r"(h_bf16[q_1 * 4 + 2]), "r"(h_bf16[q_1 * 4 + 3]) : "memory");
                }
                float hd[16];
                #pragma unroll
                for (int j = 0; j < 16; j++) {
                    hd[j] = h_i_0[j] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base, hd);
                float h_i_1[16];
                tmem_ld_x16(&h_i_1[0], taddr + (unsigned int)h_row_base + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                unsigned int h_bf16_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_i_1[_lp*2 + 0], h_i_1[_lp*2+1 + 0]));
                    h_bf16_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_2 = 0; q_2 < 2; q_2++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((16 + q_2 * 8) / 64 * 16384 + k_row * 128 + (16 + q_2 * 8) % 64 * 2 ^ ((16 + q_2 * 8) / 64 * 16384 + k_row * 128 + (16 + q_2 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(h_bf16_0[q_2 * 4]), "r"(h_bf16_0[q_2 * 4 + 1]), "r"(h_bf16_0[q_2 * 4 + 2]), "r"(h_bf16_0[q_2 * 4 + 3]) : "memory");
                }
                float hd_0[16];
                #pragma unroll
                for (int j_1 = 0; j_1 < 16; j_1++) {
                    hd_0[j_1] = h_i_1[j_1] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 16, hd_0);
                float h_i_2[16];
                tmem_ld_x16(&h_i_2[0], taddr + (unsigned int)h_row_base + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                unsigned int h_bf16_1[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_i_2[_lp*2 + 0], h_i_2[_lp*2+1 + 0]));
                    h_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_3 = 0; q_3 < 2; q_3++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((32 + q_3 * 8) / 64 * 16384 + k_row * 128 + (32 + q_3 * 8) % 64 * 2 ^ ((32 + q_3 * 8) / 64 * 16384 + k_row * 128 + (32 + q_3 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(h_bf16_1[q_3 * 4]), "r"(h_bf16_1[q_3 * 4 + 1]), "r"(h_bf16_1[q_3 * 4 + 2]), "r"(h_bf16_1[q_3 * 4 + 3]) : "memory");
                }
                float hd_1[16];
                #pragma unroll
                for (int j_2 = 0; j_2 < 16; j_2++) {
                    hd_1[j_2] = h_i_2[j_2] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 32, hd_1);
                float h_i_3[16];
                tmem_ld_x16(&h_i_3[0], taddr + (unsigned int)h_row_base + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                unsigned int h_bf16_2[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_i_3[_lp*2 + 0], h_i_3[_lp*2+1 + 0]));
                    h_bf16_2[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_4 = 0; q_4 < 2; q_4++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((48 + q_4 * 8) / 64 * 16384 + k_row * 128 + (48 + q_4 * 8) % 64 * 2 ^ ((48 + q_4 * 8) / 64 * 16384 + k_row * 128 + (48 + q_4 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(h_bf16_2[q_4 * 4]), "r"(h_bf16_2[q_4 * 4 + 1]), "r"(h_bf16_2[q_4 * 4 + 2]), "r"(h_bf16_2[q_4 * 4 + 3]) : "memory");
                }
                float hd_2[16];
                #pragma unroll
                for (int j_3 = 0; j_3 < 16; j_3++) {
                    hd_2[j_3] = h_i_3[j_3] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 48, hd_2);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(hs_ready_addr);
                mbarrier_wait(u_full_addr + (u_cursor) * 8, _phase_u_full);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&u_pk[0])), "=r"(*reinterpret_cast<uint32_t*>(&u_pk[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&u_pk[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&u_pk[(0) + 3]))
                    : "r"((u_panel_addr + u_cursor * 8192 + (unsigned int)(lane_half * 16 / 64 * 8192 + t_row * 128 + lane_half * 16 % 64 * 2 ^ (lane_half * 16 / 64 * 8192 + t_row * 128 + lane_half * 16 % 64 * 2 >> 7 & 7) << 4))) : "memory");
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&u_pk[4])), "=r"(*reinterpret_cast<uint32_t*>(&u_pk[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&u_pk[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&u_pk[(4) + 3]))
                    : "r"((u_panel_addr + u_cursor * 8192 + (unsigned int)((lane_half * 16 + 8) / 64 * 8192 + t_row * 128 + (lane_half * 16 + 8) % 64 * 2 ^ ((lane_half * 16 + 8) / 64 * 8192 + t_row * 128 + (lane_half * 16 + 8) % 64 * 2 >> 7 & 7) << 4))) : "memory");
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&u_pk[8])), "=r"(*reinterpret_cast<uint32_t*>(&u_pk[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&u_pk[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&u_pk[(8) + 3]))
                    : "r"((u_panel_addr + u_cursor * 8192 + (unsigned int)((32 + lane_half * 16) / 64 * 8192 + t_row * 128 + (32 + lane_half * 16) % 64 * 2 ^ ((32 + lane_half * 16) / 64 * 8192 + t_row * 128 + (32 + lane_half * 16) % 64 * 2 >> 7 & 7) << 4))) : "memory");
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&u_pk[12])), "=r"(*reinterpret_cast<uint32_t*>(&u_pk[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&u_pk[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&u_pk[(12) + 3]))
                    : "r"((u_panel_addr + u_cursor * 8192 + (unsigned int)((32 + lane_half * 16 + 8) / 64 * 8192 + t_row * 128 + (32 + lane_half * 16 + 8) % 64 * 2 ^ ((32 + lane_half * 16 + 8) / 64 * 8192 + t_row * 128 + (32 + lane_half * 16 + 8) % 64 * 2 >> 7 & 7) << 4))) : "memory");
                mbarrier_arrive(u_empty_addr + (u_cursor) * 8);
                u_cursor += 1;
                if (u_cursor == 2) { u_cursor = 0; _phase_u_full ^= 1; }
                if (nc > c + 1) {
                    g_next = (long long)(c0 + c + 1) * (long long)num_heads + (long long)head;
                    rs_next = (long long)(row0 + (c + 1) * 64 + t_row) * (long long)num_heads + (long long)head;
                    {
                        uint32_t _scalar_bits_1;
                        asm volatile("ld.global.nc.b32 %0, [%1];"
                            : "=r"(_scalar_bits_1) : "l"((const void*)(gamma + (g_next))) : "memory");
                        next_f[0] = __uint_as_float(_scalar_bits_1);
                    }
                    {
                        uint32_t _scalar_bits_2;
                        asm volatile("ld.global.nc.b32 %0, [%1];"
                            : "=r"(_scalar_bits_2) : "l"((const void*)(rowscale + (rs_next))) : "memory");
                        next_f[1] = __uint_as_float(_scalar_bits_2);
                    }
                    {
                        uint32_t _scalar_bits_3;
                        asm volatile("ld.global.nc.b32 %0, [%1];"
                            : "=r"(_scalar_bits_3) : "l"((const void*)(qscale + (rs_next))) : "memory");
                        next_f[2] = __uint_as_float(_scalar_bits_3);
                    }
                    {
                        uint32_t _scalar_bits_4;
                        asm volatile("ld.global.nc.b32 %0, [%1];"
                            : "=r"(_scalar_bits_4) : "l"((const void*)(chunk_bos + (c0 + c + 1))) : "memory");
                        next_i[0] = (int32_t)_scalar_bits_4;
                    }
                    {
                        uint32_t _scalar_bits_5;
                        asm volatile("ld.global.nc.b32 %0, [%1];"
                            : "=r"(_scalar_bits_5) : "l"((const void*)(chunk_len + (c0 + c + 1))) : "memory");
                        next_i[1] = (int32_t)_scalar_bits_5;
                    }
                }
                if (c > 0) {
                    long long e_obase = ((long long)(bos_prev + t_row) * (long long)num_heads + (long long)head) * 128ll + (long long)vbase;
                    int _parity_o_done = (int)((unsigned int)(int)(c - 1) % 2);
                    mbarrier_wait(o_done_addr, _parity_o_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float qh_0[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qh_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&qh_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&qh_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&qh_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&qh_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&qh_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&qh_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&qh_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&qh_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&qh_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&qh_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&qh_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&qh_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&qh_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&qh_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&qh_0[15]))
                        : "r"(taddr + 128 + (unsigned int)vp_row_base));
                    float av_0[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&av_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&av_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&av_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&av_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&av_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&av_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&av_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&av_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&av_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&av_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&av_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&av_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&av_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&av_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&av_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&av_0[15]))
                        : "r"(taddr + 192 + (unsigned int)vp_row_base));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float ov_0[16];
                    #pragma unroll
                    for (int j_0 = 0; j_0 < 16; j_0++) {
                        ov_0[j_0] = __fmaf_rn(qh_0[j_0], qs_prev, av_0[j_0]);
                    }
                    if (t_row < len_prev) {
                        unsigned int packed_0[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_0[_lp*2 + 0], ov_0[_lp*2+1 + 0]));
                            packed_0[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_0[_lp*2 + 2], ov_0[_lp*2+1 + 2]));
                            packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_0[_lp*2 + 4], ov_0[_lp*2+1 + 4]));
                            packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_0[_lp*2 + 6], ov_0[_lp*2+1 + 6]));
                            packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_0[_lp*2 + 8], ov_0[_lp*2+1 + 8]));
                            packed_0[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_0[_lp*2 + 10], ov_0[_lp*2+1 + 10]));
                            packed_0[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_0[_lp*2 + 12], ov_0[_lp*2+1 + 12]));
                            packed_0[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_0[_lp*2 + 14], ov_0[_lp*2+1 + 14]));
                            packed_0[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_6 = reinterpret_cast<const unsigned*>(packed_0);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(o_out + (e_obase + (long long)(lane_half * 16)))), "r"(_raw_stv8_6[0]), "r"(_raw_stv8_6[1]), "r"(_raw_stv8_6[2]), "r"(_raw_stv8_6[3]), "r"(_raw_stv8_6[4]), "r"(_raw_stv8_6[5]), "r"(_raw_stv8_6[6]), "r"(_raw_stv8_6[7]) : "memory");
                        }
                    }
                    float qh_1[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qh_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&qh_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&qh_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&qh_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&qh_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&qh_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&qh_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&qh_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&qh_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&qh_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&qh_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&qh_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&qh_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&qh_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&qh_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&qh_1[15]))
                        : "r"(taddr + 128 + (unsigned int)vp_row_base + 32));
                    float av_1[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&av_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&av_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&av_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&av_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&av_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&av_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&av_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&av_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&av_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&av_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&av_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&av_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&av_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&av_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&av_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&av_1[15]))
                        : "r"(taddr + 192 + (unsigned int)vp_row_base + 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float ov_1[16];
                    #pragma unroll
                    for (int j_1_1 = 0; j_1_1 < 16; j_1_1++) {
                        ov_1[j_1_1] = __fmaf_rn(qh_1[j_1_1], qs_prev, av_1[j_1_1]);
                    }
                    if (t_row < len_prev) {
                        unsigned int packed_1[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_1[_lp*2 + 0], ov_1[_lp*2+1 + 0]));
                            packed_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_1[_lp*2 + 2], ov_1[_lp*2+1 + 2]));
                            packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_1[_lp*2 + 4], ov_1[_lp*2+1 + 4]));
                            packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_1[_lp*2 + 6], ov_1[_lp*2+1 + 6]));
                            packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_1[_lp*2 + 8], ov_1[_lp*2+1 + 8]));
                            packed_1[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_1[_lp*2 + 10], ov_1[_lp*2+1 + 10]));
                            packed_1[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_1[_lp*2 + 12], ov_1[_lp*2+1 + 12]));
                            packed_1[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_1[_lp*2 + 14], ov_1[_lp*2+1 + 14]));
                            packed_1[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_7 = reinterpret_cast<const unsigned*>(packed_1);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(o_out + (e_obase + (long long)(32 + lane_half * 16)))), "r"(_raw_stv8_7[0]), "r"(_raw_stv8_7[1]), "r"(_raw_stv8_7[2]), "r"(_raw_stv8_7[3]), "r"(_raw_stv8_7[4]), "r"(_raw_stv8_7[5]), "r"(_raw_stv8_7[6]), "r"(_raw_stv8_7[7]) : "memory");
                        }
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(o_free_addr);
                float uv0[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&uv0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(u_pk[_pair]) << 16);
                    (&uv0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(u_pk[_pair]) & 0xffff0000u);
                }
                mbarrier_wait(vp_ready_addr, _phase_vp_ready_0);
                _phase_vp_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float vp_0[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&vp_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&vp_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&vp_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&vp_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&vp_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&vp_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&vp_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&vp_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&vp_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&vp_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&vp_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&vp_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&vp_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&vp_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&vp_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&vp_0[15]))
                    : "r"(taddr + 64 + (unsigned int)vp_row_base));
                float uv1_0[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&uv1_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((u_pk + 8)[_pair]) << 16);
                    (&uv1_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((u_pk + 8)[_pair]) & 0xffff0000u);
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float bvv_0[16];
                #pragma unroll
                for (int j_2_1 = 0; j_2_1 < 16; j_2_1++) {
                    bvv_0[j_2_1] = uv0[j_2_1] - vp_0[j_2_1];
                }
                unsigned int bvv_bf16_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_0[_lp*2 + 0], bvv_0[_lp*2+1 + 0]));
                    bvv_bf16_0[_lp] = *(uint32_t*)&_bf2;
                }
                float bo_0[16];
                #pragma unroll
                for (int j_3_1 = 0; j_3_1 < 16; j_3_1++) {
                    bo_0[j_3_1] = bvv_0[j_3_1] * rs;
                }
                unsigned int bo_bf16_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bo_0[_lp*2 + 0], bo_0[_lp*2+1 + 0]));
                    bo_bf16_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_0 = 0; q_0 < 2; q_0++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv_addr + (unsigned int)((lane_half * 16 + q_0 * 8) / 64 * 8192 + t_row * 128 + (lane_half * 16 + q_0 * 8) % 64 * 2 ^ ((lane_half * 16 + q_0 * 8) / 64 * 8192 + t_row * 128 + (lane_half * 16 + q_0 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bo_bf16_0[q_0 * 4]), "r"(bo_bf16_0[q_0 * 4 + 1]), "r"(bo_bf16_0[q_0 * 4 + 2]), "r"(bo_bf16_0[q_0 * 4 + 3]) : "memory");
                }
                float vp_1[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&vp_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&vp_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&vp_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&vp_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&vp_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&vp_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&vp_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&vp_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&vp_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&vp_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&vp_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&vp_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&vp_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&vp_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&vp_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&vp_1[15]))
                    : "r"(taddr + 64 + (unsigned int)vp_row_base + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float bvv_1[16];
                #pragma unroll
                for (int j_4 = 0; j_4 < 16; j_4++) {
                    bvv_1[j_4] = uv1_0[j_4] - vp_1[j_4];
                }
                unsigned int bvv_bf16_1[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_1[_lp*2 + 0], bvv_1[_lp*2+1 + 0]));
                    bvv_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float bo_1[16];
                #pragma unroll
                for (int j_5 = 0; j_5 < 16; j_5++) {
                    bo_1[j_5] = bvv_1[j_5] * rs;
                }
                unsigned int bo_bf16_1[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bo_1[_lp*2 + 0], bo_1[_lp*2+1 + 0]));
                    bo_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_1_1 = 0; q_1_1 < 2; q_1_1++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv_addr + (unsigned int)((32 + lane_half * 16 + q_1_1 * 8) / 64 * 8192 + t_row * 128 + (32 + lane_half * 16 + q_1_1 * 8) % 64 * 2 ^ ((32 + lane_half * 16 + q_1_1 * 8) / 64 * 8192 + t_row * 128 + (32 + lane_half * 16 + q_1_1 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bo_bf16_1[q_1_1 * 4]), "r"(bo_bf16_1[q_1_1 * 4 + 1]), "r"(bo_bf16_1[q_1_1 * 4 + 2]), "r"(bo_bf16_1[q_1_1 * 4 + 3]) : "memory");
                }
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (int q_2_1 = 0; q_2_1 < 2; q_2_1++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((vn_addr + (unsigned int)((lane_half * 16 + q_2_1 * 8) / 64 * 8192 + t_row * 128 + (lane_half * 16 + q_2_1 * 8) % 64 * 2 ^ ((lane_half * 16 + q_2_1 * 8) / 64 * 8192 + t_row * 128 + (lane_half * 16 + q_2_1 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_bf16_0[q_2_1 * 4]), "r"(bvv_bf16_0[q_2_1 * 4 + 1]), "r"(bvv_bf16_0[q_2_1 * 4 + 2]), "r"(bvv_bf16_0[q_2_1 * 4 + 3]) : "memory");
                }
                #pragma unroll
                for (int q_3_1 = 0; q_3_1 < 2; q_3_1++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((vn_addr + (unsigned int)((32 + lane_half * 16 + q_3_1 * 8) / 64 * 8192 + t_row * 128 + (32 + lane_half * 16 + q_3_1 * 8) % 64 * 2 ^ ((32 + lane_half * 16 + q_3_1 * 8) / 64 * 8192 + t_row * 128 + (32 + lane_half * 16 + q_3_1 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_bf16_1[q_3_1 * 4]), "r"(bvv_bf16_1[q_3_1 * 4 + 1]), "r"(bvv_bf16_1[q_3_1 * 4 + 2]), "r"(bvv_bf16_1[q_3_1 * 4 + 3]) : "memory");
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(bv_ready_addr);
                float qs = qs_next;
                int bos = bos_next;
                int clen = len_next;
                if (nc > c + 1) {
                    decay = next_f[0];
                    rs = next_f[1];
                    qs_next = next_f[2];
                    bos_next = next_i[0];
                    len_next = next_i[1];
                }
                qs_prev = qs;
                bos_prev = bos;
                len_prev = clen;
                mbarrier_wait(h_ready_addr, _phase_h_ready_0);
                _phase_h_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (c + 1 == nc) {
                    long long e_obase_1 = ((long long)(bos_prev + t_row) * (long long)num_heads + (long long)head) * 128ll + (long long)vbase;
                    int _parity_o_done_1 = (int)((unsigned int)(int)c % 2);
                    mbarrier_wait(o_done_addr, _parity_o_done_1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float qh_2[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qh_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&qh_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&qh_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&qh_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&qh_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&qh_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&qh_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&qh_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&qh_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&qh_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&qh_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&qh_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&qh_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&qh_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&qh_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&qh_2[15]))
                        : "r"(taddr + 128 + (unsigned int)vp_row_base));
                    float av_2[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&av_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&av_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&av_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&av_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&av_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&av_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&av_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&av_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&av_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&av_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&av_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&av_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&av_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&av_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&av_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&av_2[15]))
                        : "r"(taddr + 192 + (unsigned int)vp_row_base));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float ov_2[16];
                    #pragma unroll
                    for (int j_6 = 0; j_6 < 16; j_6++) {
                        ov_2[j_6] = __fmaf_rn(qh_2[j_6], qs_prev, av_2[j_6]);
                    }
                    if (t_row < len_prev) {
                        unsigned int packed_2[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2[_lp*2 + 0], ov_2[_lp*2+1 + 0]));
                            packed_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2[_lp*2 + 2], ov_2[_lp*2+1 + 2]));
                            packed_2[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2[_lp*2 + 4], ov_2[_lp*2+1 + 4]));
                            packed_2[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2[_lp*2 + 6], ov_2[_lp*2+1 + 6]));
                            packed_2[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2[_lp*2 + 8], ov_2[_lp*2+1 + 8]));
                            packed_2[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2[_lp*2 + 10], ov_2[_lp*2+1 + 10]));
                            packed_2[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2[_lp*2 + 12], ov_2[_lp*2+1 + 12]));
                            packed_2[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_2[_lp*2 + 14], ov_2[_lp*2+1 + 14]));
                            packed_2[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_8 = reinterpret_cast<const unsigned*>(packed_2);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(o_out + (e_obase_1 + (long long)(lane_half * 16)))), "r"(_raw_stv8_8[0]), "r"(_raw_stv8_8[1]), "r"(_raw_stv8_8[2]), "r"(_raw_stv8_8[3]), "r"(_raw_stv8_8[4]), "r"(_raw_stv8_8[5]), "r"(_raw_stv8_8[6]), "r"(_raw_stv8_8[7]) : "memory");
                        }
                    }
                    float qh_3[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&qh_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&qh_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&qh_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&qh_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&qh_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&qh_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&qh_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&qh_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&qh_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&qh_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&qh_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&qh_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&qh_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&qh_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&qh_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&qh_3[15]))
                        : "r"(taddr + 128 + (unsigned int)vp_row_base + 32));
                    float av_3[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&av_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&av_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&av_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&av_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&av_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&av_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&av_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&av_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&av_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&av_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&av_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&av_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&av_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&av_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&av_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&av_3[15]))
                        : "r"(taddr + 192 + (unsigned int)vp_row_base + 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float ov_3[16];
                    #pragma unroll
                    for (int j_7 = 0; j_7 < 16; j_7++) {
                        ov_3[j_7] = __fmaf_rn(qh_3[j_7], qs_prev, av_3[j_7]);
                    }
                    if (t_row < len_prev) {
                        unsigned int packed_3[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_3[_lp*2 + 0], ov_3[_lp*2+1 + 0]));
                            packed_3[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_3[_lp*2 + 2], ov_3[_lp*2+1 + 2]));
                            packed_3[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_3[_lp*2 + 4], ov_3[_lp*2+1 + 4]));
                            packed_3[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_3[_lp*2 + 6], ov_3[_lp*2+1 + 6]));
                            packed_3[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_3[_lp*2 + 8], ov_3[_lp*2+1 + 8]));
                            packed_3[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_3[_lp*2 + 10], ov_3[_lp*2+1 + 10]));
                            packed_3[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_3[_lp*2 + 12], ov_3[_lp*2+1 + 12]));
                            packed_3[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_3[_lp*2 + 14], ov_3[_lp*2+1 + 14]));
                            packed_3[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_9 = reinterpret_cast<const unsigned*>(packed_3);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(o_out + (e_obase_1 + (long long)(32 + lane_half * 16)))), "r"(_raw_stv8_9[0]), "r"(_raw_stv8_9[1]), "r"(_raw_stv8_9[2]), "r"(_raw_stv8_9[3]), "r"(_raw_stv8_9[4]), "r"(_raw_stv8_9[5]), "r"(_raw_stv8_9[6]), "r"(_raw_stv8_9[7]) : "memory");
                        }
                    }
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
            }
        }
    }
    // ---- Role: load ----
    if (warp == 4) {
        { // load_main
            int hb_1 = blockIdx.x / 2;
            int vbase_1 = blockIdx.x % 2 * 64;
            int head_1 = hb_1 % num_heads;
            int walk_1 = hb_1 / num_heads;
            int seg_1 = walk_seg[walk_1];
            int c0_1 = seq_chunk_start[seg_1];
            int nc_1 = seq_chunk_start[seg_1 + 1] - c0_1;
            int row0_1 = c0_1 * 64;
            int k_head = head_1 / (num_heads / num_qk_heads);
            unsigned int u_lc = 0;
            unsigned int _phase_u_empty = 1;
            unsigned int _phase_w_empty_0 = 1;
            unsigned int _phase_q_empty_0 = 1;
            unsigned int _phase_k_empty_0 = 1;
            unsigned int _phase_aqk_empty_0 = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int c_1 = 0; c_1 < nc_1; c_1++) {
                    int trow = row0_1 + c_1 * 64;
                    int ktok = chunk_bos[c0_1 + c_1];
                    mbarrier_wait(u_empty_addr + (u_lc) * 8, _phase_u_empty);
                    mbarrier_arrive_expect_tx(u_full_addr + (u_lc) * 8, 8192);
                    tma_3d_gmem2smem(u_panel_addr + u_lc * 8192, (&u_tma), vbase_1, head_1, trow, u_full_addr + (u_lc) * 8);
                    u_lc += 1;
                    if (u_lc == 2) { u_lc = 0; _phase_u_empty ^= 1; }
                    mbarrier_wait(w_empty_addr, _phase_w_empty_0);
                    _phase_w_empty_0 ^= 1;
                    mbarrier_arrive_expect_tx(w_full_addr, 16384);
                    #pragma unroll
                    for (int panel = 0; panel < 2; panel++) {
                        tma_3d_gmem2smem(w_panel_addr + (unsigned int)panel * 8192, (&w_tma), panel * 64, head_1, trow, w_full_addr);
                    }
                    mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                    _phase_q_empty_0 ^= 1;
                    mbarrier_arrive_expect_tx(q_full_addr, 16384);
                    #pragma unroll
                    for (int panel_1 = 0; panel_1 < 2; panel_1++) {
                        tma_3d_gmem2smem(q_panel_addr + (unsigned int)panel_1 * 8192, (&q_tma), panel_1 * 64, k_head, ktok, q_full_addr);
                    }
                    mbarrier_wait(k_empty_addr, _phase_k_empty_0);
                    _phase_k_empty_0 ^= 1;
                    mbarrier_arrive_expect_tx(k_full_addr, 16384);
                    #pragma unroll
                    for (int panel_2 = 0; panel_2 < 2; panel_2++) {
                        tma_3d_gmem2smem(k_panel_addr + (unsigned int)panel_2 * 8192, (&k_tma), panel_2 * 64, k_head, ktok, k_full_addr);
                    }
                    mbarrier_wait(aqk_empty_addr, _phase_aqk_empty_0);
                    _phase_aqk_empty_0 ^= 1;
                    mbarrier_arrive_expect_tx(aqk_full_addr, 8192);
                    tma_3d_gmem2smem(aqk_op_addr, (&aqk_tma), 0, head_1, trow, aqk_full_addr);
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 5) {
        { // mma_main
            int hb_2 = blockIdx.x / 2;
            int vbase_2 = blockIdx.x % 2 * 64;
            int head_2 = hb_2 % num_heads;
            int walk_2 = hb_2 / num_heads;
            int seg_2 = walk_seg[walk_2];
            int c0_2 = seq_chunk_start[seg_2];
            int nc_2 = seq_chunk_start[seg_2 + 1] - c0_2;
            int row0_2 = c0_2 * 64;
            unsigned int _phase_w_full_0 = 0;
            unsigned int _phase_hs_ready_0 = 0;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_bv_ready_0 = 0;
            unsigned int _phase_k_full_0 = 0;
            unsigned int _phase_aqk_full_0 = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int c_2 = 0; c_2 < nc_2; c_2++) {
                    mbarrier_wait(w_full_addr, _phase_w_full_0);
                    _phase_w_full_0 ^= 1;
                    mbarrier_wait(hs_ready_addr, _phase_hs_ready_0);
                    _phase_hs_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_0 = ((w_op_addr) >> 4) & 0x7FFF;
                    int _smem_desc_lo_1 = (((hs_mn_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
                    int _smem_desc_lo_0 = ((w_op_addr) >> 4) & 0x3FFF;
                    int _smem_desc_lo_1 = (((hs_mn_addr) >> 4) & 0x3FFF) | 0x4000000;
#endif
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
                    "mov.b32 id, 68224144;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"(tmem_acc_vp), "r"(0) : "memory");
                    tcgen05_commit(vp_ready_addr);
                    tcgen05_commit(w_empty_addr);
                    int _parity_o_free = (int)((unsigned int)(int)c_2 % 2);
                    mbarrier_wait(o_free_addr, _parity_o_free);
                    mbarrier_wait(q_full_addr, _phase_q_full_0);
                    _phase_q_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_2 = ((q_op_addr) >> 4) & 0x7FFF;
                    int _smem_desc_lo_3 = (((hs_mn_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
                    int _smem_desc_lo_2 = ((q_op_addr) >> 4) & 0x3FFF;
                    int _smem_desc_lo_3 = (((hs_mn_addr) >> 4) & 0x3FFF) | 0x4000000;
#endif
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
                    "mov.b32 id, 68224144;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"(tmem_acc_qh), "r"(0) : "memory");
                    tcgen05_commit(q_empty_addr);
                    mbarrier_wait(bv_ready_addr, _phase_bv_ready_0);
                    _phase_bv_ready_0 ^= 1;
                    mbarrier_wait(k_full_addr, _phase_k_full_0);
                    _phase_k_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_4 = (((k_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
                    int _smem_desc_lo_5 = (((bv_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                    int _smem_desc_lo_4 = (((k_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
                    int _smem_desc_lo_5 = (((bv_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
#endif
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
                    "mov.b32 id, 135365776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_4), "r"(_smem_desc_lo_5), "r"(tmem_acc_h), "r"(1) : "memory");
                    tcgen05_commit(h_ready_addr);
                    tcgen05_commit(k_empty_addr);
                    mbarrier_wait(aqk_full_addr, _phase_aqk_full_0);
                    _phase_aqk_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_6 = ((aqk_op_addr) >> 4) & 0x7FFF;
                    int _smem_desc_lo_7 = (((vn_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                    int _smem_desc_lo_6 = ((aqk_op_addr) >> 4) & 0x3FFF;
                    int _smem_desc_lo_7 = (((vn_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
#endif
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
                    "mov.b32 id, 68224144;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_6), "r"(_smem_desc_lo_7), "r"(tmem_acc_av), "r"(0) : "memory");
                    tcgen05_commit(o_done_addr);
                    tcgen05_commit(aqk_empty_addr);
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
