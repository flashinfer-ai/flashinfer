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


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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
kernel_cake_kda_train_7886f9703ccdc5a28f12_kcp(const __grid_constant__ CUtensorMap kg_tma, const __grid_constant__ CUtensorMap qg_tma, const __grid_constant__ CUtensorMap w_tma, const __grid_constant__ CUtensorMap do_tma, const __grid_constant__ CUtensorMap aqk_tma, float* __restrict__ gk, int num_heads, int* __restrict__ seq_chunk_start, int num_seqs, float scale, int* __restrict__ walk_seg, float* __restrict__ dht, int* __restrict__ dht_index, int dht_hg, int dht_n, float* __restrict__ out, int* __restrict__ out_index, int out_hg, int out_n, int tile_stride, __nv_bfloat16* __restrict__ dh_tape, __nv_bfloat16* __restrict__ dv2)
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
    #define kg_full_addr (mbar_base + 0)
    #define kg_empty_addr (mbar_base + 8)
    #define qd_full_addr (mbar_base + 16)
    #define qd_empty_addr (mbar_base + 24)
    #define w_full_addr (mbar_base + 32)
    #define w_empty_addr (mbar_base + 40)
    #define dhs_ready_addr (mbar_base + 48)
    #define dv_ready_addr (mbar_base + 56)
    #define dv_read_addr (mbar_base + 64)
    #define bdv_ready_addr (mbar_base + 72)
    #define dh_ready_addr (mbar_base + 80)
    #define aqk_full_addr (mbar_base + 88)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    __nv_bfloat16* kg_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int kg_panel_addr = smem + 1024;
    __nv_bfloat16* qg_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int qg_panel_addr = smem + 17408;
    __nv_bfloat16* w_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int w_panel_addr = smem + 33792;
    __nv_bfloat16* do_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 50176);
    const int do_panel_addr = smem + 50176;
    __nv_bfloat16* kg_op = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int kg_op_addr = smem + 1024;
    __nv_bfloat16* qg_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int qg_mn_addr = smem + 17408;
    __nv_bfloat16* w_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int w_mn_addr = smem + 33792;
    __nv_bfloat16* do_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 50176);
    const int do_mn_addr = smem + 50176;
    __nv_bfloat16* dhs = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int dhs_addr = smem + 66560;
    __nv_bfloat16* dhs_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int dhs_mn_addr = smem + 66560;
    __nv_bfloat16* bdv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int bdv_addr = smem + 99328;
    __nv_bfloat16* bdv_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int bdv_mn_addr = smem + 99328;
    __nv_bfloat16* aqk_dst = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int aqk_dst_addr = smem + 99328;
    __nv_bfloat16* aqk_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int aqk_mn_addr = smem + 99328;

    // Mbarrier init (12 barriers at smem_raw[0..96))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'kg_pipe' ---
            // kg_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // kg_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'qd_pipe' ---
            // qd_full: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            // qd_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            // --- pipeline 'w_pipe' ---
            // w_full: 1 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            // w_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            // --- pipeline 'one' ---
            // dhs_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 48, 128);
            // dv_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            // dv_read: 1 barriers, init_count=128
            mbarrier_init(smem + 64, 128);
            // bdv_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 72, 128);
            // dh_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // aqk_full: 1 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 96);
    if (warp == 0) {
        int _tmem_hold = smem + 96;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_acc_dh = taddr;
    const unsigned int tmem_acc_dv = taddr + 128;
    const unsigned int tmem_acc_p = taddr + 128;

    // ---- Role: compute ----
    if (warp <= 3) {
        { // compute_main
            int head = blockIdx.x % num_heads;
            int walk = blockIdx.x / num_heads;
            int seg = walk_seg[walk];
            int c0 = seq_chunk_start[seg];
            int nc = seq_chunk_start[seg + 1] - c0;
            int row0 = c0 * 64;
            int warp_in_wg = warp % 4;
            int k_row = warp_in_wg * 32 + lane;
            int m128_row_base = warp_in_wg * 32 << 16;
            int lane_half = lane >> 4;
            int t_row = warp_in_wg * 16 + (lane & 15);
            int m64_row_base = warp_in_wg * 32 << 16;
            float init_row[16];
            int dht_slot = dht_index[seg];
            if (dht_slot >= 0) {
                int dht_tile = (head / dht_hg * dht_n + dht_slot) * dht_hg + head % dht_hg;
                long long dht_base = (long long)dht_tile * (long long)tile_stride * 16384ll + (long long)(k_row * 128);
                #pragma unroll
                for (int blk = 0; blk < 8; blk++) {
                    #pragma unroll
                    for (int q = 0; q < 4; q++) {
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(dht + dht_base + (long long)(blk * 16 + q * 4));
                            init_row[q * 4 + 0] = _v4.x;
                            init_row[q * 4 + 1] = _v4.y;
                            init_row[q * 4 + 2] = _v4.z;
                            init_row[q * 4 + 3] = _v4.w;
                        }
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + (unsigned int)(blk * 16), init_row);
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
                for (int blk_1 = 0; blk_1 < 8; blk_1++) {
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + (unsigned int)(blk_1 * 16), init_row);
                }
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            float gk_pre = 0.0f;
            float decay = 1.0f;
            if (nc > 0) {
                long long gn_first = ((long long)(row0 + nc * 64 - 1) * (long long)num_heads + (long long)head) * 128ll + (long long)k_row;
                gk_pre = gk[gn_first];
                float dh_blk_0[16];
                tmem_ld_x16(&dh_blk_0[0], taddr + (unsigned int)m128_row_base + 0);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int state_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dh_blk_0[_lp*2 + 0], dh_blk_0[_lp*2+1 + 0]));
                    state_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_1 = 0; q_1 < 2; q_1++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)(q_1 * 8 / 64 * 16384 + k_row * 128 + q_1 * 8 % 64 * 2 ^ (q_1 * 8 / 64 * 16384 + k_row * 128 + q_1 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(state_bf16[q_1 * 4]), "r"(state_bf16[q_1 * 4 + 1]), "r"(state_bf16[q_1 * 4 + 2]), "r"(state_bf16[q_1 * 4 + 3]) : "memory");
                }
                float dh_blk_1[16];
                tmem_ld_x16(&dh_blk_1[0], taddr + (unsigned int)m128_row_base + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int state_bf16_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dh_blk_1[_lp*2 + 0], dh_blk_1[_lp*2+1 + 0]));
                    state_bf16_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_2 = 0; q_2 < 2; q_2++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((16 + q_2 * 8) / 64 * 16384 + k_row * 128 + (16 + q_2 * 8) % 64 * 2 ^ ((16 + q_2 * 8) / 64 * 16384 + k_row * 128 + (16 + q_2 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(state_bf16_0[q_2 * 4]), "r"(state_bf16_0[q_2 * 4 + 1]), "r"(state_bf16_0[q_2 * 4 + 2]), "r"(state_bf16_0[q_2 * 4 + 3]) : "memory");
                }
                float dh_blk_2[16];
                tmem_ld_x16(&dh_blk_2[0], taddr + (unsigned int)m128_row_base + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int state_bf16_1[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dh_blk_2[_lp*2 + 0], dh_blk_2[_lp*2+1 + 0]));
                    state_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_3 = 0; q_3 < 2; q_3++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((32 + q_3 * 8) / 64 * 16384 + k_row * 128 + (32 + q_3 * 8) % 64 * 2 ^ ((32 + q_3 * 8) / 64 * 16384 + k_row * 128 + (32 + q_3 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(state_bf16_1[q_3 * 4]), "r"(state_bf16_1[q_3 * 4 + 1]), "r"(state_bf16_1[q_3 * 4 + 2]), "r"(state_bf16_1[q_3 * 4 + 3]) : "memory");
                }
                float dh_blk_3[16];
                tmem_ld_x16(&dh_blk_3[0], taddr + (unsigned int)m128_row_base + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int state_bf16_2[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dh_blk_3[_lp*2 + 0], dh_blk_3[_lp*2+1 + 0]));
                    state_bf16_2[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_4 = 0; q_4 < 2; q_4++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((48 + q_4 * 8) / 64 * 16384 + k_row * 128 + (48 + q_4 * 8) % 64 * 2 ^ ((48 + q_4 * 8) / 64 * 16384 + k_row * 128 + (48 + q_4 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(state_bf16_2[q_4 * 4]), "r"(state_bf16_2[q_4 * 4 + 1]), "r"(state_bf16_2[q_4 * 4 + 2]), "r"(state_bf16_2[q_4 * 4 + 3]) : "memory");
                }
                float dh_blk_4[16];
                tmem_ld_x16(&dh_blk_4[0], taddr + (unsigned int)m128_row_base + 64);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int state_bf16_3[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dh_blk_4[_lp*2 + 0], dh_blk_4[_lp*2+1 + 0]));
                    state_bf16_3[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_5 = 0; q_5 < 2; q_5++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((64 + q_5 * 8) / 64 * 16384 + k_row * 128 + (64 + q_5 * 8) % 64 * 2 ^ ((64 + q_5 * 8) / 64 * 16384 + k_row * 128 + (64 + q_5 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(state_bf16_3[q_5 * 4]), "r"(state_bf16_3[q_5 * 4 + 1]), "r"(state_bf16_3[q_5 * 4 + 2]), "r"(state_bf16_3[q_5 * 4 + 3]) : "memory");
                }
                float dh_blk_5[16];
                tmem_ld_x16(&dh_blk_5[0], taddr + (unsigned int)m128_row_base + 80);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int state_bf16_4[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dh_blk_5[_lp*2 + 0], dh_blk_5[_lp*2+1 + 0]));
                    state_bf16_4[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_6 = 0; q_6 < 2; q_6++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((80 + q_6 * 8) / 64 * 16384 + k_row * 128 + (80 + q_6 * 8) % 64 * 2 ^ ((80 + q_6 * 8) / 64 * 16384 + k_row * 128 + (80 + q_6 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(state_bf16_4[q_6 * 4]), "r"(state_bf16_4[q_6 * 4 + 1]), "r"(state_bf16_4[q_6 * 4 + 2]), "r"(state_bf16_4[q_6 * 4 + 3]) : "memory");
                }
                float dh_blk_6[16];
                tmem_ld_x16(&dh_blk_6[0], taddr + (unsigned int)m128_row_base + 96);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int state_bf16_5[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dh_blk_6[_lp*2 + 0], dh_blk_6[_lp*2+1 + 0]));
                    state_bf16_5[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_7 = 0; q_7 < 2; q_7++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((96 + q_7 * 8) / 64 * 16384 + k_row * 128 + (96 + q_7 * 8) % 64 * 2 ^ ((96 + q_7 * 8) / 64 * 16384 + k_row * 128 + (96 + q_7 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(state_bf16_5[q_7 * 4]), "r"(state_bf16_5[q_7 * 4 + 1]), "r"(state_bf16_5[q_7 * 4 + 2]), "r"(state_bf16_5[q_7 * 4 + 3]) : "memory");
                }
                float dh_blk_7[16];
                tmem_ld_x16(&dh_blk_7[0], taddr + (unsigned int)m128_row_base + 112);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int state_bf16_6[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dh_blk_7[_lp*2 + 0], dh_blk_7[_lp*2+1 + 0]));
                    state_bf16_6[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_8 = 0; q_8 < 2; q_8++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((112 + q_8 * 8) / 64 * 16384 + k_row * 128 + (112 + q_8 * 8) % 64 * 2 ^ ((112 + q_8 * 8) / 64 * 16384 + k_row * 128 + (112 + q_8 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(state_bf16_6[q_8 * 4]), "r"(state_bf16_6[q_8 * 4 + 1]), "r"(state_bf16_6[q_8 * 4 + 2]), "r"(state_bf16_6[q_8 * 4 + 3]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(dhs_ready_addr);
                float _exp2_1 = approx_exp2(gk_pre);
                decay = _exp2_1;
                float dh_blk_8[16];
                tmem_ld_x16(&dh_blk_8[0], taddr + (unsigned int)m128_row_base + 0);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float sc_0[16];
                #pragma unroll
                for (int j_0 = 0; j_0 < 16; j_0++) {
                    sc_0[j_0] = dh_blk_8[j_0] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 0, sc_0);
                float dh_blk_9[16];
                tmem_ld_x16(&dh_blk_9[0], taddr + (unsigned int)m128_row_base + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float sc_1[16];
                #pragma unroll
                for (int j_1 = 0; j_1 < 16; j_1++) {
                    sc_1[j_1] = dh_blk_9[j_1] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 16, sc_1);
                float dh_blk_10[16];
                tmem_ld_x16(&dh_blk_10[0], taddr + (unsigned int)m128_row_base + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float sc_2[16];
                #pragma unroll
                for (int j_2 = 0; j_2 < 16; j_2++) {
                    sc_2[j_2] = dh_blk_10[j_2] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 32, sc_2);
                float dh_blk_11[16];
                tmem_ld_x16(&dh_blk_11[0], taddr + (unsigned int)m128_row_base + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float sc_3[16];
                #pragma unroll
                for (int j_3 = 0; j_3 < 16; j_3++) {
                    sc_3[j_3] = dh_blk_11[j_3] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 48, sc_3);
                float dh_blk_12[16];
                tmem_ld_x16(&dh_blk_12[0], taddr + (unsigned int)m128_row_base + 64);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float sc_4[16];
                #pragma unroll
                for (int j_4 = 0; j_4 < 16; j_4++) {
                    sc_4[j_4] = dh_blk_12[j_4] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 64, sc_4);
                float dh_blk_13[16];
                tmem_ld_x16(&dh_blk_13[0], taddr + (unsigned int)m128_row_base + 80);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float sc_5[16];
                #pragma unroll
                for (int j_5 = 0; j_5 < 16; j_5++) {
                    sc_5[j_5] = dh_blk_13[j_5] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 80, sc_5);
                float dh_blk_14[16];
                tmem_ld_x16(&dh_blk_14[0], taddr + (unsigned int)m128_row_base + 96);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float sc_6[16];
                #pragma unroll
                for (int j_6 = 0; j_6 < 16; j_6++) {
                    sc_6[j_6] = dh_blk_14[j_6] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 96, sc_6);
                float dh_blk_15[16];
                tmem_ld_x16(&dh_blk_15[0], taddr + (unsigned int)m128_row_base + 112);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float sc_7[16];
                #pragma unroll
                for (int j_7 = 0; j_7 < 16; j_7++) {
                    sc_7[j_7] = dh_blk_15[j_7] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 112, sc_7);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            }
            unsigned int _phase_dv_ready_0 = 0;
            unsigned int _phase_dh_ready_0 = 0;
            #pragma unroll 1
            for (int rc = 0; rc < nc; rc++) {
                int c = nc - 1 - rc;
                mbarrier_wait(dv_ready_addr, _phase_dv_ready_0);
                _phase_dv_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                long long t_base = ((long long)(row0 + c * 64 + t_row) * (long long)num_heads + (long long)head) * 128ll;
                float dv_a_0[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&dv_a_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&dv_a_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&dv_a_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&dv_a_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&dv_a_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&dv_a_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&dv_a_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&dv_a_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&dv_a_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&dv_a_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&dv_a_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&dv_a_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&dv_a_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&dv_a_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&dv_a_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&dv_a_0[15]))
                    : "r"(taddr + 128 + (unsigned int)m64_row_base + 0));
                float dv_b_0[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&dv_b_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&dv_b_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&dv_b_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&dv_b_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&dv_b_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&dv_b_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&dv_b_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&dv_b_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&dv_b_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&dv_b_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&dv_b_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&dv_b_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&dv_b_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&dv_b_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&dv_b_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&dv_b_0[15]))
                    : "r"(taddr + 128 + (unsigned int)m64_row_base + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                int col = lane_half * 16;
                unsigned int bdvv_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dv_a_0[_lp*2 + 0], dv_a_0[_lp*2+1 + 0]));
                    bdvv_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_9 = 0; q_9 < 2; q_9++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bdv_addr + (unsigned int)((col + q_9 * 8) / 64 * 8192 + t_row * 128 + (col + q_9 * 8) % 64 * 2 ^ ((col + q_9 * 8) / 64 * 8192 + t_row * 128 + (col + q_9 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bdvv_bf16[q_9 * 4]), "r"(bdvv_bf16[q_9 * 4 + 1]), "r"(bdvv_bf16[q_9 * 4 + 2]), "r"(bdvv_bf16[q_9 * 4 + 3]) : "memory");
                }
                int col_0 = 32 + lane_half * 16;
                unsigned int bdvv_bf16_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dv_b_0[_lp*2 + 0], dv_b_0[_lp*2+1 + 0]));
                    bdvv_bf16_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_10 = 0; q_10 < 2; q_10++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bdv_addr + (unsigned int)((col_0 + q_10 * 8) / 64 * 8192 + t_row * 128 + (col_0 + q_10 * 8) % 64 * 2 ^ ((col_0 + q_10 * 8) / 64 * 8192 + t_row * 128 + (col_0 + q_10 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bdvv_bf16_0[q_10 * 4]), "r"(bdvv_bf16_0[q_10 * 4 + 1]), "r"(bdvv_bf16_0[q_10 * 4 + 2]), "r"(bdvv_bf16_0[q_10 * 4 + 3]) : "memory");
                }
                float dv_a_1[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&dv_a_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&dv_a_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&dv_a_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&dv_a_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&dv_a_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&dv_a_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&dv_a_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&dv_a_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&dv_a_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&dv_a_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&dv_a_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&dv_a_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&dv_a_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&dv_a_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&dv_a_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&dv_a_1[15]))
                    : "r"(taddr + 128 + (unsigned int)m64_row_base + 64));
                float dv_b_1[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&dv_b_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&dv_b_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&dv_b_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&dv_b_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&dv_b_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&dv_b_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&dv_b_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&dv_b_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&dv_b_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&dv_b_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&dv_b_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&dv_b_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&dv_b_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&dv_b_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&dv_b_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&dv_b_1[15]))
                    : "r"(taddr + 128 + (unsigned int)m64_row_base + 96));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(dv_read_addr);
                int col_1 = 64 + lane_half * 16;
                unsigned int bdvv_bf16_1[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dv_a_1[_lp*2 + 0], dv_a_1[_lp*2+1 + 0]));
                    bdvv_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_11 = 0; q_11 < 2; q_11++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bdv_addr + (unsigned int)((col_1 + q_11 * 8) / 64 * 8192 + t_row * 128 + (col_1 + q_11 * 8) % 64 * 2 ^ ((col_1 + q_11 * 8) / 64 * 8192 + t_row * 128 + (col_1 + q_11 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bdvv_bf16_1[q_11 * 4]), "r"(bdvv_bf16_1[q_11 * 4 + 1]), "r"(bdvv_bf16_1[q_11 * 4 + 2]), "r"(bdvv_bf16_1[q_11 * 4 + 3]) : "memory");
                }
                int col_2 = 96 + lane_half * 16;
                unsigned int bdvv_bf16_2[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dv_b_1[_lp*2 + 0], dv_b_1[_lp*2+1 + 0]));
                    bdvv_bf16_2[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_12 = 0; q_12 < 2; q_12++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bdv_addr + (unsigned int)((col_2 + q_12 * 8) / 64 * 8192 + t_row * 128 + (col_2 + q_12 * 8) % 64 * 2 ^ ((col_2 + q_12 * 8) / 64 * 8192 + t_row * 128 + (col_2 + q_12 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bdvv_bf16_2[q_12 * 4]), "r"(bdvv_bf16_2[q_12 * 4 + 1]), "r"(bdvv_bf16_2[q_12 * 4 + 2]), "r"(bdvv_bf16_2[q_12 * 4 + 3]) : "memory");
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(bdv_ready_addr);
                if (nc > rc + 1) {
                    long long gn_next = ((long long)(row0 + c * 64 - 1) * (long long)num_heads + (long long)head) * 128ll + (long long)k_row;
                    gk_pre = gk[gn_next];
                }
                mbarrier_wait(dh_ready_addr, _phase_dh_ready_0);
                _phase_dh_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (nc > rc + 1) {
                    float _exp2_2 = approx_exp2(gk_pre);
                    decay = _exp2_2;
                    float dh_blk[16];
                    tmem_ld_x16(&dh_blk[0], taddr + (unsigned int)m128_row_base + 0);
                    float p_blk[16];
                    tmem_ld_x16(&p_blk[0], taddr + 128 + (unsigned int)m128_row_base + 0);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float upd[16];
                    #pragma unroll
                    for (int j = 0; j < 16; j++) {
                        upd[j] = __fmaf_rn(p_blk[j], scale, dh_blk[j]);
                    }
                    unsigned int upd_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 0], upd[_lp*2+1 + 0]));
                        upd_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_13 = 0; q_13 < 2; q_13++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)(q_13 * 8 / 64 * 16384 + k_row * 128 + q_13 * 8 % 64 * 2 ^ (q_13 * 8 / 64 * 16384 + k_row * 128 + q_13 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(upd_bf16[q_13 * 4]), "r"(upd_bf16[q_13 * 4 + 1]), "r"(upd_bf16[q_13 * 4 + 2]), "r"(upd_bf16[q_13 * 4 + 3]) : "memory");
                    }
                    float sc[16];
                    #pragma unroll
                    for (int j_8 = 0; j_8 < 16; j_8++) {
                        sc[j_8] = upd[j_8] * decay;
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 0, sc);
                    float dh_blk_16[16];
                    tmem_ld_x16(&dh_blk_16[0], taddr + (unsigned int)m128_row_base + 16);
                    float p_blk_0[16];
                    tmem_ld_x16(&p_blk_0[0], taddr + 128 + (unsigned int)m128_row_base + 16);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float upd_0[16];
                    #pragma unroll
                    for (int j_9 = 0; j_9 < 16; j_9++) {
                        upd_0[j_9] = __fmaf_rn(p_blk_0[j_9], scale, dh_blk_16[j_9]);
                    }
                    unsigned int upd_bf16_0[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd_0[_lp*2 + 0], upd_0[_lp*2+1 + 0]));
                        upd_bf16_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_14 = 0; q_14 < 2; q_14++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((16 + q_14 * 8) / 64 * 16384 + k_row * 128 + (16 + q_14 * 8) % 64 * 2 ^ ((16 + q_14 * 8) / 64 * 16384 + k_row * 128 + (16 + q_14 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(upd_bf16_0[q_14 * 4]), "r"(upd_bf16_0[q_14 * 4 + 1]), "r"(upd_bf16_0[q_14 * 4 + 2]), "r"(upd_bf16_0[q_14 * 4 + 3]) : "memory");
                    }
                    float sc_8[16];
                    #pragma unroll
                    for (int j_10 = 0; j_10 < 16; j_10++) {
                        sc_8[j_10] = upd_0[j_10] * decay;
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 16, sc_8);
                    float dh_blk_17[16];
                    tmem_ld_x16(&dh_blk_17[0], taddr + (unsigned int)m128_row_base + 32);
                    float p_blk_1[16];
                    tmem_ld_x16(&p_blk_1[0], taddr + 128 + (unsigned int)m128_row_base + 32);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float upd_1[16];
                    #pragma unroll
                    for (int j_11 = 0; j_11 < 16; j_11++) {
                        upd_1[j_11] = __fmaf_rn(p_blk_1[j_11], scale, dh_blk_17[j_11]);
                    }
                    unsigned int upd_bf16_1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd_1[_lp*2 + 0], upd_1[_lp*2+1 + 0]));
                        upd_bf16_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_15 = 0; q_15 < 2; q_15++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((32 + q_15 * 8) / 64 * 16384 + k_row * 128 + (32 + q_15 * 8) % 64 * 2 ^ ((32 + q_15 * 8) / 64 * 16384 + k_row * 128 + (32 + q_15 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(upd_bf16_1[q_15 * 4]), "r"(upd_bf16_1[q_15 * 4 + 1]), "r"(upd_bf16_1[q_15 * 4 + 2]), "r"(upd_bf16_1[q_15 * 4 + 3]) : "memory");
                    }
                    float sc_9[16];
                    #pragma unroll
                    for (int j_12 = 0; j_12 < 16; j_12++) {
                        sc_9[j_12] = upd_1[j_12] * decay;
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 32, sc_9);
                    float dh_blk_18[16];
                    tmem_ld_x16(&dh_blk_18[0], taddr + (unsigned int)m128_row_base + 48);
                    float p_blk_2[16];
                    tmem_ld_x16(&p_blk_2[0], taddr + 128 + (unsigned int)m128_row_base + 48);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float upd_2[16];
                    #pragma unroll
                    for (int j_13 = 0; j_13 < 16; j_13++) {
                        upd_2[j_13] = __fmaf_rn(p_blk_2[j_13], scale, dh_blk_18[j_13]);
                    }
                    unsigned int upd_bf16_2[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd_2[_lp*2 + 0], upd_2[_lp*2+1 + 0]));
                        upd_bf16_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_16 = 0; q_16 < 2; q_16++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((48 + q_16 * 8) / 64 * 16384 + k_row * 128 + (48 + q_16 * 8) % 64 * 2 ^ ((48 + q_16 * 8) / 64 * 16384 + k_row * 128 + (48 + q_16 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(upd_bf16_2[q_16 * 4]), "r"(upd_bf16_2[q_16 * 4 + 1]), "r"(upd_bf16_2[q_16 * 4 + 2]), "r"(upd_bf16_2[q_16 * 4 + 3]) : "memory");
                    }
                    float sc_10[16];
                    #pragma unroll
                    for (int j_14 = 0; j_14 < 16; j_14++) {
                        sc_10[j_14] = upd_2[j_14] * decay;
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 48, sc_10);
                    float dh_blk_19[16];
                    tmem_ld_x16(&dh_blk_19[0], taddr + (unsigned int)m128_row_base + 64);
                    float p_blk_3[16];
                    tmem_ld_x16(&p_blk_3[0], taddr + 128 + (unsigned int)m128_row_base + 64);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float upd_3[16];
                    #pragma unroll
                    for (int j_15 = 0; j_15 < 16; j_15++) {
                        upd_3[j_15] = __fmaf_rn(p_blk_3[j_15], scale, dh_blk_19[j_15]);
                    }
                    unsigned int upd_bf16_3[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd_3[_lp*2 + 0], upd_3[_lp*2+1 + 0]));
                        upd_bf16_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_17 = 0; q_17 < 2; q_17++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((64 + q_17 * 8) / 64 * 16384 + k_row * 128 + (64 + q_17 * 8) % 64 * 2 ^ ((64 + q_17 * 8) / 64 * 16384 + k_row * 128 + (64 + q_17 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(upd_bf16_3[q_17 * 4]), "r"(upd_bf16_3[q_17 * 4 + 1]), "r"(upd_bf16_3[q_17 * 4 + 2]), "r"(upd_bf16_3[q_17 * 4 + 3]) : "memory");
                    }
                    float sc_11[16];
                    #pragma unroll
                    for (int j_16 = 0; j_16 < 16; j_16++) {
                        sc_11[j_16] = upd_3[j_16] * decay;
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 64, sc_11);
                    float dh_blk_20[16];
                    tmem_ld_x16(&dh_blk_20[0], taddr + (unsigned int)m128_row_base + 80);
                    float p_blk_4[16];
                    tmem_ld_x16(&p_blk_4[0], taddr + 128 + (unsigned int)m128_row_base + 80);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float upd_4[16];
                    #pragma unroll
                    for (int j_17 = 0; j_17 < 16; j_17++) {
                        upd_4[j_17] = __fmaf_rn(p_blk_4[j_17], scale, dh_blk_20[j_17]);
                    }
                    unsigned int upd_bf16_4[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd_4[_lp*2 + 0], upd_4[_lp*2+1 + 0]));
                        upd_bf16_4[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_18 = 0; q_18 < 2; q_18++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((80 + q_18 * 8) / 64 * 16384 + k_row * 128 + (80 + q_18 * 8) % 64 * 2 ^ ((80 + q_18 * 8) / 64 * 16384 + k_row * 128 + (80 + q_18 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(upd_bf16_4[q_18 * 4]), "r"(upd_bf16_4[q_18 * 4 + 1]), "r"(upd_bf16_4[q_18 * 4 + 2]), "r"(upd_bf16_4[q_18 * 4 + 3]) : "memory");
                    }
                    float sc_12[16];
                    #pragma unroll
                    for (int j_18 = 0; j_18 < 16; j_18++) {
                        sc_12[j_18] = upd_4[j_18] * decay;
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 80, sc_12);
                    float dh_blk_21[16];
                    tmem_ld_x16(&dh_blk_21[0], taddr + (unsigned int)m128_row_base + 96);
                    float p_blk_5[16];
                    tmem_ld_x16(&p_blk_5[0], taddr + 128 + (unsigned int)m128_row_base + 96);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float upd_5[16];
                    #pragma unroll
                    for (int j_19 = 0; j_19 < 16; j_19++) {
                        upd_5[j_19] = __fmaf_rn(p_blk_5[j_19], scale, dh_blk_21[j_19]);
                    }
                    unsigned int upd_bf16_5[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd_5[_lp*2 + 0], upd_5[_lp*2+1 + 0]));
                        upd_bf16_5[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_19 = 0; q_19 < 2; q_19++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((96 + q_19 * 8) / 64 * 16384 + k_row * 128 + (96 + q_19 * 8) % 64 * 2 ^ ((96 + q_19 * 8) / 64 * 16384 + k_row * 128 + (96 + q_19 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(upd_bf16_5[q_19 * 4]), "r"(upd_bf16_5[q_19 * 4 + 1]), "r"(upd_bf16_5[q_19 * 4 + 2]), "r"(upd_bf16_5[q_19 * 4 + 3]) : "memory");
                    }
                    float sc_13[16];
                    #pragma unroll
                    for (int j_20 = 0; j_20 < 16; j_20++) {
                        sc_13[j_20] = upd_5[j_20] * decay;
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 96, sc_13);
                    float dh_blk_22[16];
                    tmem_ld_x16(&dh_blk_22[0], taddr + (unsigned int)m128_row_base + 112);
                    float p_blk_6[16];
                    tmem_ld_x16(&p_blk_6[0], taddr + 128 + (unsigned int)m128_row_base + 112);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float upd_6[16];
                    #pragma unroll
                    for (int j_21 = 0; j_21 < 16; j_21++) {
                        upd_6[j_21] = __fmaf_rn(p_blk_6[j_21], scale, dh_blk_22[j_21]);
                    }
                    unsigned int upd_bf16_6[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd_6[_lp*2 + 0], upd_6[_lp*2+1 + 0]));
                        upd_bf16_6[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_20 = 0; q_20 < 2; q_20++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((112 + q_20 * 8) / 64 * 16384 + k_row * 128 + (112 + q_20 * 8) % 64 * 2 ^ ((112 + q_20 * 8) / 64 * 16384 + k_row * 128 + (112 + q_20 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(upd_bf16_6[q_20 * 4]), "r"(upd_bf16_6[q_20 * 4 + 1]), "r"(upd_bf16_6[q_20 * 4 + 2]), "r"(upd_bf16_6[q_20 * 4 + 3]) : "memory");
                    }
                    float sc_14[16];
                    #pragma unroll
                    for (int j_22 = 0; j_22 < 16; j_22++) {
                        sc_14[j_22] = upd_6[j_22] * decay;
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 112, sc_14);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(dhs_ready_addr);
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            }
            if (nc > 0) {
                float dh_blk_23[16];
                tmem_ld_x16(&dh_blk_23[0], taddr + (unsigned int)m128_row_base + 0);
                float p_blk_7[16];
                tmem_ld_x16(&p_blk_7[0], taddr + 128 + (unsigned int)m128_row_base + 0);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float upd_7[16];
                #pragma unroll
                for (int j_23 = 0; j_23 < 16; j_23++) {
                    upd_7[j_23] = __fmaf_rn(p_blk_7[j_23], scale, dh_blk_23[j_23]);
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 0, upd_7);
                float dh_blk_23_1[16];
                tmem_ld_x16(&dh_blk_23_1[0], taddr + (unsigned int)m128_row_base + 16);
                float p_blk_7_1[16];
                tmem_ld_x16(&p_blk_7_1[0], taddr + 128 + (unsigned int)m128_row_base + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float upd_7_1[16];
                #pragma unroll
                for (int j_24 = 0; j_24 < 16; j_24++) {
                    upd_7_1[j_24] = __fmaf_rn(p_blk_7_1[j_24], scale, dh_blk_23_1[j_24]);
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 16, upd_7_1);
                float dh_blk_24[16];
                tmem_ld_x16(&dh_blk_24[0], taddr + (unsigned int)m128_row_base + 32);
                float p_blk_8[16];
                tmem_ld_x16(&p_blk_8[0], taddr + 128 + (unsigned int)m128_row_base + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float upd_8[16];
                #pragma unroll
                for (int j_25 = 0; j_25 < 16; j_25++) {
                    upd_8[j_25] = __fmaf_rn(p_blk_8[j_25], scale, dh_blk_24[j_25]);
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 32, upd_8);
                float dh_blk_25[16];
                tmem_ld_x16(&dh_blk_25[0], taddr + (unsigned int)m128_row_base + 48);
                float p_blk_9[16];
                tmem_ld_x16(&p_blk_9[0], taddr + 128 + (unsigned int)m128_row_base + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float upd_9[16];
                #pragma unroll
                for (int j_26 = 0; j_26 < 16; j_26++) {
                    upd_9[j_26] = __fmaf_rn(p_blk_9[j_26], scale, dh_blk_25[j_26]);
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 48, upd_9);
                float dh_blk_26[16];
                tmem_ld_x16(&dh_blk_26[0], taddr + (unsigned int)m128_row_base + 64);
                float p_blk_10[16];
                tmem_ld_x16(&p_blk_10[0], taddr + 128 + (unsigned int)m128_row_base + 64);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float upd_10[16];
                #pragma unroll
                for (int j_27 = 0; j_27 < 16; j_27++) {
                    upd_10[j_27] = __fmaf_rn(p_blk_10[j_27], scale, dh_blk_26[j_27]);
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 64, upd_10);
                float dh_blk_27[16];
                tmem_ld_x16(&dh_blk_27[0], taddr + (unsigned int)m128_row_base + 80);
                float p_blk_11[16];
                tmem_ld_x16(&p_blk_11[0], taddr + 128 + (unsigned int)m128_row_base + 80);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float upd_11[16];
                #pragma unroll
                for (int j_28 = 0; j_28 < 16; j_28++) {
                    upd_11[j_28] = __fmaf_rn(p_blk_11[j_28], scale, dh_blk_27[j_28]);
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 80, upd_11);
                float dh_blk_28[16];
                tmem_ld_x16(&dh_blk_28[0], taddr + (unsigned int)m128_row_base + 96);
                float p_blk_12[16];
                tmem_ld_x16(&p_blk_12[0], taddr + 128 + (unsigned int)m128_row_base + 96);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float upd_12[16];
                #pragma unroll
                for (int j_29 = 0; j_29 < 16; j_29++) {
                    upd_12[j_29] = __fmaf_rn(p_blk_12[j_29], scale, dh_blk_28[j_29]);
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 96, upd_12);
                float dh_blk_29[16];
                tmem_ld_x16(&dh_blk_29[0], taddr + (unsigned int)m128_row_base + 112);
                float p_blk_13[16];
                tmem_ld_x16(&p_blk_13[0], taddr + 128 + (unsigned int)m128_row_base + 112);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float upd_13[16];
                #pragma unroll
                for (int j_30 = 0; j_30 < 16; j_30++) {
                    upd_13[j_30] = __fmaf_rn(p_blk_13[j_30], scale, dh_blk_29[j_30]);
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 112, upd_13);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            }
            int out_slot = out_index[seg];
            if (out_slot >= 0) {
                int out_tile = (head / out_hg * out_n + out_slot) * out_hg + head % out_hg;
                long long out_base = (long long)out_tile * (long long)tile_stride * 16384ll + (long long)(k_row * 128);
                #pragma unroll
                for (int blk_2 = 0; blk_2 < 8; blk_2++) {
                    float dh_out[16];
                    tmem_ld_x16(&dh_out[0], taddr + (unsigned int)m128_row_base + (unsigned int)(blk_2 * 16));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int q_21 = 0; q_21 < 4; q_21++) {
                        {
                            float4 _v4 = make_float4(dh_out[q_21 * 4 + 0], dh_out[q_21 * 4 + 1], dh_out[q_21 * 4 + 2], dh_out[q_21 * 4 + 3]);
                            *reinterpret_cast<float4*>(out + out_base + (long long)(blk_2 * 16 + q_21 * 4)) = _v4;
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
            int head_1 = blockIdx.x % num_heads;
            int walk_1 = blockIdx.x / num_heads;
            int seg_1 = walk_seg[walk_1];
            int c0_1 = seq_chunk_start[seg_1];
            int nc_1 = seq_chunk_start[seg_1 + 1] - c0_1;
            int row0_1 = c0_1 * 64;
            unsigned int _phase_kg_empty = 1;
            unsigned int _phase_qd_empty = 1;
            unsigned int _phase_w_empty = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int rc_1 = 0; rc_1 < nc_1; rc_1++) {
                    int trow = row0_1 + (nc_1 - 1 - rc_1) * 64;
                    mbarrier_wait(kg_empty_addr, _phase_kg_empty);
                    mbarrier_arrive_expect_tx(kg_full_addr, 16384);
                    #pragma unroll
                    for (int panel = 0; panel < 2; panel++) {
                        tma_3d_gmem2smem(kg_panel_addr + (unsigned int)(panel * 8192), (&kg_tma), panel * 64, head_1, trow, kg_full_addr);
                    }
                    _phase_kg_empty ^= 1;
                    mbarrier_wait(qd_empty_addr, _phase_qd_empty);
                    mbarrier_arrive_expect_tx(qd_full_addr, 32768);
                    #pragma unroll
                    for (int panel_1 = 0; panel_1 < 2; panel_1++) {
                        tma_3d_gmem2smem(qg_panel_addr + (unsigned int)(panel_1 * 8192), (&qg_tma), panel_1 * 64, head_1, trow, qd_full_addr);
                        tma_3d_gmem2smem(do_panel_addr + (unsigned int)(panel_1 * 8192), (&do_tma), panel_1 * 64, head_1, trow, qd_full_addr);
                    }
                    _phase_qd_empty ^= 1;
                    mbarrier_wait(w_empty_addr, _phase_w_empty);
                    mbarrier_arrive_expect_tx(w_full_addr, 16384);
                    #pragma unroll
                    for (int panel_2 = 0; panel_2 < 2; panel_2++) {
                        tma_3d_gmem2smem(w_panel_addr + (unsigned int)(panel_2 * 8192), (&w_tma), panel_2 * 64, head_1, trow, w_full_addr);
                    }
                    _phase_w_empty ^= 1;
                    mbarrier_arrive_expect_tx(aqk_full_addr, 8192);
                    tma_3d_gmem2smem(aqk_dst_addr, (&aqk_tma), 0, head_1, trow, aqk_full_addr);
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 5) {
        { // mma_main
            int head_2 = blockIdx.x % num_heads;
            int walk_2 = blockIdx.x / num_heads;
            int seg_2 = walk_seg[walk_2];
            int c0_2 = seq_chunk_start[seg_2];
            int nc_2 = seq_chunk_start[seg_2 + 1] - c0_2;
            int row0_2 = c0_2 * 64;
            unsigned int _phase_kg_full = 0;
            unsigned int _phase_dhs_ready_0 = 0;
            unsigned int _phase_qd_full = 0;
            unsigned int _phase_aqk_full_0 = 0;
            unsigned int _phase_dv_read_0 = 0;
            unsigned int _phase_bdv_ready_0 = 0;
            unsigned int _phase_w_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int rc_2 = 0; rc_2 < nc_2; rc_2++) {
                    mbarrier_wait(kg_full_addr, _phase_kg_full);
                    mbarrier_wait(dhs_ready_addr, _phase_dhs_ready_0);
                    _phase_dhs_ready_0 ^= 1;
                    mbarrier_wait(qd_full_addr, _phase_qd_full);
                    mbarrier_wait(aqk_full_addr, _phase_aqk_full_0);
                    _phase_aqk_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_0 = (((aqk_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
                    int _smem_desc_lo_1 = (((do_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                    int _smem_desc_lo_0 = (((aqk_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
                    int _smem_desc_lo_1 = (((do_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    "mov.b32 id, 69305488;\n\t"
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
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"(tmem_acc_dv), "r"(0) : "memory");
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_2 = ((kg_op_addr) >> 4) & 0x7FFF;
                    int _smem_desc_lo_3 = (((dhs_mn_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
                    int _smem_desc_lo_2 = ((kg_op_addr) >> 4) & 0x3FFF;
                    int _smem_desc_lo_3 = (((dhs_mn_addr) >> 4) & 0x3FFF) | 0x4000000;
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
                    "mov.b32 id, 69272720;\n\t"
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
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"(tmem_acc_dv), "r"(1) : "memory");
                    tcgen05_commit(dv_ready_addr);
                    tcgen05_commit(kg_empty_addr);
                    _phase_kg_full ^= 1;
                    mbarrier_wait(dv_read_addr, _phase_dv_read_0);
                    _phase_dv_read_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_4 = (((qg_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
                    int _smem_desc_lo_5 = (((do_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                    int _smem_desc_lo_4 = (((qg_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
                    int _smem_desc_lo_5 = (((do_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    "mov.b32 id, 136414352;\n\t"
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
                    :: "r"(_smem_desc_lo_4), "r"(_smem_desc_lo_5), "r"(tmem_acc_p), "r"(0) : "memory");
                    tcgen05_commit(qd_empty_addr);
                    _phase_qd_full ^= 1;
                    mbarrier_wait(bdv_ready_addr, _phase_bdv_ready_0);
                    _phase_bdv_ready_0 ^= 1;
                    mbarrier_wait(w_full_addr, _phase_w_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_6 = (((w_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
                    int _smem_desc_lo_7 = (((bdv_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                    int _smem_desc_lo_6 = (((w_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
                    int _smem_desc_lo_7 = (((bdv_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    "mov.b32 id, %4;\n\t"
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
                    :: "r"(_smem_desc_lo_6), "r"(_smem_desc_lo_7), "r"(tmem_acc_dh), "r"(1), "r"((((uint32_t)(136414352)) | ((uint32_t)(1) << 13) | ((uint32_t)(0) << 14))) : "memory");
                    tcgen05_commit(dh_ready_addr);
                    tcgen05_commit(w_empty_addr);
                    _phase_w_full ^= 1;
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
