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


extern "C" {

__global__ __launch_bounds__(320, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_train_7d43005ed0adb9ffadf2_kcp(const __grid_constant__ CUtensorMap w_tma, const __grid_constant__ CUtensorMap kg_tma, __nv_bfloat16* __restrict__ u, float* __restrict__ gk, int num_heads, int* __restrict__ seq_chunk_start, int num_seqs, int* __restrict__ walk_seg, float* __restrict__ h0, int* __restrict__ h0_index, int h0_hg, int h0_n, float* __restrict__ out, int* __restrict__ out_index, int out_hg, int out_n, int tile_stride, __nv_bfloat16* __restrict__ h_tape, __nv_bfloat16* __restrict__ v_new)
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
    #define tiles_full_addr (mbar_base + 0)
    #define tiles_empty_addr (mbar_base + 16)
    #define hs_ready_addr (mbar_base + 32)
    #define vp_ready_addr (mbar_base + 40)
    #define bv_ready_addr (mbar_base + 48)
    #define dec_done_addr (mbar_base + 56)
    #define h_ready_addr (mbar_base + 64)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    __nv_bfloat16* w_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int w_panel_addr = smem + 1024;
    __nv_bfloat16* kg_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int kg_panel_addr = smem + 33792;
    __nv_bfloat16* w_op = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int w_op_addr = smem + 1024;
    __nv_bfloat16* kg_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int kg_mn_addr = smem + 33792;
    __nv_bfloat16* hs = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int hs_addr = smem + 66560;
    __nv_bfloat16* hs_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int hs_mn_addr = smem + 66560;
    __nv_bfloat16* bv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int bv_addr = smem + 99328;
    __nv_bfloat16* bv_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int bv_mn_addr = smem + 99328;

    // Mbarrier init (9 barriers at smem_raw[0..72))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ring' ---
            // tiles_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // tiles_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // --- pipeline 'one' ---
            // hs_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 32, 128);
            // vp_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            // bv_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 48, 128);
            // dec_done: 1 barriers, init_count=128
            mbarrier_init(smem + 56, 128);
            // h_ready: 1 barriers, init_count=2
            mbarrier_init(smem + 64, 2);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 72);
    if (warp == 0) {
        int _tmem_hold = smem + 72;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_acc_h = taddr;
    const unsigned int tmem_acc_vp = taddr + 128;

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
            int h_row_base = warp_in_wg * 32 << 16;
            int lane_half = lane >> 4;
            int t_row = warp_in_wg * 16 + (lane & 15);
            int vp_row_base = warp_in_wg * 32 << 16;
            float init_row[16];
            int h0_slot = h0_index[seg];
            if (h0_slot >= 0) {
                int h0_tile = (head / h0_hg * h0_n + h0_slot) * h0_hg + head % h0_hg;
                long long h0_base = (long long)h0_tile * (long long)tile_stride * 16384ll + (long long)(k_row * 128);
                #pragma unroll
                for (int blk = 0; blk < 8; blk++) {
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
                for (int blk_1 = 0; blk_1 < 8; blk_1++) {
                    tmem_st_x16_f32(taddr + (unsigned int)h_row_base + (unsigned int)(blk_1 * 16), init_row);
                }
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            unsigned int u_pk[32];
            u_pk[0] = 0;
            u_pk[1] = 0;
            u_pk[2] = 0;
            u_pk[3] = 0;
            u_pk[4] = 0;
            u_pk[5] = 0;
            u_pk[6] = 0;
            u_pk[7] = 0;
            u_pk[8] = 0;
            u_pk[9] = 0;
            u_pk[10] = 0;
            u_pk[11] = 0;
            u_pk[12] = 0;
            u_pk[13] = 0;
            u_pk[14] = 0;
            u_pk[15] = 0;
            u_pk[16] = 0;
            u_pk[17] = 0;
            u_pk[18] = 0;
            u_pk[19] = 0;
            u_pk[20] = 0;
            u_pk[21] = 0;
            u_pk[22] = 0;
            u_pk[23] = 0;
            u_pk[24] = 0;
            u_pk[25] = 0;
            u_pk[26] = 0;
            u_pk[27] = 0;
            u_pk[28] = 0;
            u_pk[29] = 0;
            u_pk[30] = 0;
            u_pk[31] = 0;
            if (nc > 0) {
                {
                    uint4 _uv4_1 = *reinterpret_cast<const uint4*>(u + ((long long)(row0 + t_row) * (long long)num_heads + (long long)head) * 128ll + (long long)(lane_half * 16));
                    u_pk[0 + 0] = _uv4_1.x;
                    u_pk[0 + 1] = _uv4_1.y;
                    u_pk[0 + 2] = _uv4_1.z;
                    u_pk[0 + 3] = _uv4_1.w;
                }
                {
                    uint4 _uv4_2 = *reinterpret_cast<const uint4*>(u + ((long long)(row0 + t_row) * (long long)num_heads + (long long)head) * 128ll + (long long)(lane_half * 16 + 8));
                    u_pk[4 + 0] = _uv4_2.x;
                    u_pk[4 + 1] = _uv4_2.y;
                    u_pk[4 + 2] = _uv4_2.z;
                    u_pk[4 + 3] = _uv4_2.w;
                }
                {
                    uint4 _uv4_3 = *reinterpret_cast<const uint4*>(u + ((long long)(row0 + t_row) * (long long)num_heads + (long long)head) * 128ll + (long long)(32 + lane_half * 16));
                    u_pk[8 + 0] = _uv4_3.x;
                    u_pk[8 + 1] = _uv4_3.y;
                    u_pk[8 + 2] = _uv4_3.z;
                    u_pk[8 + 3] = _uv4_3.w;
                }
                {
                    uint4 _uv4_4 = *reinterpret_cast<const uint4*>(u + ((long long)(row0 + t_row) * (long long)num_heads + (long long)head) * 128ll + (long long)(32 + lane_half * 16 + 8));
                    u_pk[12 + 0] = _uv4_4.x;
                    u_pk[12 + 1] = _uv4_4.y;
                    u_pk[12 + 2] = _uv4_4.z;
                    u_pk[12 + 3] = _uv4_4.w;
                }
                {
                    uint4 _uv4_5 = *reinterpret_cast<const uint4*>(u + ((long long)(row0 + t_row) * (long long)num_heads + (long long)head) * 128ll + (long long)(64 + lane_half * 16));
                    u_pk[16 + 0] = _uv4_5.x;
                    u_pk[16 + 1] = _uv4_5.y;
                    u_pk[16 + 2] = _uv4_5.z;
                    u_pk[16 + 3] = _uv4_5.w;
                }
                {
                    uint4 _uv4_6 = *reinterpret_cast<const uint4*>(u + ((long long)(row0 + t_row) * (long long)num_heads + (long long)head) * 128ll + (long long)(64 + lane_half * 16 + 8));
                    u_pk[20 + 0] = _uv4_6.x;
                    u_pk[20 + 1] = _uv4_6.y;
                    u_pk[20 + 2] = _uv4_6.z;
                    u_pk[20 + 3] = _uv4_6.w;
                }
                {
                    uint4 _uv4_7 = *reinterpret_cast<const uint4*>(u + ((long long)(row0 + t_row) * (long long)num_heads + (long long)head) * 128ll + (long long)(96 + lane_half * 16));
                    u_pk[24 + 0] = _uv4_7.x;
                    u_pk[24 + 1] = _uv4_7.y;
                    u_pk[24 + 2] = _uv4_7.z;
                    u_pk[24 + 3] = _uv4_7.w;
                }
                {
                    uint4 _uv4_8 = *reinterpret_cast<const uint4*>(u + ((long long)(row0 + t_row) * (long long)num_heads + (long long)head) * 128ll + (long long)(96 + lane_half * 16 + 8));
                    u_pk[28 + 0] = _uv4_8.x;
                    u_pk[28 + 1] = _uv4_8.y;
                    u_pk[28 + 2] = _uv4_8.z;
                    u_pk[28 + 3] = _uv4_8.w;
                }
            }
            unsigned int _phase_vp_ready_0 = 0;
            unsigned int _phase_h_ready_0 = 0;
            #pragma unroll 1
            for (int c = 0; c < nc; c++) {
                #pragma unroll
                for (int blk_2 = 0; blk_2 < 8; blk_2++) {
                    float h_blk[16];
                    tmem_ld_x16(&h_blk[0], taddr + (unsigned int)h_row_base + (unsigned int)(blk_2 * 16));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    unsigned int h_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_blk[_lp*2 + 0], h_blk[_lp*2+1 + 0]));
                        h_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_1 = 0; q_1 < 2; q_1++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((blk_2 * 16 + q_1 * 8) / 64 * 16384 + k_row * 128 + (blk_2 * 16 + q_1 * 8) % 64 * 2 ^ ((blk_2 * 16 + q_1 * 8) / 64 * 16384 + k_row * 128 + (blk_2 * 16 + q_1 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(h_bf16[q_1 * 4]), "r"(h_bf16[q_1 * 4 + 1]), "r"(h_bf16[q_1 * 4 + 2]), "r"(h_bf16[q_1 * 4 + 3]) : "memory");
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(hs_ready_addr);
                mbarrier_wait(vp_ready_addr, _phase_vp_ready_0);
                _phase_vp_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                long long u_base = ((long long)(row0 + c * 64 + t_row) * (long long)num_heads + (long long)head) * 128ll;
                float vp_blk_0[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_0[15]))
                    : "r"(taddr + 128 + (unsigned int)vp_row_base + 0));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col_0 = lane_half * 16;
                float u_blk_0[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&u_blk_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(u_pk[_pair]) << 16);
                    (&u_blk_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(u_pk[_pair]) & 0xffff0000u);
                }
                float bvv_0[16];
                #pragma unroll
                for (int j_8 = 0; j_8 < 16; j_8++) {
                    bvv_0[j_8] = u_blk_0[j_8] - vp_blk_0[j_8];
                }
                unsigned int bvv_bf16_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_0[_lp*2 + 0], bvv_0[_lp*2+1 + 0]));
                    bvv_bf16_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_0 = 0; q_0 < 2; q_0++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv_addr + (unsigned int)((col_0 + q_0 * 8) / 64 * 8192 + t_row * 128 + (col_0 + q_0 * 8) % 64 * 2 ^ ((col_0 + q_0 * 8) / 64 * 8192 + t_row * 128 + (col_0 + q_0 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_bf16_0[q_0 * 4]), "r"(bvv_bf16_0[q_0 * 4 + 1]), "r"(bvv_bf16_0[q_0 * 4 + 2]), "r"(bvv_bf16_0[q_0 * 4 + 3]) : "memory");
                }
                float vp_blk_1[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_1[15]))
                    : "r"(taddr + 128 + (unsigned int)vp_row_base + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col_1 = 32 + lane_half * 16;
                float u_blk_1[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&u_blk_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(u_pk[8 + _pair]) << 16);
                    (&u_blk_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(u_pk[8 + _pair]) & 0xffff0000u);
                }
                float bvv_1[16];
                #pragma unroll
                for (int j_9 = 0; j_9 < 16; j_9++) {
                    bvv_1[j_9] = u_blk_1[j_9] - vp_blk_1[j_9];
                }
                unsigned int bvv_bf16_1[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_1[_lp*2 + 0], bvv_1[_lp*2+1 + 0]));
                    bvv_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_1_1 = 0; q_1_1 < 2; q_1_1++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv_addr + (unsigned int)((col_1 + q_1_1 * 8) / 64 * 8192 + t_row * 128 + (col_1 + q_1_1 * 8) % 64 * 2 ^ ((col_1 + q_1_1 * 8) / 64 * 8192 + t_row * 128 + (col_1 + q_1_1 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_bf16_1[q_1_1 * 4]), "r"(bvv_bf16_1[q_1_1 * 4 + 1]), "r"(bvv_bf16_1[q_1_1 * 4 + 2]), "r"(bvv_bf16_1[q_1_1 * 4 + 3]) : "memory");
                }
                float vp_blk_2[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_2[15]))
                    : "r"(taddr + 128 + (unsigned int)vp_row_base + 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col_2 = 64 + lane_half * 16;
                float u_blk_2[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&u_blk_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(u_pk[16 + _pair]) << 16);
                    (&u_blk_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(u_pk[16 + _pair]) & 0xffff0000u);
                }
                float bvv_2[16];
                #pragma unroll
                for (int j_10 = 0; j_10 < 16; j_10++) {
                    bvv_2[j_10] = u_blk_2[j_10] - vp_blk_2[j_10];
                }
                unsigned int bvv_bf16_2[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_2[_lp*2 + 0], bvv_2[_lp*2+1 + 0]));
                    bvv_bf16_2[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_2 = 0; q_2 < 2; q_2++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv_addr + (unsigned int)((col_2 + q_2 * 8) / 64 * 8192 + t_row * 128 + (col_2 + q_2 * 8) % 64 * 2 ^ ((col_2 + q_2 * 8) / 64 * 8192 + t_row * 128 + (col_2 + q_2 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_bf16_2[q_2 * 4]), "r"(bvv_bf16_2[q_2 * 4 + 1]), "r"(bvv_bf16_2[q_2 * 4 + 2]), "r"(bvv_bf16_2[q_2 * 4 + 3]) : "memory");
                }
                float vp_blk_3[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&vp_blk_3[15]))
                    : "r"(taddr + 128 + (unsigned int)vp_row_base + 96));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col_3 = 96 + lane_half * 16;
                float u_blk_3[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&u_blk_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(u_pk[24 + _pair]) << 16);
                    (&u_blk_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(u_pk[24 + _pair]) & 0xffff0000u);
                }
                float bvv_3[16];
                #pragma unroll
                for (int j_11 = 0; j_11 < 16; j_11++) {
                    bvv_3[j_11] = u_blk_3[j_11] - vp_blk_3[j_11];
                }
                unsigned int bvv_bf16_3[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_3[_lp*2 + 0], bvv_3[_lp*2+1 + 0]));
                    bvv_bf16_3[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_3 = 0; q_3 < 2; q_3++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv_addr + (unsigned int)((col_3 + q_3 * 8) / 64 * 8192 + t_row * 128 + (col_3 + q_3 * 8) % 64 * 2 ^ ((col_3 + q_3 * 8) / 64 * 8192 + t_row * 128 + (col_3 + q_3 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_bf16_3[q_3 * 4]), "r"(bvv_bf16_3[q_3 * 4 + 1]), "r"(bvv_bf16_3[q_3 * 4 + 2]), "r"(bvv_bf16_3[q_3 * 4 + 3]) : "memory");
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(bv_ready_addr);
                if (nc > c + 1) {
                    long long u_next = u_base + (long long)(64 * num_heads * 128);
                    {
                        uint4 _uv4_9 = *reinterpret_cast<const uint4*>(u + u_next + (long long)(lane_half * 16));
                        u_pk[0 + 0] = _uv4_9.x;
                        u_pk[0 + 1] = _uv4_9.y;
                        u_pk[0 + 2] = _uv4_9.z;
                        u_pk[0 + 3] = _uv4_9.w;
                    }
                    {
                        uint4 _uv4_10 = *reinterpret_cast<const uint4*>(u + u_next + (long long)(lane_half * 16 + 8));
                        u_pk[4 + 0] = _uv4_10.x;
                        u_pk[4 + 1] = _uv4_10.y;
                        u_pk[4 + 2] = _uv4_10.z;
                        u_pk[4 + 3] = _uv4_10.w;
                    }
                    {
                        uint4 _uv4_11 = *reinterpret_cast<const uint4*>(u + u_next + (long long)(32 + lane_half * 16));
                        u_pk[8 + 0] = _uv4_11.x;
                        u_pk[8 + 1] = _uv4_11.y;
                        u_pk[8 + 2] = _uv4_11.z;
                        u_pk[8 + 3] = _uv4_11.w;
                    }
                    {
                        uint4 _uv4_12 = *reinterpret_cast<const uint4*>(u + u_next + (long long)(32 + lane_half * 16 + 8));
                        u_pk[12 + 0] = _uv4_12.x;
                        u_pk[12 + 1] = _uv4_12.y;
                        u_pk[12 + 2] = _uv4_12.z;
                        u_pk[12 + 3] = _uv4_12.w;
                    }
                    {
                        uint4 _uv4_13 = *reinterpret_cast<const uint4*>(u + u_next + (long long)(64 + lane_half * 16));
                        u_pk[16 + 0] = _uv4_13.x;
                        u_pk[16 + 1] = _uv4_13.y;
                        u_pk[16 + 2] = _uv4_13.z;
                        u_pk[16 + 3] = _uv4_13.w;
                    }
                    {
                        uint4 _uv4_14 = *reinterpret_cast<const uint4*>(u + u_next + (long long)(64 + lane_half * 16 + 8));
                        u_pk[20 + 0] = _uv4_14.x;
                        u_pk[20 + 1] = _uv4_14.y;
                        u_pk[20 + 2] = _uv4_14.z;
                        u_pk[20 + 3] = _uv4_14.w;
                    }
                    {
                        uint4 _uv4_15 = *reinterpret_cast<const uint4*>(u + u_next + (long long)(96 + lane_half * 16));
                        u_pk[24 + 0] = _uv4_15.x;
                        u_pk[24 + 1] = _uv4_15.y;
                        u_pk[24 + 2] = _uv4_15.z;
                        u_pk[24 + 3] = _uv4_15.w;
                    }
                    {
                        uint4 _uv4_16 = *reinterpret_cast<const uint4*>(u + u_next + (long long)(96 + lane_half * 16 + 8));
                        u_pk[28 + 0] = _uv4_16.x;
                        u_pk[28 + 1] = _uv4_16.y;
                        u_pk[28 + 2] = _uv4_16.z;
                        u_pk[28 + 3] = _uv4_16.w;
                    }
                }
                mbarrier_wait(h_ready_addr, _phase_h_ready_0);
                _phase_h_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
            }
            int out_slot = out_index[seg];
            if (out_slot >= 0) {
                int out_tile = (head / out_hg * out_n + out_slot) * out_hg + head % out_hg;
                long long out_base = (long long)out_tile * (long long)tile_stride * 16384ll + (long long)(k_row * 128);
                #pragma unroll
                for (int blk_3 = 0; blk_3 < 8; blk_3++) {
                    float h_out[16];
                    tmem_ld_x16(&h_out[0], taddr + (unsigned int)h_row_base + (unsigned int)(blk_3 * 16));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int q_4 = 0; q_4 < 4; q_4++) {
                        {
                            float4 _v4 = make_float4(h_out[q_4 * 4 + 0], h_out[q_4 * 4 + 1], h_out[q_4 * 4 + 2], h_out[q_4 * 4 + 3]);
                            *reinterpret_cast<float4*>(out + out_base + (long long)(blk_3 * 16 + q_4 * 4)) = _v4;
                        }
                    }
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
            }
        }
    }
    // ---- Role: decayw ----
    if (warp >= 4 && warp <= 7) {
        { // decayw_main
            int head_1 = blockIdx.x % num_heads;
            int walk_1 = blockIdx.x / num_heads;
            int seg_1 = walk_seg[walk_1];
            int c0_1 = seq_chunk_start[seg_1];
            int nc_1 = seq_chunk_start[seg_1 + 1] - c0_1;
            int row0_1 = c0_1 * 64;
            int warp_in_wg_1 = warp % 4;
            int k_row_1 = warp_in_wg_1 * 32 + lane;
            int h_row_base_1 = warp_in_wg_1 * 32 << 16;
            float gk_pre = 0.0f;
            if (nc_1 > 0) {
                long long gn_first = ((long long)(row0_1 + 64 - 1) * (long long)num_heads + (long long)head_1) * 128ll + (long long)k_row_1;
                gk_pre = gk[gn_first];
            }
            unsigned int _phase_hs_ready_0 = 0;
            #pragma unroll 1
            for (int c_1 = 0; c_1 < nc_1; c_1++) {
                mbarrier_wait(hs_ready_addr, _phase_hs_ready_0);
                _phase_hs_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float decay = approx_exp2(gk_pre);
                float h_blk_0[16];
                tmem_ld_x16(&h_blk_0[0], taddr + (unsigned int)h_row_base_1 + 0);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hd_0[16];
                #pragma unroll
                for (int j_0 = 0; j_0 < 16; j_0++) {
                    hd_0[j_0] = h_blk_0[j_0] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 0, hd_0);
                float h_blk_1[16];
                tmem_ld_x16(&h_blk_1[0], taddr + (unsigned int)h_row_base_1 + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hd_1[16];
                #pragma unroll
                for (int j_1 = 0; j_1 < 16; j_1++) {
                    hd_1[j_1] = h_blk_1[j_1] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 16, hd_1);
                float h_blk_2[16];
                tmem_ld_x16(&h_blk_2[0], taddr + (unsigned int)h_row_base_1 + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hd_2[16];
                #pragma unroll
                for (int j_2 = 0; j_2 < 16; j_2++) {
                    hd_2[j_2] = h_blk_2[j_2] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 32, hd_2);
                float h_blk_3[16];
                tmem_ld_x16(&h_blk_3[0], taddr + (unsigned int)h_row_base_1 + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hd_3[16];
                #pragma unroll
                for (int j_3 = 0; j_3 < 16; j_3++) {
                    hd_3[j_3] = h_blk_3[j_3] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 48, hd_3);
                float h_blk_4[16];
                tmem_ld_x16(&h_blk_4[0], taddr + (unsigned int)h_row_base_1 + 64);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hd_4[16];
                #pragma unroll
                for (int j_4 = 0; j_4 < 16; j_4++) {
                    hd_4[j_4] = h_blk_4[j_4] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 64, hd_4);
                float h_blk_5[16];
                tmem_ld_x16(&h_blk_5[0], taddr + (unsigned int)h_row_base_1 + 80);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hd_5[16];
                #pragma unroll
                for (int j_5 = 0; j_5 < 16; j_5++) {
                    hd_5[j_5] = h_blk_5[j_5] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 80, hd_5);
                float h_blk_6[16];
                tmem_ld_x16(&h_blk_6[0], taddr + (unsigned int)h_row_base_1 + 96);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hd_6[16];
                #pragma unroll
                for (int j_6 = 0; j_6 < 16; j_6++) {
                    hd_6[j_6] = h_blk_6[j_6] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 96, hd_6);
                float h_blk_7[16];
                tmem_ld_x16(&h_blk_7[0], taddr + (unsigned int)h_row_base_1 + 112);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hd_7[16];
                #pragma unroll
                for (int j_7 = 0; j_7 < 16; j_7++) {
                    hd_7[j_7] = h_blk_7[j_7] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 112, hd_7);
                if (nc_1 > c_1 + 1) {
                    long long gn_next = ((long long)(row0_1 + c_1 * 64 + 128 - 1) * (long long)num_heads + (long long)head_1) * 128ll + (long long)k_row_1;
                    gk_pre = gk[gn_next];
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(dec_done_addr);
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 256;" ::: "memory");
        }
    }
    // ---- Role: load ----
    if (warp == 8) {
        { // load_main
            int head_2 = blockIdx.x % num_heads;
            int walk_2 = blockIdx.x / num_heads;
            int seg_2 = walk_seg[walk_2];
            int c0_2 = seq_chunk_start[seg_2];
            int nc_2 = seq_chunk_start[seg_2 + 1] - c0_2;
            int row0_2 = c0_2 * 64;
            unsigned int stage = 0;
            unsigned int _phase_tiles_empty = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int c_2 = 0; c_2 < nc_2; c_2++) {
                    mbarrier_wait(tiles_empty_addr + (stage) * 8, _phase_tiles_empty);
                    mbarrier_arrive_expect_tx(tiles_full_addr + (stage) * 8, 32768);
                    int trow = row0_2 + c_2 * 64;
                    #pragma unroll
                    for (int panel = 0; panel < 2; panel++) {
                        tma_3d_gmem2smem(w_panel_addr + (stage * 2 + (unsigned int)panel) * 8192, (&w_tma), panel * 64, head_2, trow, tiles_full_addr + (stage) * 8);
                        tma_3d_gmem2smem(kg_panel_addr + (stage * 2 + (unsigned int)panel) * 8192, (&kg_tma), panel * 64, head_2, trow, tiles_full_addr + (stage) * 8);
                    }
                    stage += 1;
                    if (stage == 2) { stage = 0; _phase_tiles_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 9) {
        { // mma_main
            int head_3 = blockIdx.x % num_heads;
            int walk_3 = blockIdx.x / num_heads;
            int seg_3 = walk_seg[walk_3];
            int c0_3 = seq_chunk_start[seg_3];
            int nc_3 = seq_chunk_start[seg_3 + 1] - c0_3;
            int row0_3 = c0_3 * 64;
            unsigned int stage_1 = 0;
            unsigned int _phase_tiles_full = 0;
            unsigned int _phase_hs_ready_0_1 = 0;
            unsigned int _phase_bv_ready_0 = 0;
            unsigned int _phase_dec_done_0 = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int c_3 = 0; c_3 < nc_3; c_3++) {
                    mbarrier_wait(tiles_full_addr + (stage_1) * 8, _phase_tiles_full);
                    mbarrier_wait(hs_ready_addr, _phase_hs_ready_0_1);
                    _phase_hs_ready_0_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_0 = (((w_op_addr) >> 4) & 0x7FFF) + (stage_1) * 1024;
                    int _smem_desc_lo_1 = (((hs_mn_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
                    int _smem_desc_lo_0 = (((w_op_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
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
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"(tmem_acc_vp), "r"(0) : "memory");
                    tcgen05_commit(vp_ready_addr);
                    mbarrier_wait(bv_ready_addr, _phase_bv_ready_0);
                    _phase_bv_ready_0 ^= 1;
                    mbarrier_wait(dec_done_addr, _phase_dec_done_0);
                    _phase_dec_done_0 ^= 1;
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(h_ready_addr);
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_2 = ((((kg_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (stage_1) * 1024;
                    int _smem_desc_lo_3 = (((bv_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                    int _smem_desc_lo_2 = ((((kg_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (stage_1) * 1024;
                    int _smem_desc_lo_3 = (((bv_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"(tmem_acc_h), "r"(1) : "memory");
                    tcgen05_commit(h_ready_addr);
                    tcgen05_commit(tiles_empty_addr + (stage_1) * 8);
                    stage_1 += 1;
                    if (stage_1 == 2) { stage_1 = 0; _phase_tiles_full ^= 1; }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
