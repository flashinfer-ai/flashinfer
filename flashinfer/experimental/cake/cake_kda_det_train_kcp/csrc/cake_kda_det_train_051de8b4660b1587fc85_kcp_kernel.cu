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

__global__ __launch_bounds__(320, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_det_train_051de8b4660b1587fc85_kcp(const __grid_constant__ CUtensorMap w_tma, const __grid_constant__ CUtensorMap kg_tma, __nv_bfloat16* __restrict__ u, float* __restrict__ gk, int num_heads, int* __restrict__ seq_chunk_start, int num_seqs, int* __restrict__ walk_seg, float* __restrict__ h0, int* __restrict__ h0_index, int h0_hg, int h0_n, int tile_stride, const __grid_constant__ CUtensorMap qg_tma, const __grid_constant__ CUtensorMap aqk_tma, __nv_bfloat16* __restrict__ o_out, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, float scale)
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
    #define qg_full_addr (mbar_base + 72)
    #define aqk_full_addr (mbar_base + 80)
    #define qg_empty_addr (mbar_base + 88)
    #define aqk_empty_addr (mbar_base + 96)
    #define o_done_addr (mbar_base + 104)
    #define o_drained_addr (mbar_base + 112)

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
    __nv_bfloat16* bv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int bv_addr = smem + 99328;
    __nv_bfloat16* hs_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int hs_mn_addr = smem + 66560;
    __nv_bfloat16* bv_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int bv_mn_addr = smem + 99328;
    __nv_bfloat16* qg_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 115712);
    const int qg_panel_addr = smem + 115712;
    __nv_bfloat16* qg_op = reinterpret_cast<__nv_bfloat16*>(smem_raw + 115712);
    const int qg_op_addr = smem + 115712;
    __nv_bfloat16* aqk_op = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int aqk_op_addr = smem + 132096;

    // Mbarrier init (15 barriers at smem_raw[0..120))

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
            // qg_full: 1 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            // aqk_full: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // qg_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            // aqk_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            // o_done: 1 barriers, init_count=1
            mbarrier_init(smem + 104, 1);
            // o_drained: 1 barriers, init_count=128
            mbarrier_init(smem + 112, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 120);
    if (warp == 0) {
        int _tmem_hold = smem + 120;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_acc_h = taddr;
    const unsigned int tmem_acc_vp = taddr + 128;
    const unsigned int tmem_acc_qh = taddr + 256;
    const unsigned int tmem_acc_av = taddr + 384;

    // ---- Role: compute ----
    if (warp <= 3) {
        { // compute_main
            int head = blockIdx.x % num_heads;
            int seg = walk_seg[blockIdx.x / num_heads];
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
                for (int q_0 = 0; q_0 < 4; q_0++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(h0 + h0_base + (long long)(q_0 * 4));
                        init_row[q_0 * 4 + 0] = _v4.x;
                        init_row[q_0 * 4 + 1] = _v4.y;
                        init_row[q_0 * 4 + 2] = _v4.z;
                        init_row[q_0 * 4 + 3] = _v4.w;
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base, init_row);
                #pragma unroll
                for (int q_1 = 0; q_1 < 4; q_1++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(h0 + h0_base + (long long)(16 + q_1 * 4));
                        init_row[q_1 * 4 + 0] = _v4.x;
                        init_row[q_1 * 4 + 1] = _v4.y;
                        init_row[q_1 * 4 + 2] = _v4.z;
                        init_row[q_1 * 4 + 3] = _v4.w;
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 16, init_row);
                #pragma unroll
                for (int q_2 = 0; q_2 < 4; q_2++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(h0 + h0_base + (long long)(32 + q_2 * 4));
                        init_row[q_2 * 4 + 0] = _v4.x;
                        init_row[q_2 * 4 + 1] = _v4.y;
                        init_row[q_2 * 4 + 2] = _v4.z;
                        init_row[q_2 * 4 + 3] = _v4.w;
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 32, init_row);
                #pragma unroll
                for (int q_3 = 0; q_3 < 4; q_3++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(h0 + h0_base + (long long)(48 + q_3 * 4));
                        init_row[q_3 * 4 + 0] = _v4.x;
                        init_row[q_3 * 4 + 1] = _v4.y;
                        init_row[q_3 * 4 + 2] = _v4.z;
                        init_row[q_3 * 4 + 3] = _v4.w;
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 48, init_row);
                #pragma unroll
                for (int q_4 = 0; q_4 < 4; q_4++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(h0 + h0_base + (long long)(64 + q_4 * 4));
                        init_row[q_4 * 4 + 0] = _v4.x;
                        init_row[q_4 * 4 + 1] = _v4.y;
                        init_row[q_4 * 4 + 2] = _v4.z;
                        init_row[q_4 * 4 + 3] = _v4.w;
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 64, init_row);
                #pragma unroll
                for (int q_5 = 0; q_5 < 4; q_5++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(h0 + h0_base + (long long)(80 + q_5 * 4));
                        init_row[q_5 * 4 + 0] = _v4.x;
                        init_row[q_5 * 4 + 1] = _v4.y;
                        init_row[q_5 * 4 + 2] = _v4.z;
                        init_row[q_5 * 4 + 3] = _v4.w;
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 80, init_row);
                #pragma unroll
                for (int q_6 = 0; q_6 < 4; q_6++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(h0 + h0_base + (long long)(96 + q_6 * 4));
                        init_row[q_6 * 4 + 0] = _v4.x;
                        init_row[q_6 * 4 + 1] = _v4.y;
                        init_row[q_6 * 4 + 2] = _v4.z;
                        init_row[q_6 * 4 + 3] = _v4.w;
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 96, init_row);
                #pragma unroll
                for (int q_7 = 0; q_7 < 4; q_7++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(h0 + h0_base + (long long)(112 + q_7 * 4));
                        init_row[q_7 * 4 + 0] = _v4.x;
                        init_row[q_7 * 4 + 1] = _v4.y;
                        init_row[q_7 * 4 + 2] = _v4.z;
                        init_row[q_7 * 4 + 3] = _v4.w;
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 112, init_row);
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
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base, init_row);
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 16, init_row);
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 32, init_row);
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 48, init_row);
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 64, init_row);
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 80, init_row);
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 96, init_row);
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 112, init_row);
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
                long long u_next = ((long long)(row0 + t_row) * (long long)num_heads + (long long)head) * 128ll;
                #pragma unroll
                for (int q_8 = 0; q_8 < 2; q_8++) {
                    {
                        uint32_t _uv4_8_0;
                        uint32_t _uv4_8_1;
                        uint32_t _uv4_8_2;
                        uint32_t _uv4_8_3;
                        asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_uv4_8_0), "=r"(_uv4_8_1), "=r"(_uv4_8_2), "=r"(_uv4_8_3) : "l"((const void*)(u + (u_next + (long long)(lane_half * 16 + q_8 * 8)))) : "memory");
                        u_pk[q_8 * 4 + 0] = _uv4_8_0;
                        u_pk[q_8 * 4 + 1] = _uv4_8_1;
                        u_pk[q_8 * 4 + 2] = _uv4_8_2;
                        u_pk[q_8 * 4 + 3] = _uv4_8_3;
                    }
                }
                #pragma unroll
                for (int q_9 = 0; q_9 < 2; q_9++) {
                    {
                        uint32_t _uv4_9_0;
                        uint32_t _uv4_9_1;
                        uint32_t _uv4_9_2;
                        uint32_t _uv4_9_3;
                        asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_uv4_9_0), "=r"(_uv4_9_1), "=r"(_uv4_9_2), "=r"(_uv4_9_3) : "l"((const void*)(u + (u_next + (long long)(32 + lane_half * 16 + q_9 * 8)))) : "memory");
                        u_pk[8 + q_9 * 4 + 0] = _uv4_9_0;
                        u_pk[8 + q_9 * 4 + 1] = _uv4_9_1;
                        u_pk[8 + q_9 * 4 + 2] = _uv4_9_2;
                        u_pk[8 + q_9 * 4 + 3] = _uv4_9_3;
                    }
                }
                #pragma unroll
                for (int q_10 = 0; q_10 < 2; q_10++) {
                    {
                        uint32_t _uv4_10_0;
                        uint32_t _uv4_10_1;
                        uint32_t _uv4_10_2;
                        uint32_t _uv4_10_3;
                        asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_uv4_10_0), "=r"(_uv4_10_1), "=r"(_uv4_10_2), "=r"(_uv4_10_3) : "l"((const void*)(u + (u_next + (long long)(64 + lane_half * 16 + q_10 * 8)))) : "memory");
                        u_pk[16 + q_10 * 4 + 0] = _uv4_10_0;
                        u_pk[16 + q_10 * 4 + 1] = _uv4_10_1;
                        u_pk[16 + q_10 * 4 + 2] = _uv4_10_2;
                        u_pk[16 + q_10 * 4 + 3] = _uv4_10_3;
                    }
                }
                #pragma unroll
                for (int q_11 = 0; q_11 < 2; q_11++) {
                    {
                        uint32_t _uv4_11_0;
                        uint32_t _uv4_11_1;
                        uint32_t _uv4_11_2;
                        uint32_t _uv4_11_3;
                        asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_uv4_11_0), "=r"(_uv4_11_1), "=r"(_uv4_11_2), "=r"(_uv4_11_3) : "l"((const void*)(u + (u_next + (long long)(96 + lane_half * 16 + q_11 * 8)))) : "memory");
                        u_pk[24 + q_11 * 4 + 0] = _uv4_11_0;
                        u_pk[24 + q_11 * 4 + 1] = _uv4_11_1;
                        u_pk[24 + q_11 * 4 + 2] = _uv4_11_2;
                        u_pk[24 + q_11 * 4 + 3] = _uv4_11_3;
                    }
                }
            }
            unsigned int _phase_vp_ready_0 = 0;
            unsigned int _phase_h_ready_0 = 0;
            #pragma unroll 1
            for (int c = 0; c < nc; c++) {
                float hv_8[16];
                tmem_ld_x16(&hv_8[0], taddr + (unsigned int)h_row_base);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int hv_bf16_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(hv_8[_lp*2 + 0], hv_8[_lp*2+1 + 0]));
                    hv_bf16_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_12 = 0; q_12 < 2; q_12++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)(q_12 * 8 / 64 * 16384 + k_row * 128 + q_12 * 8 % 64 * 2 ^ (q_12 * 8 / 64 * 16384 + k_row * 128 + q_12 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(hv_bf16_0[q_12 * 4]), "r"(hv_bf16_0[q_12 * 4 + 1]), "r"(hv_bf16_0[q_12 * 4 + 2]), "r"(hv_bf16_0[q_12 * 4 + 3]) : "memory");
                }
                float hv_9[16];
                tmem_ld_x16(&hv_9[0], taddr + (unsigned int)h_row_base + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int hv_bf16_1[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(hv_9[_lp*2 + 0], hv_9[_lp*2+1 + 0]));
                    hv_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_13 = 0; q_13 < 2; q_13++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((16 + q_13 * 8) / 64 * 16384 + k_row * 128 + (16 + q_13 * 8) % 64 * 2 ^ ((16 + q_13 * 8) / 64 * 16384 + k_row * 128 + (16 + q_13 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(hv_bf16_1[q_13 * 4]), "r"(hv_bf16_1[q_13 * 4 + 1]), "r"(hv_bf16_1[q_13 * 4 + 2]), "r"(hv_bf16_1[q_13 * 4 + 3]) : "memory");
                }
                float hv_10[16];
                tmem_ld_x16(&hv_10[0], taddr + (unsigned int)h_row_base + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int hv_bf16_2[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(hv_10[_lp*2 + 0], hv_10[_lp*2+1 + 0]));
                    hv_bf16_2[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_14 = 0; q_14 < 2; q_14++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((32 + q_14 * 8) / 64 * 16384 + k_row * 128 + (32 + q_14 * 8) % 64 * 2 ^ ((32 + q_14 * 8) / 64 * 16384 + k_row * 128 + (32 + q_14 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(hv_bf16_2[q_14 * 4]), "r"(hv_bf16_2[q_14 * 4 + 1]), "r"(hv_bf16_2[q_14 * 4 + 2]), "r"(hv_bf16_2[q_14 * 4 + 3]) : "memory");
                }
                float hv_11[16];
                tmem_ld_x16(&hv_11[0], taddr + (unsigned int)h_row_base + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int hv_bf16_3[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(hv_11[_lp*2 + 0], hv_11[_lp*2+1 + 0]));
                    hv_bf16_3[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_15 = 0; q_15 < 2; q_15++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((48 + q_15 * 8) / 64 * 16384 + k_row * 128 + (48 + q_15 * 8) % 64 * 2 ^ ((48 + q_15 * 8) / 64 * 16384 + k_row * 128 + (48 + q_15 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(hv_bf16_3[q_15 * 4]), "r"(hv_bf16_3[q_15 * 4 + 1]), "r"(hv_bf16_3[q_15 * 4 + 2]), "r"(hv_bf16_3[q_15 * 4 + 3]) : "memory");
                }
                float hv_12[16];
                tmem_ld_x16(&hv_12[0], taddr + (unsigned int)h_row_base + 64);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int hv_bf16_4[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(hv_12[_lp*2 + 0], hv_12[_lp*2+1 + 0]));
                    hv_bf16_4[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_16 = 0; q_16 < 2; q_16++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((64 + q_16 * 8) / 64 * 16384 + k_row * 128 + (64 + q_16 * 8) % 64 * 2 ^ ((64 + q_16 * 8) / 64 * 16384 + k_row * 128 + (64 + q_16 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(hv_bf16_4[q_16 * 4]), "r"(hv_bf16_4[q_16 * 4 + 1]), "r"(hv_bf16_4[q_16 * 4 + 2]), "r"(hv_bf16_4[q_16 * 4 + 3]) : "memory");
                }
                float hv_13[16];
                tmem_ld_x16(&hv_13[0], taddr + (unsigned int)h_row_base + 80);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int hv_bf16_5[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(hv_13[_lp*2 + 0], hv_13[_lp*2+1 + 0]));
                    hv_bf16_5[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_17 = 0; q_17 < 2; q_17++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((80 + q_17 * 8) / 64 * 16384 + k_row * 128 + (80 + q_17 * 8) % 64 * 2 ^ ((80 + q_17 * 8) / 64 * 16384 + k_row * 128 + (80 + q_17 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(hv_bf16_5[q_17 * 4]), "r"(hv_bf16_5[q_17 * 4 + 1]), "r"(hv_bf16_5[q_17 * 4 + 2]), "r"(hv_bf16_5[q_17 * 4 + 3]) : "memory");
                }
                float hv_14[16];
                tmem_ld_x16(&hv_14[0], taddr + (unsigned int)h_row_base + 96);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int hv_bf16_6[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(hv_14[_lp*2 + 0], hv_14[_lp*2+1 + 0]));
                    hv_bf16_6[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_18 = 0; q_18 < 2; q_18++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((96 + q_18 * 8) / 64 * 16384 + k_row * 128 + (96 + q_18 * 8) % 64 * 2 ^ ((96 + q_18 * 8) / 64 * 16384 + k_row * 128 + (96 + q_18 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(hv_bf16_6[q_18 * 4]), "r"(hv_bf16_6[q_18 * 4 + 1]), "r"(hv_bf16_6[q_18 * 4 + 2]), "r"(hv_bf16_6[q_18 * 4 + 3]) : "memory");
                }
                float hv_15[16];
                tmem_ld_x16(&hv_15[0], taddr + (unsigned int)h_row_base + 112);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int hv_bf16_7[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(hv_15[_lp*2 + 0], hv_15[_lp*2+1 + 0]));
                    hv_bf16_7[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_19 = 0; q_19 < 2; q_19++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((112 + q_19 * 8) / 64 * 16384 + k_row * 128 + (112 + q_19 * 8) % 64 * 2 ^ ((112 + q_19 * 8) / 64 * 16384 + k_row * 128 + (112 + q_19 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(hv_bf16_7[q_19 * 4]), "r"(hv_bf16_7[q_19 * 4 + 1]), "r"(hv_bf16_7[q_19 * 4 + 2]), "r"(hv_bf16_7[q_19 * 4 + 3]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(hs_ready_addr);
                mbarrier_wait(vp_ready_addr, _phase_vp_ready_0);
                _phase_vp_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int trow = row0 + c * 64 + t_row;
                long long u_base = ((long long)trow * (long long)num_heads + (long long)head) * 128ll;
                float vp_f_0[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&vp_f_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_0[15]))
                    : "r"(taddr + 128 + (unsigned int)vp_row_base));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col_4 = lane_half * 16;
                float u_f_0[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&u_f_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(u_pk[_pair]) << 16);
                    (&u_f_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(u_pk[_pair]) & 0xffff0000u);
                }
                float bvv_0[16];
                #pragma unroll
                for (int j_12 = 0; j_12 < 16; j_12++) {
                    bvv_0[j_12] = u_f_0[j_12] - vp_f_0[j_12];
                }
                unsigned int bvv_bf16_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_0[_lp*2 + 0], bvv_0[_lp*2+1 + 0]));
                    bvv_bf16_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_20 = 0; q_20 < 2; q_20++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv_addr + (unsigned int)((col_4 + q_20 * 8) / 64 * 8192 + t_row * 128 + (col_4 + q_20 * 8) % 64 * 2 ^ ((col_4 + q_20 * 8) / 64 * 8192 + t_row * 128 + (col_4 + q_20 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_bf16_0[q_20 * 4]), "r"(bvv_bf16_0[q_20 * 4 + 1]), "r"(bvv_bf16_0[q_20 * 4 + 2]), "r"(bvv_bf16_0[q_20 * 4 + 3]) : "memory");
                }
                float vp_f_1[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&vp_f_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_1[15]))
                    : "r"(taddr + 128 + (unsigned int)vp_row_base + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col_5 = 32 + lane_half * 16;
                float u_f_1[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&u_f_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(u_pk[8 + _pair]) << 16);
                    (&u_f_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(u_pk[8 + _pair]) & 0xffff0000u);
                }
                float bvv_1[16];
                #pragma unroll
                for (int j_13 = 0; j_13 < 16; j_13++) {
                    bvv_1[j_13] = u_f_1[j_13] - vp_f_1[j_13];
                }
                unsigned int bvv_bf16_1[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_1[_lp*2 + 0], bvv_1[_lp*2+1 + 0]));
                    bvv_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_21 = 0; q_21 < 2; q_21++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv_addr + (unsigned int)((col_5 + q_21 * 8) / 64 * 8192 + t_row * 128 + (col_5 + q_21 * 8) % 64 * 2 ^ ((col_5 + q_21 * 8) / 64 * 8192 + t_row * 128 + (col_5 + q_21 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_bf16_1[q_21 * 4]), "r"(bvv_bf16_1[q_21 * 4 + 1]), "r"(bvv_bf16_1[q_21 * 4 + 2]), "r"(bvv_bf16_1[q_21 * 4 + 3]) : "memory");
                }
                float vp_f_2[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&vp_f_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_2[15]))
                    : "r"(taddr + 128 + (unsigned int)vp_row_base + 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col_6 = 64 + lane_half * 16;
                float u_f_2[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&u_f_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(u_pk[16 + _pair]) << 16);
                    (&u_f_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(u_pk[16 + _pair]) & 0xffff0000u);
                }
                float bvv_2[16];
                #pragma unroll
                for (int j_14 = 0; j_14 < 16; j_14++) {
                    bvv_2[j_14] = u_f_2[j_14] - vp_f_2[j_14];
                }
                unsigned int bvv_bf16_2[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_2[_lp*2 + 0], bvv_2[_lp*2+1 + 0]));
                    bvv_bf16_2[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_22 = 0; q_22 < 2; q_22++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv_addr + (unsigned int)((col_6 + q_22 * 8) / 64 * 8192 + t_row * 128 + (col_6 + q_22 * 8) % 64 * 2 ^ ((col_6 + q_22 * 8) / 64 * 8192 + t_row * 128 + (col_6 + q_22 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_bf16_2[q_22 * 4]), "r"(bvv_bf16_2[q_22 * 4 + 1]), "r"(bvv_bf16_2[q_22 * 4 + 2]), "r"(bvv_bf16_2[q_22 * 4 + 3]) : "memory");
                }
                float vp_f_3[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&vp_f_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&vp_f_3[15]))
                    : "r"(taddr + 128 + (unsigned int)vp_row_base + 96));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col_7 = 96 + lane_half * 16;
                float u_f_3[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&u_f_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(u_pk[24 + _pair]) << 16);
                    (&u_f_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(u_pk[24 + _pair]) & 0xffff0000u);
                }
                float bvv_3[16];
                #pragma unroll
                for (int j_15 = 0; j_15 < 16; j_15++) {
                    bvv_3[j_15] = u_f_3[j_15] - vp_f_3[j_15];
                }
                unsigned int bvv_bf16_3[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_3[_lp*2 + 0], bvv_3[_lp*2+1 + 0]));
                    bvv_bf16_3[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_23 = 0; q_23 < 2; q_23++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv_addr + (unsigned int)((col_7 + q_23 * 8) / 64 * 8192 + t_row * 128 + (col_7 + q_23 * 8) % 64 * 2 ^ ((col_7 + q_23 * 8) / 64 * 8192 + t_row * 128 + (col_7 + q_23 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_bf16_3[q_23 * 4]), "r"(bvv_bf16_3[q_23 * 4 + 1]), "r"(bvv_bf16_3[q_23 * 4 + 2]), "r"(bvv_bf16_3[q_23 * 4 + 3]) : "memory");
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(bv_ready_addr);
                if (nc > c + 1) {
                    #pragma unroll
                    for (int q_24 = 0; q_24 < 2; q_24++) {
                        {
                            uint32_t _uv4_12_0;
                            uint32_t _uv4_12_1;
                            uint32_t _uv4_12_2;
                            uint32_t _uv4_12_3;
                            asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                                : "=r"(_uv4_12_0), "=r"(_uv4_12_1), "=r"(_uv4_12_2), "=r"(_uv4_12_3) : "l"((const void*)(u + (u_base + (long long)(64 * num_heads * 128) + (long long)(lane_half * 16 + q_24 * 8)))) : "memory");
                            u_pk[q_24 * 4 + 0] = _uv4_12_0;
                            u_pk[q_24 * 4 + 1] = _uv4_12_1;
                            u_pk[q_24 * 4 + 2] = _uv4_12_2;
                            u_pk[q_24 * 4 + 3] = _uv4_12_3;
                        }
                    }
                    #pragma unroll
                    for (int q_25 = 0; q_25 < 2; q_25++) {
                        {
                            uint32_t _uv4_13_0;
                            uint32_t _uv4_13_1;
                            uint32_t _uv4_13_2;
                            uint32_t _uv4_13_3;
                            asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                                : "=r"(_uv4_13_0), "=r"(_uv4_13_1), "=r"(_uv4_13_2), "=r"(_uv4_13_3) : "l"((const void*)(u + (u_base + (long long)(64 * num_heads * 128) + (long long)(32 + lane_half * 16 + q_25 * 8)))) : "memory");
                            u_pk[8 + q_25 * 4 + 0] = _uv4_13_0;
                            u_pk[8 + q_25 * 4 + 1] = _uv4_13_1;
                            u_pk[8 + q_25 * 4 + 2] = _uv4_13_2;
                            u_pk[8 + q_25 * 4 + 3] = _uv4_13_3;
                        }
                    }
                    #pragma unroll
                    for (int q_26 = 0; q_26 < 2; q_26++) {
                        {
                            uint32_t _uv4_14_0;
                            uint32_t _uv4_14_1;
                            uint32_t _uv4_14_2;
                            uint32_t _uv4_14_3;
                            asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                                : "=r"(_uv4_14_0), "=r"(_uv4_14_1), "=r"(_uv4_14_2), "=r"(_uv4_14_3) : "l"((const void*)(u + (u_base + (long long)(64 * num_heads * 128) + (long long)(64 + lane_half * 16 + q_26 * 8)))) : "memory");
                            u_pk[16 + q_26 * 4 + 0] = _uv4_14_0;
                            u_pk[16 + q_26 * 4 + 1] = _uv4_14_1;
                            u_pk[16 + q_26 * 4 + 2] = _uv4_14_2;
                            u_pk[16 + q_26 * 4 + 3] = _uv4_14_3;
                        }
                    }
                    #pragma unroll
                    for (int q_27 = 0; q_27 < 2; q_27++) {
                        {
                            uint32_t _uv4_15_0;
                            uint32_t _uv4_15_1;
                            uint32_t _uv4_15_2;
                            uint32_t _uv4_15_3;
                            asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                                : "=r"(_uv4_15_0), "=r"(_uv4_15_1), "=r"(_uv4_15_2), "=r"(_uv4_15_3) : "l"((const void*)(u + (u_base + (long long)(64 * num_heads * 128) + (long long)(96 + lane_half * 16 + q_27 * 8)))) : "memory");
                            u_pk[24 + q_27 * 4 + 0] = _uv4_15_0;
                            u_pk[24 + q_27 * 4 + 1] = _uv4_15_1;
                            u_pk[24 + q_27 * 4 + 2] = _uv4_15_2;
                            u_pk[24 + q_27 * 4 + 3] = _uv4_15_3;
                        }
                    }
                }
                mbarrier_wait(h_ready_addr, _phase_h_ready_0);
                _phase_h_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
            }
        }
    }
    // ---- Role: decayw ----
    if (warp >= 4 && warp <= 7) {
        { // decayw_main
            int head_1 = blockIdx.x % num_heads;
            int seg_1 = walk_seg[blockIdx.x / num_heads];
            int c0_1 = seq_chunk_start[seg_1];
            int nc_1 = seq_chunk_start[seg_1 + 1] - c0_1;
            int row0_1 = c0_1 * 64;
            int warp_in_wg_1 = warp % 4;
            int k_row_1 = warp_in_wg_1 * 32 + lane;
            int h_row_base_1 = warp_in_wg_1 * 32 << 16;
            int lane_half_1 = lane >> 4;
            int t_row_1 = warp_in_wg_1 * 16 + (lane & 15);
            int o_row_base = warp_in_wg_1 * 32 << 16;
            int ext_next = 0;
            int clen_next = 0;
            if (nc_1 > 0) {
                ext_next = chunk_bos[c0_1];
                clen_next = chunk_len[c0_1];
            }
            float gk_pre = 0.0f;
            if (nc_1 > 0) {
                long long gn_next = ((long long)(row0_1 + 64 - 1) * (long long)num_heads + (long long)head_1) * 128ll + (long long)k_row_1;
                gk_pre = gk[gn_next];
            }
            unsigned int _phase_hs_ready_0 = 0;
            unsigned int _phase_o_done_0 = 0;
            #pragma unroll 1
            for (int c_1 = 0; c_1 < nc_1; c_1++) {
                mbarrier_wait(hs_ready_addr, _phase_hs_ready_0);
                _phase_hs_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float decay = approx_exp2(gk_pre);
                float hv_0[16];
                tmem_ld_x16(&hv_0[0], taddr + (unsigned int)h_row_base_1);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_0[16];
                #pragma unroll
                for (int j_0 = 0; j_0 < 16; j_0++) {
                    hs2_0[j_0] = hv_0[j_0] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1, hs2_0);
                float hv_1[16];
                tmem_ld_x16(&hv_1[0], taddr + (unsigned int)h_row_base_1 + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_1[16];
                #pragma unroll
                for (int j_1 = 0; j_1 < 16; j_1++) {
                    hs2_1[j_1] = hv_1[j_1] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 16, hs2_1);
                float hv_2[16];
                tmem_ld_x16(&hv_2[0], taddr + (unsigned int)h_row_base_1 + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_2[16];
                #pragma unroll
                for (int j_2 = 0; j_2 < 16; j_2++) {
                    hs2_2[j_2] = hv_2[j_2] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 32, hs2_2);
                float hv_3[16];
                tmem_ld_x16(&hv_3[0], taddr + (unsigned int)h_row_base_1 + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_3[16];
                #pragma unroll
                for (int j_3 = 0; j_3 < 16; j_3++) {
                    hs2_3[j_3] = hv_3[j_3] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 48, hs2_3);
                float hv_4[16];
                tmem_ld_x16(&hv_4[0], taddr + (unsigned int)h_row_base_1 + 64);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_4[16];
                #pragma unroll
                for (int j_4 = 0; j_4 < 16; j_4++) {
                    hs2_4[j_4] = hv_4[j_4] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 64, hs2_4);
                float hv_5[16];
                tmem_ld_x16(&hv_5[0], taddr + (unsigned int)h_row_base_1 + 80);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_5[16];
                #pragma unroll
                for (int j_5 = 0; j_5 < 16; j_5++) {
                    hs2_5[j_5] = hv_5[j_5] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 80, hs2_5);
                float hv_6[16];
                tmem_ld_x16(&hv_6[0], taddr + (unsigned int)h_row_base_1 + 96);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_6[16];
                #pragma unroll
                for (int j_6 = 0; j_6 < 16; j_6++) {
                    hs2_6[j_6] = hv_6[j_6] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 96, hs2_6);
                float hv_7[16];
                tmem_ld_x16(&hv_7[0], taddr + (unsigned int)h_row_base_1 + 112);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_7[16];
                #pragma unroll
                for (int j_7 = 0; j_7 < 16; j_7++) {
                    hs2_7[j_7] = hv_7[j_7] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 112, hs2_7);
                if (nc_1 > c_1 + 1) {
                    long long gn_next_1 = ((long long)(row0_1 + c_1 * 64 + 128 - 1) * (long long)num_heads + (long long)head_1) * 128ll + (long long)k_row_1;
                    gk_pre = gk[gn_next_1];
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(dec_done_addr);
                int ext0 = ext_next;
                int clen = clen_next;
                if (nc_1 > c_1 + 1) {
                    ext_next = chunk_bos[c0_1 + c_1 + 1];
                    clen_next = chunk_len[c0_1 + c_1 + 1];
                }
                long long obase = ((long long)(ext0 + t_row_1) * (long long)num_heads + (long long)head_1) * 128ll;
                mbarrier_wait(o_done_addr, _phase_o_done_0);
                _phase_o_done_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int col_0 = lane_half_1 * 16;
                float aqh_0[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&aqh_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_0[15]))
                    : "r"(taddr + (unsigned int)o_row_base + 256));
                float aav_0[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&aav_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&aav_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&aav_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&aav_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&aav_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&aav_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&aav_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&aav_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&aav_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&aav_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&aav_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&aav_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&aav_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&aav_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&aav_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&aav_0[15]))
                    : "r"(taddr + (unsigned int)o_row_base + 384));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float ov_0[16];
                #pragma unroll
                for (int j_8 = 0; j_8 < 16; j_8++) {
                    ov_0[j_8] = __fmaf_rn(aqh_0[j_8], scale, aav_0[j_8]);
                }
                if (t_row_1 < clen) {
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
                        const unsigned* _raw_stv8_0 = reinterpret_cast<const unsigned*>(packed_0);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(o_out + (obase + (long long)col_0))), "r"(_raw_stv8_0[0]), "r"(_raw_stv8_0[1]), "r"(_raw_stv8_0[2]), "r"(_raw_stv8_0[3]), "r"(_raw_stv8_0[4]), "r"(_raw_stv8_0[5]), "r"(_raw_stv8_0[6]), "r"(_raw_stv8_0[7]) : "memory");
                    }
                }
                int col_1 = 32 + lane_half_1 * 16;
                float aqh_1[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&aqh_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_1[15]))
                    : "r"(taddr + (unsigned int)o_row_base + 288));
                float aav_1[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&aav_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&aav_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&aav_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&aav_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&aav_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&aav_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&aav_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&aav_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&aav_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&aav_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&aav_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&aav_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&aav_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&aav_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&aav_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&aav_1[15]))
                    : "r"(taddr + (unsigned int)o_row_base + 416));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float ov_1[16];
                #pragma unroll
                for (int j_9 = 0; j_9 < 16; j_9++) {
                    ov_1[j_9] = __fmaf_rn(aqh_1[j_9], scale, aav_1[j_9]);
                }
                if (t_row_1 < clen) {
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
                        const unsigned* _raw_stv8_1 = reinterpret_cast<const unsigned*>(packed_1);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(o_out + (obase + (long long)col_1))), "r"(_raw_stv8_1[0]), "r"(_raw_stv8_1[1]), "r"(_raw_stv8_1[2]), "r"(_raw_stv8_1[3]), "r"(_raw_stv8_1[4]), "r"(_raw_stv8_1[5]), "r"(_raw_stv8_1[6]), "r"(_raw_stv8_1[7]) : "memory");
                    }
                }
                int col_2 = 64 + lane_half_1 * 16;
                float aqh_2[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&aqh_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_2[15]))
                    : "r"(taddr + (unsigned int)o_row_base + 320));
                float aav_2[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&aav_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&aav_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&aav_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&aav_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&aav_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&aav_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&aav_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&aav_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&aav_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&aav_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&aav_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&aav_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&aav_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&aav_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&aav_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&aav_2[15]))
                    : "r"(taddr + (unsigned int)o_row_base + 448));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float ov_2[16];
                #pragma unroll
                for (int j_10 = 0; j_10 < 16; j_10++) {
                    ov_2[j_10] = __fmaf_rn(aqh_2[j_10], scale, aav_2[j_10]);
                }
                if (t_row_1 < clen) {
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
                        const unsigned* _raw_stv8_2 = reinterpret_cast<const unsigned*>(packed_2);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(o_out + (obase + (long long)col_2))), "r"(_raw_stv8_2[0]), "r"(_raw_stv8_2[1]), "r"(_raw_stv8_2[2]), "r"(_raw_stv8_2[3]), "r"(_raw_stv8_2[4]), "r"(_raw_stv8_2[5]), "r"(_raw_stv8_2[6]), "r"(_raw_stv8_2[7]) : "memory");
                    }
                }
                int col_3 = 96 + lane_half_1 * 16;
                float aqh_3[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&aqh_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&aqh_3[15]))
                    : "r"(taddr + (unsigned int)o_row_base + 352));
                float aav_3[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&aav_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&aav_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&aav_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&aav_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&aav_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&aav_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&aav_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&aav_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&aav_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&aav_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&aav_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&aav_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&aav_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&aav_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&aav_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&aav_3[15]))
                    : "r"(taddr + (unsigned int)o_row_base + 480));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float ov_3[16];
                #pragma unroll
                for (int j_11 = 0; j_11 < 16; j_11++) {
                    ov_3[j_11] = __fmaf_rn(aqh_3[j_11], scale, aav_3[j_11]);
                }
                if (t_row_1 < clen) {
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
                        const unsigned* _raw_stv8_3 = reinterpret_cast<const unsigned*>(packed_3);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(o_out + (obase + (long long)col_3))), "r"(_raw_stv8_3[0]), "r"(_raw_stv8_3[1]), "r"(_raw_stv8_3[2]), "r"(_raw_stv8_3[3]), "r"(_raw_stv8_3[4]), "r"(_raw_stv8_3[5]), "r"(_raw_stv8_3[6]), "r"(_raw_stv8_3[7]) : "memory");
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(o_drained_addr);
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 256;" ::: "memory");
        }
    }
    // ---- Role: load ----
    if (warp == 8) {
        { // load_main
            int head_2 = blockIdx.x % num_heads;
            int seg_2 = walk_seg[blockIdx.x / num_heads];
            int c0_2 = seq_chunk_start[seg_2];
            int nc_2 = seq_chunk_start[seg_2 + 1] - c0_2;
            int row0_2 = c0_2 * 64;
            unsigned int stage = 0;
            unsigned int _phase_tiles_empty = 1;
            unsigned int _phase_qg_empty_0 = 1;
            unsigned int _phase_aqk_empty_0 = 1;
            if (elect_sync()) {
                if (nc_2 > 0) {
                    mbarrier_wait(tiles_empty_addr + (stage) * 8, _phase_tiles_empty);
                    mbarrier_arrive_expect_tx(tiles_full_addr + (stage) * 8, 32768);
                    #pragma unroll
                    for (int p = 0; p < 2; p++) {
                        tma_3d_gmem2smem(w_panel_addr + (stage * 2 + (unsigned int)p) * 8192, (&w_tma), p * 64, head_2, row0_2, tiles_full_addr + (stage) * 8);
                        tma_3d_gmem2smem(kg_panel_addr + (stage * 2 + (unsigned int)p) * 8192, (&kg_tma), p * 64, head_2, row0_2, tiles_full_addr + (stage) * 8);
                    }
                    stage += 1;
                    if (stage == 2) { stage = 0; _phase_tiles_empty ^= 1; }
                    mbarrier_wait(qg_empty_addr, _phase_qg_empty_0);
                    _phase_qg_empty_0 ^= 1;
                    mbarrier_arrive_expect_tx(qg_full_addr, 16384);
                    #pragma unroll
                    for (int p_1 = 0; p_1 < 2; p_1++) {
                        tma_3d_gmem2smem(qg_panel_addr + (unsigned int)p_1 * 8192, (&qg_tma), p_1 * 64, head_2, row0_2, qg_full_addr);
                    }
                    if (nc_2 > 1) {
                        #pragma unroll
                        for (int p_2 = 0; p_2 < 2; p_2++) {
                            asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&qg_tma))), "r"((int)(p_2 * 64)), "r"((int)(head_2)), "r"((int)(row0_2 + 64)) : "memory");
                        }
                    }
                    mbarrier_wait(aqk_empty_addr, _phase_aqk_empty_0);
                    _phase_aqk_empty_0 ^= 1;
                    mbarrier_arrive_expect_tx(aqk_full_addr, 8192);
                    tma_3d_gmem2smem(aqk_op_addr, (&aqk_tma), 0, head_2, row0_2, aqk_full_addr);
                    if (nc_2 > 1) {
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&aqk_tma))), "r"((int)(0)), "r"((int)(head_2)), "r"((int)(row0_2 + 64)) : "memory");
                    }
                }
                #pragma unroll 1
                for (int c_2 = 0; c_2 < nc_2; c_2++) {
                    if (nc_2 > c_2 + 1) {
                        mbarrier_wait(tiles_empty_addr + (stage) * 8, _phase_tiles_empty);
                        mbarrier_arrive_expect_tx(tiles_full_addr + (stage) * 8, 32768);
                        #pragma unroll
                        for (int p_3 = 0; p_3 < 2; p_3++) {
                            tma_3d_gmem2smem(w_panel_addr + (stage * 2 + (unsigned int)p_3) * 8192, (&w_tma), p_3 * 64, head_2, row0_2 + (c_2 + 1) * 64, tiles_full_addr + (stage) * 8);
                            tma_3d_gmem2smem(kg_panel_addr + (stage * 2 + (unsigned int)p_3) * 8192, (&kg_tma), p_3 * 64, head_2, row0_2 + (c_2 + 1) * 64, tiles_full_addr + (stage) * 8);
                        }
                        stage += 1;
                        if (stage == 2) { stage = 0; _phase_tiles_empty ^= 1; }
                    }
                    if (nc_2 > c_2 + 1) {
                        mbarrier_wait(qg_empty_addr, _phase_qg_empty_0);
                        _phase_qg_empty_0 ^= 1;
                        mbarrier_arrive_expect_tx(qg_full_addr, 16384);
                        #pragma unroll
                        for (int p_4 = 0; p_4 < 2; p_4++) {
                            tma_3d_gmem2smem(qg_panel_addr + (unsigned int)p_4 * 8192, (&qg_tma), p_4 * 64, head_2, row0_2 + (c_2 + 1) * 64, qg_full_addr);
                        }
                        if (nc_2 > c_2 + 2) {
                            #pragma unroll
                            for (int p_5 = 0; p_5 < 2; p_5++) {
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&qg_tma))), "r"((int)(p_5 * 64)), "r"((int)(head_2)), "r"((int)(row0_2 + (c_2 + 1) * 64 + 64)) : "memory");
                            }
                        }
                    }
                    if (nc_2 > c_2 + 1) {
                        mbarrier_wait(aqk_empty_addr, _phase_aqk_empty_0);
                        _phase_aqk_empty_0 ^= 1;
                        mbarrier_arrive_expect_tx(aqk_full_addr, 8192);
                        tma_3d_gmem2smem(aqk_op_addr, (&aqk_tma), 0, head_2, row0_2 + (c_2 + 1) * 64, aqk_full_addr);
                        if (nc_2 > c_2 + 2) {
                            asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&aqk_tma))), "r"((int)(0)), "r"((int)(head_2)), "r"((int)(row0_2 + (c_2 + 1) * 64 + 64)) : "memory");
                        }
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 9) {
        { // mma_main
            int head_3 = blockIdx.x % num_heads;
            int seg_3 = walk_seg[blockIdx.x / num_heads];
            int c0_3 = seq_chunk_start[seg_3];
            int nc_3 = seq_chunk_start[seg_3 + 1] - c0_3;
            int row0_3 = c0_3 * 64;
            unsigned int stage_1 = 0;
            unsigned int _phase_tiles_full = 0;
            unsigned int _phase_hs_ready_0_1 = 0;
            unsigned int _phase_qg_full_0 = 0;
            unsigned int _phase_o_drained_0 = 1;
            unsigned int _phase_bv_ready_0 = 0;
            unsigned int _phase_dec_done_0 = 0;
            unsigned int _phase_aqk_full_0 = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int c_3 = 0; c_3 < nc_3; c_3++) {
                    mbarrier_wait(tiles_full_addr + (stage_1) * 8, _phase_tiles_full);
                    mbarrier_wait(hs_ready_addr, _phase_hs_ready_0_1);
                    _phase_hs_ready_0_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int a_lo_1 = (((w_op_addr) >> 4) & 0x7FFF) + (stage_1) * 1024;
                    int b_lo_1 = (((hs_mn_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
                    int a_lo_1 = (((w_op_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                    int b_lo_1 = (((hs_mn_addr) >> 4) & 0x3FFF) | 0x4000000;
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
                    :: "r"(a_lo_1), "r"(b_lo_1), "r"(tmem_acc_vp), "r"(0) : "memory");
                    tcgen05_commit(vp_ready_addr);
                    mbarrier_wait(qg_full_addr, _phase_qg_full_0);
                    _phase_qg_full_0 ^= 1;
                    mbarrier_wait(o_drained_addr, _phase_o_drained_0);
                    _phase_o_drained_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_0 = ((qg_op_addr) >> 4) & 0x7FFF;
#else
                    int _smem_desc_lo_0 = ((qg_op_addr) >> 4) & 0x3FFF;
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
                    :: "r"(_smem_desc_lo_0), "r"(b_lo_1), "r"(tmem_acc_qh), "r"(0) : "memory");
                    tcgen05_commit(qg_empty_addr);
                    mbarrier_wait(bv_ready_addr, _phase_bv_ready_0);
                    _phase_bv_ready_0 ^= 1;
                    mbarrier_wait(dec_done_addr, _phase_dec_done_0);
                    _phase_dec_done_0 ^= 1;
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(h_ready_addr);
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int a_lo_2 = ((((kg_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (stage_1) * 1024;
                    int b_lo_2 = (((bv_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                    int a_lo_2 = ((((kg_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (stage_1) * 1024;
                    int b_lo_2 = (((bv_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    :: "r"(a_lo_2), "r"(b_lo_2), "r"(tmem_acc_h), "r"(1) : "memory");
                    tcgen05_commit(h_ready_addr);
                    tcgen05_commit(tiles_empty_addr + (stage_1) * 8);
                    mbarrier_wait(aqk_full_addr, _phase_aqk_full_0);
                    _phase_aqk_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_1 = ((aqk_op_addr) >> 4) & 0x7FFF;
#else
                    int _smem_desc_lo_1 = ((aqk_op_addr) >> 4) & 0x3FFF;
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
                    "}\n"
                    :: "r"(_smem_desc_lo_1), "r"(b_lo_2), "r"(tmem_acc_av), "r"(0) : "memory");
                    tcgen05_commit(aqk_empty_addr);
                    tcgen05_commit(o_done_addr);
                    stage_1 += 1;
                    if (stage_1 == 2) { stage_1 = 0; _phase_tiles_full ^= 1; }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
