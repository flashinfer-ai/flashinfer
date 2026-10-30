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

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 128
#define TMEM_ACC_H_OFFSET 0
#define TMEM_ACC_VP_OFFSET 64
#define NUM_RING_STAGES 2
#define NUM_ONE_STAGES 1
#define SMEM_W_PANEL_OFF 1024
#define SMEM_W_PANEL_STAGE_BYTES 8192
#define SMEM_W_PANEL_STRIDE 8192
#define SMEM_KG_PANEL_OFF 33792
#define SMEM_KG_PANEL_STAGE_BYTES 8192
#define SMEM_KG_PANEL_STRIDE 8192
#define SMEM_W_OP_OFF 1024
#define SMEM_W_OP_STAGE_BYTES 16384
#define SMEM_W_OP_STRIDE 16384
#define SMEM_KG_MN_OFF 33792
#define SMEM_KG_MN_STAGE_BYTES 16384
#define SMEM_KG_MN_STRIDE 16384
#define SMEM_HS_OFF 66560
#define SMEM_HS_STAGE_BYTES 16384
#define SMEM_HS_STRIDE 16384
#define SMEM_HS_MN_OFF 66560
#define SMEM_HS_MN_STAGE_BYTES 16384
#define SMEM_HS_MN_STRIDE 16384
#define SMEM_BV_OFF 82944
#define SMEM_BV_STAGE_BYTES 8192
#define SMEM_BV_STRIDE 8192
#define SMEM_BV_MN_OFF 82944
#define SMEM_BV_MN_STAGE_BYTES 8192
#define SMEM_BV_MN_STRIDE 8192
#define SMEM_TOTAL 91136
#define THREADS 192
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
kernel_cake_kda_train_5a20cd12bb62bb6f3ac5(const __grid_constant__ CUtensorMap w_tma, const __grid_constant__ CUtensorMap kg_tma, __nv_bfloat16* __restrict__ u, float* __restrict__ gk, int num_heads, int* __restrict__ seq_chunk_start, int num_seqs, int* __restrict__ walk_seg, float* __restrict__ h0, int* __restrict__ h0_index, int h0_hg, int h0_n, float* __restrict__ out, int* __restrict__ out_index, int out_hg, int out_n, int tile_stride, __nv_bfloat16* __restrict__ h_tape, __nv_bfloat16* __restrict__ v_new)
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
    #define h_ready_addr (mbar_base + 56)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
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
    __nv_bfloat16* bv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 82944);
    const int bv_addr = smem + 82944;
    __nv_bfloat16* bv_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 82944);
    const int bv_mn_addr = smem + 82944;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 8 barriers)
    // Mbarriers at smem_raw[0..64)

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
            // h_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 128 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 64);
    if (warp == 0) {
        int _tmem_hold = smem + 64;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc_h = taddr;
    const int tmem_acc_vp = taddr + 64;

    // ---- Role: compute ----
    if (warp <= 3) {
        { // compute_main
            int head = blockIdx.x / 2 % num_heads;
            int walk = blockIdx.x / (2 * num_heads);
            int seg = walk_seg[walk];
            int c0 = seq_chunk_start[seg];
            int nc = seq_chunk_start[seg + 1] - c0;
            int row0 = c0 * 64;
            int v0 = blockIdx.x % 2 * 64;
            long long v0_i64 = (long long)v0;
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
                long long h0_base = (long long)h0_tile * (long long)tile_stride * 16384 + (long long)(k_row * 128 + v0);
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
            unsigned int _phase_vp_ready_0 = 0;
            unsigned int _phase_h_ready_0 = 0;
            #pragma unroll 1
            for (int c = 0; c < nc; c++) {
                int trow_pf = row0 + c * 64 + t_row;
                long long u_base_pf = ((long long)trow_pf * (long long)num_heads + (long long)head) * 128 + v0_i64;
                int col_pf = lane_half * 16;
                float uv_pf[16];
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(u + u_base_pf + (long long)col_pf);
                    uint4 _vld_1[2];
                    #pragma unroll
                    for (int _blk = 0; _blk < 2; _blk++) {
                        _vld_1[_blk] = _vptr_1[_blk];
                        uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&uv_pf[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                            (&uv_pf[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                        }
                    }
                }
                int col_pf_0 = 32 + lane_half * 16;
                float uv_pf_1[16];
                {
                    const uint4* _vptr_2 = reinterpret_cast<const uint4*>(u + u_base_pf + (long long)col_pf_0);
                    uint4 _vld_2[2];
                    #pragma unroll
                    for (int _blk = 0; _blk < 2; _blk++) {
                        _vld_2[_blk] = _vptr_2[_blk];
                        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&uv_pf_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                            (&uv_pf_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                        }
                    }
                }
                long long gn_index_pf = ((long long)(row0 + c * 64 + 64 - 1) * (long long)num_heads + (long long)head) * 128 + (long long)k_row;
                float decay_pf = approx_exp2(gk[gn_index_pf]);
                float hv_b[16];
                tmem_ld_x16(&hv_b[0], taddr + (unsigned int)h_row_base);
                float hv_b_2[16];
                tmem_ld_x16(&hv_b_2[0], taddr + (unsigned int)h_row_base + 16);
                float hv_b_3[16];
                tmem_ld_x16(&hv_b_3[0], taddr + (unsigned int)h_row_base + 32);
                float hv_b_4[16];
                tmem_ld_x16(&hv_b_4[0], taddr + (unsigned int)h_row_base + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                unsigned int hv_b_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(hv_b[_lp*2 + 0], hv_b[_lp*2+1 + 0]));
                    hv_b_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_1 = 0; q_1 < 2; q_1++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)(q_1 * 8 / 64 * 16384 + k_row * 128 + q_1 * 8 % 64 * 2 ^ (q_1 * 8 / 64 * 16384 + k_row * 128 + q_1 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(hv_b_bf16[q_1 * 4]), "r"(hv_b_bf16[q_1 * 4 + 1]), "r"(hv_b_bf16[q_1 * 4 + 2]), "r"(hv_b_bf16[q_1 * 4 + 3]) : "memory");
                }
                unsigned int hv_b_2_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(hv_b_2[_lp*2 + 0], hv_b_2[_lp*2+1 + 0]));
                    hv_b_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_2 = 0; q_2 < 2; q_2++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((16 + q_2 * 8) / 64 * 16384 + k_row * 128 + (16 + q_2 * 8) % 64 * 2 ^ ((16 + q_2 * 8) / 64 * 16384 + k_row * 128 + (16 + q_2 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(hv_b_2_bf16[q_2 * 4]), "r"(hv_b_2_bf16[q_2 * 4 + 1]), "r"(hv_b_2_bf16[q_2 * 4 + 2]), "r"(hv_b_2_bf16[q_2 * 4 + 3]) : "memory");
                }
                unsigned int hv_b_3_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(hv_b_3[_lp*2 + 0], hv_b_3[_lp*2+1 + 0]));
                    hv_b_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_3 = 0; q_3 < 2; q_3++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((32 + q_3 * 8) / 64 * 16384 + k_row * 128 + (32 + q_3 * 8) % 64 * 2 ^ ((32 + q_3 * 8) / 64 * 16384 + k_row * 128 + (32 + q_3 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(hv_b_3_bf16[q_3 * 4]), "r"(hv_b_3_bf16[q_3 * 4 + 1]), "r"(hv_b_3_bf16[q_3 * 4 + 2]), "r"(hv_b_3_bf16[q_3 * 4 + 3]) : "memory");
                }
                unsigned int hv_b_4_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(hv_b_4[_lp*2 + 0], hv_b_4[_lp*2+1 + 0]));
                    hv_b_4_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_4 = 0; q_4 < 2; q_4++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((48 + q_4 * 8) / 64 * 16384 + k_row * 128 + (48 + q_4 * 8) % 64 * 2 ^ ((48 + q_4 * 8) / 64 * 16384 + k_row * 128 + (48 + q_4 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(hv_b_4_bf16[q_4 * 4]), "r"(hv_b_4_bf16[q_4 * 4 + 1]), "r"(hv_b_4_bf16[q_4 * 4 + 2]), "r"(hv_b_4_bf16[q_4 * 4 + 3]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(hs_ready_addr);
                mbarrier_wait(vp_ready_addr, _phase_vp_ready_0);
                _phase_vp_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_0[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15]))
                    : "r"(taddr + 64 + (unsigned int)vp_row_base));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col = lane_half * 16;
                float bvv[16];
                #pragma unroll
                for (int j = 0; j < 16; j++) {
                    bvv[j] = uv_pf[j] - _tmem_load_0[j];
                }
                unsigned int bvv_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv[_lp*2 + 0], bvv[_lp*2+1 + 0]));
                    bvv_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_5 = 0; q_5 < 2; q_5++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv_addr + (unsigned int)((col + q_5 * 8) / 64 * 8192 + t_row * 128 + (col + q_5 * 8) % 64 * 2 ^ ((col + q_5 * 8) / 64 * 8192 + t_row * 128 + (col + q_5 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_bf16[q_5 * 4]), "r"(bvv_bf16[q_5 * 4 + 1]), "r"(bvv_bf16[q_5 * 4 + 2]), "r"(bvv_bf16[q_5 * 4 + 3]) : "memory");
                }
                float _tmem_load_1[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15]))
                    : "r"(taddr + 64 + (unsigned int)vp_row_base + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col_5 = 32 + lane_half * 16;
                float bvv_6[16];
                #pragma unroll
                for (int j_1 = 0; j_1 < 16; j_1++) {
                    bvv_6[j_1] = uv_pf_1[j_1] - _tmem_load_1[j_1];
                }
                unsigned int bvv_6_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_6[_lp*2 + 0], bvv_6[_lp*2+1 + 0]));
                    bvv_6_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_6 = 0; q_6 < 2; q_6++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv_addr + (unsigned int)((col_5 + q_6 * 8) / 64 * 8192 + t_row * 128 + (col_5 + q_6 * 8) % 64 * 2 ^ ((col_5 + q_6 * 8) / 64 * 8192 + t_row * 128 + (col_5 + q_6 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_6_bf16[q_6 * 4]), "r"(bvv_6_bf16[q_6 * 4 + 1]), "r"(bvv_6_bf16[q_6 * 4 + 2]), "r"(bvv_6_bf16[q_6 * 4 + 3]) : "memory");
                }
                float hv2_b[16];
                tmem_ld_x16(&hv2_b[0], taddr + (unsigned int)h_row_base);
                float hv2_b_7[16];
                tmem_ld_x16(&hv2_b_7[0], taddr + (unsigned int)h_row_base + 16);
                float hv2_b_8[16];
                tmem_ld_x16(&hv2_b_8[0], taddr + (unsigned int)h_row_base + 32);
                float hv2_b_9[16];
                tmem_ld_x16(&hv2_b_9[0], taddr + (unsigned int)h_row_base + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float hs2[16];
                #pragma unroll
                for (int j_2 = 0; j_2 < 16; j_2++) {
                    hs2[j_2] = hv2_b[j_2] * decay_pf;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base, hs2);
                float hs2_10[16];
                #pragma unroll
                for (int j_3 = 0; j_3 < 16; j_3++) {
                    hs2_10[j_3] = hv2_b_7[j_3] * decay_pf;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 16, hs2_10);
                float hs2_11[16];
                #pragma unroll
                for (int j_4 = 0; j_4 < 16; j_4++) {
                    hs2_11[j_4] = hv2_b_8[j_4] * decay_pf;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 32, hs2_11);
                float hs2_12[16];
                #pragma unroll
                for (int j_5 = 0; j_5 < 16; j_5++) {
                    hs2_12[j_5] = hv2_b_9[j_5] * decay_pf;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 48, hs2_12);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(bv_ready_addr);
                mbarrier_wait(h_ready_addr, _phase_h_ready_0);
                _phase_h_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
            }
            int out_slot = out_index[seg];
            if (out_slot >= 0) {
                int out_tile = (head / out_hg * out_n + out_slot) * out_hg + head % out_hg;
                long long out_base = (long long)out_tile * (long long)tile_stride * 16384 + (long long)(k_row * 128 + v0);
                #pragma unroll
                for (int blk_2 = 0; blk_2 < 4; blk_2++) {
                    float _tmem_load_2[16];
                    tmem_ld_x16(&_tmem_load_2[0], taddr + (unsigned int)h_row_base + (unsigned int)(blk_2 * 16));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int q_7 = 0; q_7 < 4; q_7++) {
                        {
                            float4 _v4 = make_float4(_tmem_load_2[q_7 * 4 + 0], _tmem_load_2[q_7 * 4 + 1], _tmem_load_2[q_7 * 4 + 2], _tmem_load_2[q_7 * 4 + 3]);
                            *reinterpret_cast<float4*>(out + out_base + (long long)(blk_2 * 16 + q_7 * 4)) = _v4;
                        }
                    }
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
            }
        }
    }
    // ---- Role: load ----
    if (warp == 4) {
        { // load_main
            int head_1 = blockIdx.x / 2 % num_heads;
            int walk_1 = blockIdx.x / (2 * num_heads);
            int seg_1 = walk_seg[walk_1];
            int c0_1 = seq_chunk_start[seg_1];
            int nc_1 = seq_chunk_start[seg_1 + 1] - c0_1;
            int row0_1 = c0_1 * 64;
            unsigned int stage = 0;
            unsigned int _phase_tiles_empty = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int c_1 = 0; c_1 < nc_1; c_1++) {
                    mbarrier_wait(tiles_empty_addr + (stage) * 8, _phase_tiles_empty);
                    mbarrier_arrive_expect_tx(tiles_full_addr + (stage) * 8, 32768);
                    int trow = row0_1 + c_1 * 64;
                    #pragma unroll
                    for (int panel = 0; panel < 2; panel++) {
                        tma_3d_gmem2smem(w_panel_addr + (stage * 2 + (unsigned int)panel) * 8192, (&w_tma), panel * 64, head_1, trow, tiles_full_addr + (stage) * 8);
                        tma_3d_gmem2smem(kg_panel_addr + (stage * 2 + (unsigned int)panel) * 8192, (&kg_tma), panel * 64, head_1, trow, tiles_full_addr + (stage) * 8);
                    }
                    stage += 1;
                    if (stage == 2) { stage = 0; _phase_tiles_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 5) {
        { // mma_main
            int head_2 = blockIdx.x / 2 % num_heads;
            int walk_2 = blockIdx.x / (2 * num_heads);
            int seg_2 = walk_seg[walk_2];
            int c0_2 = seq_chunk_start[seg_2];
            int nc_2 = seq_chunk_start[seg_2 + 1] - c0_2;
            int row0_2 = c0_2 * 64;
            unsigned int stage_1 = 0;
            unsigned int _phase_tiles_full = 0;
            unsigned int _phase_hs_ready_0 = 0;
            unsigned int _phase_bv_ready_0 = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int c_2 = 0; c_2 < nc_2; c_2++) {
                    mbarrier_wait(tiles_full_addr + (stage_1) * 8, _phase_tiles_full);
                    mbarrier_wait(hs_ready_addr, _phase_hs_ready_0);
                    _phase_hs_ready_0 ^= 1;
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
                    mbarrier_wait(bv_ready_addr, _phase_bv_ready_0);
                    _phase_bv_ready_0 ^= 1;
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
