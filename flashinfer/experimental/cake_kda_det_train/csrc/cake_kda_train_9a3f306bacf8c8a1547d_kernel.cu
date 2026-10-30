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
#define TMEM_NCOLS 192
#define TMEM_ACC_H_OFFSET 0
#define TMEM_ACC_VP_OFFSET 64
#define TMEM_ACC_DX_OFFSET 128
#define NUM_RING_PIPE_STAGES 1
#define NUM_ONE_STAGES 1
#define SMEM_W_PANEL_OFF 1024
#define SMEM_W_PANEL_STAGE_BYTES 8192
#define SMEM_W_PANEL_STRIDE 8192
#define SMEM_KG_PANEL_OFF 17408
#define SMEM_KG_PANEL_STAGE_BYTES 8192
#define SMEM_KG_PANEL_STRIDE 8192
#define SMEM_W_OP_OFF 1024
#define SMEM_W_OP_STAGE_BYTES 16384
#define SMEM_W_OP_STRIDE 16384
#define SMEM_KG_MN_OFF 17408
#define SMEM_KG_MN_STAGE_BYTES 16384
#define SMEM_KG_MN_STRIDE 16384
#define SMEM_HS0_OFF 33792
#define SMEM_HS0_STAGE_BYTES 16384
#define SMEM_HS0_STRIDE 16384
#define SMEM_HS1_OFF 50176
#define SMEM_HS1_STAGE_BYTES 16384
#define SMEM_HS1_STRIDE 16384
#define SMEM_HS2_OFF 66560
#define SMEM_HS2_STAGE_BYTES 16384
#define SMEM_HS2_STRIDE 16384
#define SMEM_HS_MN0_OFF 33792
#define SMEM_HS_MN0_STAGE_BYTES 16384
#define SMEM_HS_MN0_STRIDE 16384
#define SMEM_HS_MN1_OFF 50176
#define SMEM_HS_MN1_STAGE_BYTES 16384
#define SMEM_HS_MN1_STRIDE 16384
#define SMEM_HS_MN2_OFF 66560
#define SMEM_HS_MN2_STAGE_BYTES 16384
#define SMEM_HS_MN2_STRIDE 16384
#define SMEM_BV0_OFF 82944
#define SMEM_BV0_STAGE_BYTES 8192
#define SMEM_BV0_STRIDE 8192
#define SMEM_BV1_OFF 91136
#define SMEM_BV1_STAGE_BYTES 8192
#define SMEM_BV1_STRIDE 8192
#define SMEM_BV2_OFF 99328
#define SMEM_BV2_STAGE_BYTES 8192
#define SMEM_BV2_STRIDE 8192
#define SMEM_BV_MN0_OFF 82944
#define SMEM_BV_MN0_STAGE_BYTES 8192
#define SMEM_BV_MN0_STRIDE 8192
#define SMEM_BV_MN1_OFF 91136
#define SMEM_BV_MN1_STAGE_BYTES 8192
#define SMEM_BV_MN1_STRIDE 8192
#define SMEM_BV_MN2_OFF 99328
#define SMEM_BV_MN2_STAGE_BYTES 8192
#define SMEM_BV_MN2_STRIDE 8192
#define SMEM_TOTAL 107520
#define THREADS 192
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

__global__ __launch_bounds__(192, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_train_9a3f306bacf8c8a1547d(const __grid_constant__ CUtensorMap w_tma, const __grid_constant__ CUtensorMap kg_tma, __nv_bfloat16* __restrict__ u, float* __restrict__ gk, int num_heads, int* __restrict__ seq_chunk_start, int num_seqs, int* __restrict__ walk_seg, float* __restrict__ out, int* __restrict__ out_index, int out_hg, int out_n, int tile_stride, int init_mode, int use_u, int write_transposed)
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
    #define tiles_empty_addr (mbar_base + 8)
    #define hs_ready_addr (mbar_base + 16)
    #define vp_ready_addr (mbar_base + 24)
    #define bv_ready_addr (mbar_base + 32)
    #define h_ready_addr (mbar_base + 40)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* w_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int w_panel_addr = smem + 1024;
    __nv_bfloat16* kg_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int kg_panel_addr = smem + 17408;
    __nv_bfloat16* w_op = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int w_op_addr = smem + 1024;
    __nv_bfloat16* kg_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int kg_mn_addr = smem + 17408;
    __nv_bfloat16* hs0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int hs0_addr = smem + 33792;
    __nv_bfloat16* hs1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 50176);
    const int hs1_addr = smem + 50176;
    __nv_bfloat16* hs2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int hs2_addr = smem + 66560;
    __nv_bfloat16* hs_mn0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int hs_mn0_addr = smem + 33792;
    __nv_bfloat16* hs_mn1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 50176);
    const int hs_mn1_addr = smem + 50176;
    __nv_bfloat16* hs_mn2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int hs_mn2_addr = smem + 66560;
    __nv_bfloat16* bv0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 82944);
    const int bv0_addr = smem + 82944;
    __nv_bfloat16* bv1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 91136);
    const int bv1_addr = smem + 91136;
    __nv_bfloat16* bv2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int bv2_addr = smem + 99328;
    __nv_bfloat16* bv_mn0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 82944);
    const int bv_mn0_addr = smem + 82944;
    __nv_bfloat16* bv_mn1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 91136);
    const int bv_mn1_addr = smem + 91136;
    __nv_bfloat16* bv_mn2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int bv_mn2_addr = smem + 99328;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ring_pipe' ---
            // tiles_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // tiles_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'one' ---
            // hs_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 16, 128);
            // vp_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            // bv_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 32, 128);
            // h_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 192 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 0) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc_h = taddr;
    const int tmem_acc_vp = taddr + 64;
    const int tmem_acc_dx = taddr + 128;

    // ---- Role: compute ----
    if (warp <= 3) {
        { // compute_main
            int vslice = blockIdx.x % 2;
            int head = blockIdx.x / 2 % num_heads;
            int seg = walk_seg[blockIdx.x / (2 * num_heads)];
            int c0 = seq_chunk_start[seg];
            int nc = seq_chunk_start[seg + 1] - c0;
            int row0 = c0 * 64;
            int warp_in_wg = warp % 4;
            int k_row = warp_in_wg * 32 + lane;
            int h_row_base = warp_in_wg * 32 << 16;
            int lane_half = lane >> 4;
            int t_row = warp_in_wg * 16 + (lane & 15);
            int vp_row_base = warp_in_wg * 32 << 16;
            int v0 = vslice * 64;
            unsigned int mask_hi = 4294901760u;
            float init_row[16];
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                float one_here = 0.0f;
                if (init_mode == 0) {
                    if (v0 + j == k_row) {
                        one_here = 1.0f;
                    }
                }
                init_row[j] = one_here;
            }
            tmem_st_x16_f32(taddr + (unsigned int)h_row_base, init_row);
            #pragma unroll
            for (int j_1 = 0; j_1 < 16; j_1++) {
                float one_here_1 = 0.0f;
                if (init_mode == 0) {
                    if (v0 + 16 + j_1 == k_row) {
                        one_here_1 = 1.0f;
                    }
                }
                init_row[j_1] = one_here_1;
            }
            tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 16, init_row);
            #pragma unroll
            for (int j_2 = 0; j_2 < 16; j_2++) {
                float one_here_2 = 0.0f;
                if (init_mode == 0) {
                    if (v0 + 32 + j_2 == k_row) {
                        one_here_2 = 1.0f;
                    }
                }
                init_row[j_2] = one_here_2;
            }
            tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 32, init_row);
            #pragma unroll
            for (int j_3 = 0; j_3 < 16; j_3++) {
                float one_here_3 = 0.0f;
                if (init_mode == 0) {
                    if (v0 + 48 + j_3 == k_row) {
                        one_here_3 = 1.0f;
                    }
                }
                init_row[j_3] = one_here_3;
            }
            tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 48, init_row);
            float zeros[16];
            zeros[0] = 0.0f;
            zeros[1] = 0.0f;
            zeros[2] = 0.0f;
            zeros[3] = 0.0f;
            zeros[4] = 0.0f;
            zeros[5] = 0.0f;
            zeros[6] = 0.0f;
            zeros[7] = 0.0f;
            zeros[8] = 0.0f;
            zeros[9] = 0.0f;
            zeros[10] = 0.0f;
            zeros[11] = 0.0f;
            zeros[12] = 0.0f;
            zeros[13] = 0.0f;
            zeros[14] = 0.0f;
            zeros[15] = 0.0f;
            tmem_st_x16_f32(taddr + 128 + (unsigned int)h_row_base, zeros);
            tmem_st_x16_f32(taddr + 128 + (unsigned int)h_row_base + 16, zeros);
            tmem_st_x16_f32(taddr + 128 + (unsigned int)h_row_base + 32, zeros);
            tmem_st_x16_f32(taddr + 128 + (unsigned int)h_row_base + 48, zeros);
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            unsigned int _phase_vp_ready_0 = 0;
            unsigned int _phase_h_ready_0 = 0;
            #pragma unroll 1
            for (int c = 0; c < nc; c++) {
                long long gn_index = ((long long)(row0 + c * 64 + 64 - 1) * (long long)num_heads + (long long)head) * 128 + (long long)k_row;
                float decay = approx_exp2(gk[gn_index]);
                float _tmem_load_0[16];
                tmem_ld_x16(&_tmem_load_0[0], taddr + (unsigned int)h_row_base);
                float _tmem_load_1[16];
                tmem_ld_x16(&_tmem_load_1[0], taddr + 128 + (unsigned int)h_row_base);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_4 = 0; j_4 < 16; j_4++) {
                    _tmem_load_0[j_4] = _tmem_load_0[j_4] + _tmem_load_1[j_4];
                }
                unsigned int _tmem_load_0_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                    _tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r1[16];
                r1[0] = _tmem_load_0[0] - __uint_as_float(_tmem_load_0_bf16[0] << 16);
                r1[1] = _tmem_load_0[1] - __uint_as_float(_tmem_load_0_bf16[0] & mask_hi);
                r1[2] = _tmem_load_0[2] - __uint_as_float(_tmem_load_0_bf16[1] << 16);
                r1[3] = _tmem_load_0[3] - __uint_as_float(_tmem_load_0_bf16[1] & mask_hi);
                r1[4] = _tmem_load_0[4] - __uint_as_float(_tmem_load_0_bf16[2] << 16);
                r1[5] = _tmem_load_0[5] - __uint_as_float(_tmem_load_0_bf16[2] & mask_hi);
                r1[6] = _tmem_load_0[6] - __uint_as_float(_tmem_load_0_bf16[3] << 16);
                r1[7] = _tmem_load_0[7] - __uint_as_float(_tmem_load_0_bf16[3] & mask_hi);
                r1[8] = _tmem_load_0[8] - __uint_as_float(_tmem_load_0_bf16[4] << 16);
                r1[9] = _tmem_load_0[9] - __uint_as_float(_tmem_load_0_bf16[4] & mask_hi);
                r1[10] = _tmem_load_0[10] - __uint_as_float(_tmem_load_0_bf16[5] << 16);
                r1[11] = _tmem_load_0[11] - __uint_as_float(_tmem_load_0_bf16[5] & mask_hi);
                r1[12] = _tmem_load_0[12] - __uint_as_float(_tmem_load_0_bf16[6] << 16);
                r1[13] = _tmem_load_0[13] - __uint_as_float(_tmem_load_0_bf16[6] & mask_hi);
                r1[14] = _tmem_load_0[14] - __uint_as_float(_tmem_load_0_bf16[7] << 16);
                r1[15] = _tmem_load_0[15] - __uint_as_float(_tmem_load_0_bf16[7] & mask_hi);
                unsigned int r1_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r1[_lp*2 + 0], r1[_lp*2+1 + 0]));
                    r1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r2[16];
                r2[0] = r1[0] - __uint_as_float(r1_bf16[0] << 16);
                r2[1] = r1[1] - __uint_as_float(r1_bf16[0] & mask_hi);
                r2[2] = r1[2] - __uint_as_float(r1_bf16[1] << 16);
                r2[3] = r1[3] - __uint_as_float(r1_bf16[1] & mask_hi);
                r2[4] = r1[4] - __uint_as_float(r1_bf16[2] << 16);
                r2[5] = r1[5] - __uint_as_float(r1_bf16[2] & mask_hi);
                r2[6] = r1[6] - __uint_as_float(r1_bf16[3] << 16);
                r2[7] = r1[7] - __uint_as_float(r1_bf16[3] & mask_hi);
                r2[8] = r1[8] - __uint_as_float(r1_bf16[4] << 16);
                r2[9] = r1[9] - __uint_as_float(r1_bf16[4] & mask_hi);
                r2[10] = r1[10] - __uint_as_float(r1_bf16[5] << 16);
                r2[11] = r1[11] - __uint_as_float(r1_bf16[5] & mask_hi);
                r2[12] = r1[12] - __uint_as_float(r1_bf16[6] << 16);
                r2[13] = r1[13] - __uint_as_float(r1_bf16[6] & mask_hi);
                r2[14] = r1[14] - __uint_as_float(r1_bf16[7] << 16);
                r2[15] = r1[15] - __uint_as_float(r1_bf16[7] & mask_hi);
                unsigned int r2_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r2[_lp*2 + 0], r2[_lp*2+1 + 0]));
                    r2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(k_row * 128 ^ (k_row * 128 >> 7 & 7) << 4))), "r"(_tmem_load_0_bf16[0]), "r"(_tmem_load_0_bf16[1]), "r"(_tmem_load_0_bf16[2]), "r"(_tmem_load_0_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(k_row * 128 + 16 ^ (k_row * 128 + 16 >> 7 & 7) << 4))), "r"(_tmem_load_0_bf16[4]), "r"(_tmem_load_0_bf16[5]), "r"(_tmem_load_0_bf16[6]), "r"(_tmem_load_0_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(k_row * 128 ^ (k_row * 128 >> 7 & 7) << 4))), "r"(r1_bf16[0]), "r"(r1_bf16[1]), "r"(r1_bf16[2]), "r"(r1_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(k_row * 128 + 16 ^ (k_row * 128 + 16 >> 7 & 7) << 4))), "r"(r1_bf16[4]), "r"(r1_bf16[5]), "r"(r1_bf16[6]), "r"(r1_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(k_row * 128 ^ (k_row * 128 >> 7 & 7) << 4))), "r"(r2_bf16[0]), "r"(r2_bf16[1]), "r"(r2_bf16[2]), "r"(r2_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(k_row * 128 + 16 ^ (k_row * 128 + 16 >> 7 & 7) << 4))), "r"(r2_bf16[4]), "r"(r2_bf16[5]), "r"(r2_bf16[6]), "r"(r2_bf16[7]) : "memory");
                float hs2_0[16];
                #pragma unroll
                for (int j_5 = 0; j_5 < 16; j_5++) {
                    hs2_0[j_5] = _tmem_load_0[j_5] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base, hs2_0);
                float _tmem_load_2[16];
                tmem_ld_x16(&_tmem_load_2[0], taddr + (unsigned int)h_row_base + 16);
                float _tmem_load_3[16];
                tmem_ld_x16(&_tmem_load_3[0], taddr + 128 + (unsigned int)h_row_base + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_6 = 0; j_6 < 16; j_6++) {
                    _tmem_load_2[j_6] = _tmem_load_2[j_6] + _tmem_load_3[j_6];
                }
                unsigned int _tmem_load_2_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 0], _tmem_load_2[_lp*2+1 + 0]));
                    _tmem_load_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r1_1[16];
                r1_1[0] = _tmem_load_2[0] - __uint_as_float(_tmem_load_2_bf16[0] << 16);
                r1_1[1] = _tmem_load_2[1] - __uint_as_float(_tmem_load_2_bf16[0] & mask_hi);
                r1_1[2] = _tmem_load_2[2] - __uint_as_float(_tmem_load_2_bf16[1] << 16);
                r1_1[3] = _tmem_load_2[3] - __uint_as_float(_tmem_load_2_bf16[1] & mask_hi);
                r1_1[4] = _tmem_load_2[4] - __uint_as_float(_tmem_load_2_bf16[2] << 16);
                r1_1[5] = _tmem_load_2[5] - __uint_as_float(_tmem_load_2_bf16[2] & mask_hi);
                r1_1[6] = _tmem_load_2[6] - __uint_as_float(_tmem_load_2_bf16[3] << 16);
                r1_1[7] = _tmem_load_2[7] - __uint_as_float(_tmem_load_2_bf16[3] & mask_hi);
                r1_1[8] = _tmem_load_2[8] - __uint_as_float(_tmem_load_2_bf16[4] << 16);
                r1_1[9] = _tmem_load_2[9] - __uint_as_float(_tmem_load_2_bf16[4] & mask_hi);
                r1_1[10] = _tmem_load_2[10] - __uint_as_float(_tmem_load_2_bf16[5] << 16);
                r1_1[11] = _tmem_load_2[11] - __uint_as_float(_tmem_load_2_bf16[5] & mask_hi);
                r1_1[12] = _tmem_load_2[12] - __uint_as_float(_tmem_load_2_bf16[6] << 16);
                r1_1[13] = _tmem_load_2[13] - __uint_as_float(_tmem_load_2_bf16[6] & mask_hi);
                r1_1[14] = _tmem_load_2[14] - __uint_as_float(_tmem_load_2_bf16[7] << 16);
                r1_1[15] = _tmem_load_2[15] - __uint_as_float(_tmem_load_2_bf16[7] & mask_hi);
                unsigned int r1_1_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r1_1[_lp*2 + 0], r1_1[_lp*2+1 + 0]));
                    r1_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r2_2[16];
                r2_2[0] = r1_1[0] - __uint_as_float(r1_1_bf16[0] << 16);
                r2_2[1] = r1_1[1] - __uint_as_float(r1_1_bf16[0] & mask_hi);
                r2_2[2] = r1_1[2] - __uint_as_float(r1_1_bf16[1] << 16);
                r2_2[3] = r1_1[3] - __uint_as_float(r1_1_bf16[1] & mask_hi);
                r2_2[4] = r1_1[4] - __uint_as_float(r1_1_bf16[2] << 16);
                r2_2[5] = r1_1[5] - __uint_as_float(r1_1_bf16[2] & mask_hi);
                r2_2[6] = r1_1[6] - __uint_as_float(r1_1_bf16[3] << 16);
                r2_2[7] = r1_1[7] - __uint_as_float(r1_1_bf16[3] & mask_hi);
                r2_2[8] = r1_1[8] - __uint_as_float(r1_1_bf16[4] << 16);
                r2_2[9] = r1_1[9] - __uint_as_float(r1_1_bf16[4] & mask_hi);
                r2_2[10] = r1_1[10] - __uint_as_float(r1_1_bf16[5] << 16);
                r2_2[11] = r1_1[11] - __uint_as_float(r1_1_bf16[5] & mask_hi);
                r2_2[12] = r1_1[12] - __uint_as_float(r1_1_bf16[6] << 16);
                r2_2[13] = r1_1[13] - __uint_as_float(r1_1_bf16[6] & mask_hi);
                r2_2[14] = r1_1[14] - __uint_as_float(r1_1_bf16[7] << 16);
                r2_2[15] = r1_1[15] - __uint_as_float(r1_1_bf16[7] & mask_hi);
                unsigned int r2_2_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r2_2[_lp*2 + 0], r2_2[_lp*2+1 + 0]));
                    r2_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(k_row * 128 + 32 ^ (k_row * 128 + 32 >> 7 & 7) << 4))), "r"(_tmem_load_2_bf16[0]), "r"(_tmem_load_2_bf16[1]), "r"(_tmem_load_2_bf16[2]), "r"(_tmem_load_2_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(k_row * 128 + 48 ^ (k_row * 128 + 48 >> 7 & 7) << 4))), "r"(_tmem_load_2_bf16[4]), "r"(_tmem_load_2_bf16[5]), "r"(_tmem_load_2_bf16[6]), "r"(_tmem_load_2_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(k_row * 128 + 32 ^ (k_row * 128 + 32 >> 7 & 7) << 4))), "r"(r1_1_bf16[0]), "r"(r1_1_bf16[1]), "r"(r1_1_bf16[2]), "r"(r1_1_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(k_row * 128 + 48 ^ (k_row * 128 + 48 >> 7 & 7) << 4))), "r"(r1_1_bf16[4]), "r"(r1_1_bf16[5]), "r"(r1_1_bf16[6]), "r"(r1_1_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(k_row * 128 + 32 ^ (k_row * 128 + 32 >> 7 & 7) << 4))), "r"(r2_2_bf16[0]), "r"(r2_2_bf16[1]), "r"(r2_2_bf16[2]), "r"(r2_2_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(k_row * 128 + 48 ^ (k_row * 128 + 48 >> 7 & 7) << 4))), "r"(r2_2_bf16[4]), "r"(r2_2_bf16[5]), "r"(r2_2_bf16[6]), "r"(r2_2_bf16[7]) : "memory");
                float hs2_3[16];
                #pragma unroll
                for (int j_7 = 0; j_7 < 16; j_7++) {
                    hs2_3[j_7] = _tmem_load_2[j_7] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 16, hs2_3);
                float _tmem_load_4[16];
                tmem_ld_x16(&_tmem_load_4[0], taddr + (unsigned int)h_row_base + 32);
                float _tmem_load_5[16];
                tmem_ld_x16(&_tmem_load_5[0], taddr + 128 + (unsigned int)h_row_base + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_8 = 0; j_8 < 16; j_8++) {
                    _tmem_load_4[j_8] = _tmem_load_4[j_8] + _tmem_load_5[j_8];
                }
                unsigned int _tmem_load_4_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 0], _tmem_load_4[_lp*2+1 + 0]));
                    _tmem_load_4_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r1_4[16];
                r1_4[0] = _tmem_load_4[0] - __uint_as_float(_tmem_load_4_bf16[0] << 16);
                r1_4[1] = _tmem_load_4[1] - __uint_as_float(_tmem_load_4_bf16[0] & mask_hi);
                r1_4[2] = _tmem_load_4[2] - __uint_as_float(_tmem_load_4_bf16[1] << 16);
                r1_4[3] = _tmem_load_4[3] - __uint_as_float(_tmem_load_4_bf16[1] & mask_hi);
                r1_4[4] = _tmem_load_4[4] - __uint_as_float(_tmem_load_4_bf16[2] << 16);
                r1_4[5] = _tmem_load_4[5] - __uint_as_float(_tmem_load_4_bf16[2] & mask_hi);
                r1_4[6] = _tmem_load_4[6] - __uint_as_float(_tmem_load_4_bf16[3] << 16);
                r1_4[7] = _tmem_load_4[7] - __uint_as_float(_tmem_load_4_bf16[3] & mask_hi);
                r1_4[8] = _tmem_load_4[8] - __uint_as_float(_tmem_load_4_bf16[4] << 16);
                r1_4[9] = _tmem_load_4[9] - __uint_as_float(_tmem_load_4_bf16[4] & mask_hi);
                r1_4[10] = _tmem_load_4[10] - __uint_as_float(_tmem_load_4_bf16[5] << 16);
                r1_4[11] = _tmem_load_4[11] - __uint_as_float(_tmem_load_4_bf16[5] & mask_hi);
                r1_4[12] = _tmem_load_4[12] - __uint_as_float(_tmem_load_4_bf16[6] << 16);
                r1_4[13] = _tmem_load_4[13] - __uint_as_float(_tmem_load_4_bf16[6] & mask_hi);
                r1_4[14] = _tmem_load_4[14] - __uint_as_float(_tmem_load_4_bf16[7] << 16);
                r1_4[15] = _tmem_load_4[15] - __uint_as_float(_tmem_load_4_bf16[7] & mask_hi);
                unsigned int r1_4_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r1_4[_lp*2 + 0], r1_4[_lp*2+1 + 0]));
                    r1_4_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r2_5[16];
                r2_5[0] = r1_4[0] - __uint_as_float(r1_4_bf16[0] << 16);
                r2_5[1] = r1_4[1] - __uint_as_float(r1_4_bf16[0] & mask_hi);
                r2_5[2] = r1_4[2] - __uint_as_float(r1_4_bf16[1] << 16);
                r2_5[3] = r1_4[3] - __uint_as_float(r1_4_bf16[1] & mask_hi);
                r2_5[4] = r1_4[4] - __uint_as_float(r1_4_bf16[2] << 16);
                r2_5[5] = r1_4[5] - __uint_as_float(r1_4_bf16[2] & mask_hi);
                r2_5[6] = r1_4[6] - __uint_as_float(r1_4_bf16[3] << 16);
                r2_5[7] = r1_4[7] - __uint_as_float(r1_4_bf16[3] & mask_hi);
                r2_5[8] = r1_4[8] - __uint_as_float(r1_4_bf16[4] << 16);
                r2_5[9] = r1_4[9] - __uint_as_float(r1_4_bf16[4] & mask_hi);
                r2_5[10] = r1_4[10] - __uint_as_float(r1_4_bf16[5] << 16);
                r2_5[11] = r1_4[11] - __uint_as_float(r1_4_bf16[5] & mask_hi);
                r2_5[12] = r1_4[12] - __uint_as_float(r1_4_bf16[6] << 16);
                r2_5[13] = r1_4[13] - __uint_as_float(r1_4_bf16[6] & mask_hi);
                r2_5[14] = r1_4[14] - __uint_as_float(r1_4_bf16[7] << 16);
                r2_5[15] = r1_4[15] - __uint_as_float(r1_4_bf16[7] & mask_hi);
                unsigned int r2_5_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r2_5[_lp*2 + 0], r2_5[_lp*2+1 + 0]));
                    r2_5_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(k_row * 128 + 64 ^ (k_row * 128 + 64 >> 7 & 7) << 4))), "r"(_tmem_load_4_bf16[0]), "r"(_tmem_load_4_bf16[1]), "r"(_tmem_load_4_bf16[2]), "r"(_tmem_load_4_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(k_row * 128 + 80 ^ (k_row * 128 + 80 >> 7 & 7) << 4))), "r"(_tmem_load_4_bf16[4]), "r"(_tmem_load_4_bf16[5]), "r"(_tmem_load_4_bf16[6]), "r"(_tmem_load_4_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(k_row * 128 + 64 ^ (k_row * 128 + 64 >> 7 & 7) << 4))), "r"(r1_4_bf16[0]), "r"(r1_4_bf16[1]), "r"(r1_4_bf16[2]), "r"(r1_4_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(k_row * 128 + 80 ^ (k_row * 128 + 80 >> 7 & 7) << 4))), "r"(r1_4_bf16[4]), "r"(r1_4_bf16[5]), "r"(r1_4_bf16[6]), "r"(r1_4_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(k_row * 128 + 64 ^ (k_row * 128 + 64 >> 7 & 7) << 4))), "r"(r2_5_bf16[0]), "r"(r2_5_bf16[1]), "r"(r2_5_bf16[2]), "r"(r2_5_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(k_row * 128 + 80 ^ (k_row * 128 + 80 >> 7 & 7) << 4))), "r"(r2_5_bf16[4]), "r"(r2_5_bf16[5]), "r"(r2_5_bf16[6]), "r"(r2_5_bf16[7]) : "memory");
                float hs2_6[16];
                #pragma unroll
                for (int j_9 = 0; j_9 < 16; j_9++) {
                    hs2_6[j_9] = _tmem_load_4[j_9] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 32, hs2_6);
                float _tmem_load_6[16];
                tmem_ld_x16(&_tmem_load_6[0], taddr + (unsigned int)h_row_base + 48);
                float _tmem_load_7[16];
                tmem_ld_x16(&_tmem_load_7[0], taddr + 128 + (unsigned int)h_row_base + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_10 = 0; j_10 < 16; j_10++) {
                    _tmem_load_6[j_10] = _tmem_load_6[j_10] + _tmem_load_7[j_10];
                }
                unsigned int _tmem_load_6_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 0], _tmem_load_6[_lp*2+1 + 0]));
                    _tmem_load_6_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r1_7[16];
                r1_7[0] = _tmem_load_6[0] - __uint_as_float(_tmem_load_6_bf16[0] << 16);
                r1_7[1] = _tmem_load_6[1] - __uint_as_float(_tmem_load_6_bf16[0] & mask_hi);
                r1_7[2] = _tmem_load_6[2] - __uint_as_float(_tmem_load_6_bf16[1] << 16);
                r1_7[3] = _tmem_load_6[3] - __uint_as_float(_tmem_load_6_bf16[1] & mask_hi);
                r1_7[4] = _tmem_load_6[4] - __uint_as_float(_tmem_load_6_bf16[2] << 16);
                r1_7[5] = _tmem_load_6[5] - __uint_as_float(_tmem_load_6_bf16[2] & mask_hi);
                r1_7[6] = _tmem_load_6[6] - __uint_as_float(_tmem_load_6_bf16[3] << 16);
                r1_7[7] = _tmem_load_6[7] - __uint_as_float(_tmem_load_6_bf16[3] & mask_hi);
                r1_7[8] = _tmem_load_6[8] - __uint_as_float(_tmem_load_6_bf16[4] << 16);
                r1_7[9] = _tmem_load_6[9] - __uint_as_float(_tmem_load_6_bf16[4] & mask_hi);
                r1_7[10] = _tmem_load_6[10] - __uint_as_float(_tmem_load_6_bf16[5] << 16);
                r1_7[11] = _tmem_load_6[11] - __uint_as_float(_tmem_load_6_bf16[5] & mask_hi);
                r1_7[12] = _tmem_load_6[12] - __uint_as_float(_tmem_load_6_bf16[6] << 16);
                r1_7[13] = _tmem_load_6[13] - __uint_as_float(_tmem_load_6_bf16[6] & mask_hi);
                r1_7[14] = _tmem_load_6[14] - __uint_as_float(_tmem_load_6_bf16[7] << 16);
                r1_7[15] = _tmem_load_6[15] - __uint_as_float(_tmem_load_6_bf16[7] & mask_hi);
                unsigned int r1_7_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r1_7[_lp*2 + 0], r1_7[_lp*2+1 + 0]));
                    r1_7_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r2_8[16];
                r2_8[0] = r1_7[0] - __uint_as_float(r1_7_bf16[0] << 16);
                r2_8[1] = r1_7[1] - __uint_as_float(r1_7_bf16[0] & mask_hi);
                r2_8[2] = r1_7[2] - __uint_as_float(r1_7_bf16[1] << 16);
                r2_8[3] = r1_7[3] - __uint_as_float(r1_7_bf16[1] & mask_hi);
                r2_8[4] = r1_7[4] - __uint_as_float(r1_7_bf16[2] << 16);
                r2_8[5] = r1_7[5] - __uint_as_float(r1_7_bf16[2] & mask_hi);
                r2_8[6] = r1_7[6] - __uint_as_float(r1_7_bf16[3] << 16);
                r2_8[7] = r1_7[7] - __uint_as_float(r1_7_bf16[3] & mask_hi);
                r2_8[8] = r1_7[8] - __uint_as_float(r1_7_bf16[4] << 16);
                r2_8[9] = r1_7[9] - __uint_as_float(r1_7_bf16[4] & mask_hi);
                r2_8[10] = r1_7[10] - __uint_as_float(r1_7_bf16[5] << 16);
                r2_8[11] = r1_7[11] - __uint_as_float(r1_7_bf16[5] & mask_hi);
                r2_8[12] = r1_7[12] - __uint_as_float(r1_7_bf16[6] << 16);
                r2_8[13] = r1_7[13] - __uint_as_float(r1_7_bf16[6] & mask_hi);
                r2_8[14] = r1_7[14] - __uint_as_float(r1_7_bf16[7] << 16);
                r2_8[15] = r1_7[15] - __uint_as_float(r1_7_bf16[7] & mask_hi);
                unsigned int r2_8_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r2_8[_lp*2 + 0], r2_8[_lp*2+1 + 0]));
                    r2_8_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(k_row * 128 + 96 ^ (k_row * 128 + 96 >> 7 & 7) << 4))), "r"(_tmem_load_6_bf16[0]), "r"(_tmem_load_6_bf16[1]), "r"(_tmem_load_6_bf16[2]), "r"(_tmem_load_6_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(k_row * 128 + 112 ^ (k_row * 128 + 112 >> 7 & 7) << 4))), "r"(_tmem_load_6_bf16[4]), "r"(_tmem_load_6_bf16[5]), "r"(_tmem_load_6_bf16[6]), "r"(_tmem_load_6_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(k_row * 128 + 96 ^ (k_row * 128 + 96 >> 7 & 7) << 4))), "r"(r1_7_bf16[0]), "r"(r1_7_bf16[1]), "r"(r1_7_bf16[2]), "r"(r1_7_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(k_row * 128 + 112 ^ (k_row * 128 + 112 >> 7 & 7) << 4))), "r"(r1_7_bf16[4]), "r"(r1_7_bf16[5]), "r"(r1_7_bf16[6]), "r"(r1_7_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(k_row * 128 + 96 ^ (k_row * 128 + 96 >> 7 & 7) << 4))), "r"(r2_8_bf16[0]), "r"(r2_8_bf16[1]), "r"(r2_8_bf16[2]), "r"(r2_8_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(k_row * 128 + 112 ^ (k_row * 128 + 112 >> 7 & 7) << 4))), "r"(r2_8_bf16[4]), "r"(r2_8_bf16[5]), "r"(r2_8_bf16[6]), "r"(r2_8_bf16[7]) : "memory");
                float hs2_9[16];
                #pragma unroll
                for (int j_11 = 0; j_11 < 16; j_11++) {
                    hs2_9[j_11] = _tmem_load_6[j_11] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 48, hs2_9);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(hs_ready_addr);
                mbarrier_wait(vp_ready_addr, _phase_vp_ready_0);
                _phase_vp_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int trow = row0 + c * 64 + t_row;
                long long u_base = ((long long)trow * (long long)num_heads + (long long)head) * 128 + (long long)v0;
                float _tmem_load_8[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[15]))
                    : "r"(taddr + 64 + (unsigned int)vp_row_base));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col = lane_half * 16;
                float uv[16];
                uv[0] = 0.0f;
                uv[1] = 0.0f;
                uv[2] = 0.0f;
                uv[3] = 0.0f;
                uv[4] = 0.0f;
                uv[5] = 0.0f;
                uv[6] = 0.0f;
                uv[7] = 0.0f;
                uv[8] = 0.0f;
                uv[9] = 0.0f;
                uv[10] = 0.0f;
                uv[11] = 0.0f;
                uv[12] = 0.0f;
                uv[13] = 0.0f;
                uv[14] = 0.0f;
                uv[15] = 0.0f;
                if (use_u == 1) {
                    {
                        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(u + u_base + (long long)col);
                        uint4 _vld_0[2];
                        #pragma unroll
                        for (int _blk = 0; _blk < 2; _blk++) {
                            _vld_0[_blk] = _vptr_0[_blk];
                            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&uv[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                                (&uv[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                }
                float bvv[16];
                #pragma unroll
                for (int j_12 = 0; j_12 < 16; j_12++) {
                    bvv[j_12] = uv[j_12] - _tmem_load_8[j_12];
                }
                unsigned int bvv_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv[_lp*2 + 0], bvv[_lp*2+1 + 0]));
                    bvv_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r1_10[16];
                r1_10[0] = bvv[0] - __uint_as_float(bvv_bf16[0] << 16);
                r1_10[1] = bvv[1] - __uint_as_float(bvv_bf16[0] & mask_hi);
                r1_10[2] = bvv[2] - __uint_as_float(bvv_bf16[1] << 16);
                r1_10[3] = bvv[3] - __uint_as_float(bvv_bf16[1] & mask_hi);
                r1_10[4] = bvv[4] - __uint_as_float(bvv_bf16[2] << 16);
                r1_10[5] = bvv[5] - __uint_as_float(bvv_bf16[2] & mask_hi);
                r1_10[6] = bvv[6] - __uint_as_float(bvv_bf16[3] << 16);
                r1_10[7] = bvv[7] - __uint_as_float(bvv_bf16[3] & mask_hi);
                r1_10[8] = bvv[8] - __uint_as_float(bvv_bf16[4] << 16);
                r1_10[9] = bvv[9] - __uint_as_float(bvv_bf16[4] & mask_hi);
                r1_10[10] = bvv[10] - __uint_as_float(bvv_bf16[5] << 16);
                r1_10[11] = bvv[11] - __uint_as_float(bvv_bf16[5] & mask_hi);
                r1_10[12] = bvv[12] - __uint_as_float(bvv_bf16[6] << 16);
                r1_10[13] = bvv[13] - __uint_as_float(bvv_bf16[6] & mask_hi);
                r1_10[14] = bvv[14] - __uint_as_float(bvv_bf16[7] << 16);
                r1_10[15] = bvv[15] - __uint_as_float(bvv_bf16[7] & mask_hi);
                unsigned int r1_10_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r1_10[_lp*2 + 0], r1_10[_lp*2+1 + 0]));
                    r1_10_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r2_11[16];
                r2_11[0] = r1_10[0] - __uint_as_float(r1_10_bf16[0] << 16);
                r2_11[1] = r1_10[1] - __uint_as_float(r1_10_bf16[0] & mask_hi);
                r2_11[2] = r1_10[2] - __uint_as_float(r1_10_bf16[1] << 16);
                r2_11[3] = r1_10[3] - __uint_as_float(r1_10_bf16[1] & mask_hi);
                r2_11[4] = r1_10[4] - __uint_as_float(r1_10_bf16[2] << 16);
                r2_11[5] = r1_10[5] - __uint_as_float(r1_10_bf16[2] & mask_hi);
                r2_11[6] = r1_10[6] - __uint_as_float(r1_10_bf16[3] << 16);
                r2_11[7] = r1_10[7] - __uint_as_float(r1_10_bf16[3] & mask_hi);
                r2_11[8] = r1_10[8] - __uint_as_float(r1_10_bf16[4] << 16);
                r2_11[9] = r1_10[9] - __uint_as_float(r1_10_bf16[4] & mask_hi);
                r2_11[10] = r1_10[10] - __uint_as_float(r1_10_bf16[5] << 16);
                r2_11[11] = r1_10[11] - __uint_as_float(r1_10_bf16[5] & mask_hi);
                r2_11[12] = r1_10[12] - __uint_as_float(r1_10_bf16[6] << 16);
                r2_11[13] = r1_10[13] - __uint_as_float(r1_10_bf16[6] & mask_hi);
                r2_11[14] = r1_10[14] - __uint_as_float(r1_10_bf16[7] << 16);
                r2_11[15] = r1_10[15] - __uint_as_float(r1_10_bf16[7] & mask_hi);
                unsigned int r2_11_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r2_11[_lp*2 + 0], r2_11[_lp*2+1 + 0]));
                    r2_11_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv0_addr + (unsigned int)(col / 64 * 8192 + t_row * 128 + col % 64 * 2 ^ (col / 64 * 8192 + t_row * 128 + col % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_bf16[0]), "r"(bvv_bf16[1]), "r"(bvv_bf16[2]), "r"(bvv_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv0_addr + (unsigned int)((col + 8) / 64 * 8192 + t_row * 128 + (col + 8) % 64 * 2 ^ ((col + 8) / 64 * 8192 + t_row * 128 + (col + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_bf16[4]), "r"(bvv_bf16[5]), "r"(bvv_bf16[6]), "r"(bvv_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv1_addr + (unsigned int)(col / 64 * 8192 + t_row * 128 + col % 64 * 2 ^ (col / 64 * 8192 + t_row * 128 + col % 64 * 2 >> 7 & 7) << 4))), "r"(r1_10_bf16[0]), "r"(r1_10_bf16[1]), "r"(r1_10_bf16[2]), "r"(r1_10_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv1_addr + (unsigned int)((col + 8) / 64 * 8192 + t_row * 128 + (col + 8) % 64 * 2 ^ ((col + 8) / 64 * 8192 + t_row * 128 + (col + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(r1_10_bf16[4]), "r"(r1_10_bf16[5]), "r"(r1_10_bf16[6]), "r"(r1_10_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv2_addr + (unsigned int)(col / 64 * 8192 + t_row * 128 + col % 64 * 2 ^ (col / 64 * 8192 + t_row * 128 + col % 64 * 2 >> 7 & 7) << 4))), "r"(r2_11_bf16[0]), "r"(r2_11_bf16[1]), "r"(r2_11_bf16[2]), "r"(r2_11_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv2_addr + (unsigned int)((col + 8) / 64 * 8192 + t_row * 128 + (col + 8) % 64 * 2 ^ ((col + 8) / 64 * 8192 + t_row * 128 + (col + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(r2_11_bf16[4]), "r"(r2_11_bf16[5]), "r"(r2_11_bf16[6]), "r"(r2_11_bf16[7]) : "memory");
                float _tmem_load_9[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[15]))
                    : "r"(taddr + 64 + (unsigned int)vp_row_base + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col_12 = 32 + lane_half * 16;
                float uv_13[16];
                uv_13[0] = 0.0f;
                uv_13[1] = 0.0f;
                uv_13[2] = 0.0f;
                uv_13[3] = 0.0f;
                uv_13[4] = 0.0f;
                uv_13[5] = 0.0f;
                uv_13[6] = 0.0f;
                uv_13[7] = 0.0f;
                uv_13[8] = 0.0f;
                uv_13[9] = 0.0f;
                uv_13[10] = 0.0f;
                uv_13[11] = 0.0f;
                uv_13[12] = 0.0f;
                uv_13[13] = 0.0f;
                uv_13[14] = 0.0f;
                uv_13[15] = 0.0f;
                if (use_u == 1) {
                    {
                        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(u + u_base + (long long)col_12);
                        uint4 _vld_1[2];
                        #pragma unroll
                        for (int _blk = 0; _blk < 2; _blk++) {
                            _vld_1[_blk] = _vptr_1[_blk];
                            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&uv_13[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                                (&uv_13[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                }
                float bvv_14[16];
                #pragma unroll
                for (int j_13 = 0; j_13 < 16; j_13++) {
                    bvv_14[j_13] = uv_13[j_13] - _tmem_load_9[j_13];
                }
                unsigned int bvv_14_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_14[_lp*2 + 0], bvv_14[_lp*2+1 + 0]));
                    bvv_14_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r1_15[16];
                r1_15[0] = bvv_14[0] - __uint_as_float(bvv_14_bf16[0] << 16);
                r1_15[1] = bvv_14[1] - __uint_as_float(bvv_14_bf16[0] & mask_hi);
                r1_15[2] = bvv_14[2] - __uint_as_float(bvv_14_bf16[1] << 16);
                r1_15[3] = bvv_14[3] - __uint_as_float(bvv_14_bf16[1] & mask_hi);
                r1_15[4] = bvv_14[4] - __uint_as_float(bvv_14_bf16[2] << 16);
                r1_15[5] = bvv_14[5] - __uint_as_float(bvv_14_bf16[2] & mask_hi);
                r1_15[6] = bvv_14[6] - __uint_as_float(bvv_14_bf16[3] << 16);
                r1_15[7] = bvv_14[7] - __uint_as_float(bvv_14_bf16[3] & mask_hi);
                r1_15[8] = bvv_14[8] - __uint_as_float(bvv_14_bf16[4] << 16);
                r1_15[9] = bvv_14[9] - __uint_as_float(bvv_14_bf16[4] & mask_hi);
                r1_15[10] = bvv_14[10] - __uint_as_float(bvv_14_bf16[5] << 16);
                r1_15[11] = bvv_14[11] - __uint_as_float(bvv_14_bf16[5] & mask_hi);
                r1_15[12] = bvv_14[12] - __uint_as_float(bvv_14_bf16[6] << 16);
                r1_15[13] = bvv_14[13] - __uint_as_float(bvv_14_bf16[6] & mask_hi);
                r1_15[14] = bvv_14[14] - __uint_as_float(bvv_14_bf16[7] << 16);
                r1_15[15] = bvv_14[15] - __uint_as_float(bvv_14_bf16[7] & mask_hi);
                unsigned int r1_15_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r1_15[_lp*2 + 0], r1_15[_lp*2+1 + 0]));
                    r1_15_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r2_16[16];
                r2_16[0] = r1_15[0] - __uint_as_float(r1_15_bf16[0] << 16);
                r2_16[1] = r1_15[1] - __uint_as_float(r1_15_bf16[0] & mask_hi);
                r2_16[2] = r1_15[2] - __uint_as_float(r1_15_bf16[1] << 16);
                r2_16[3] = r1_15[3] - __uint_as_float(r1_15_bf16[1] & mask_hi);
                r2_16[4] = r1_15[4] - __uint_as_float(r1_15_bf16[2] << 16);
                r2_16[5] = r1_15[5] - __uint_as_float(r1_15_bf16[2] & mask_hi);
                r2_16[6] = r1_15[6] - __uint_as_float(r1_15_bf16[3] << 16);
                r2_16[7] = r1_15[7] - __uint_as_float(r1_15_bf16[3] & mask_hi);
                r2_16[8] = r1_15[8] - __uint_as_float(r1_15_bf16[4] << 16);
                r2_16[9] = r1_15[9] - __uint_as_float(r1_15_bf16[4] & mask_hi);
                r2_16[10] = r1_15[10] - __uint_as_float(r1_15_bf16[5] << 16);
                r2_16[11] = r1_15[11] - __uint_as_float(r1_15_bf16[5] & mask_hi);
                r2_16[12] = r1_15[12] - __uint_as_float(r1_15_bf16[6] << 16);
                r2_16[13] = r1_15[13] - __uint_as_float(r1_15_bf16[6] & mask_hi);
                r2_16[14] = r1_15[14] - __uint_as_float(r1_15_bf16[7] << 16);
                r2_16[15] = r1_15[15] - __uint_as_float(r1_15_bf16[7] & mask_hi);
                unsigned int r2_16_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r2_16[_lp*2 + 0], r2_16[_lp*2+1 + 0]));
                    r2_16_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv0_addr + (unsigned int)(col_12 / 64 * 8192 + t_row * 128 + col_12 % 64 * 2 ^ (col_12 / 64 * 8192 + t_row * 128 + col_12 % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_14_bf16[0]), "r"(bvv_14_bf16[1]), "r"(bvv_14_bf16[2]), "r"(bvv_14_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv0_addr + (unsigned int)((col_12 + 8) / 64 * 8192 + t_row * 128 + (col_12 + 8) % 64 * 2 ^ ((col_12 + 8) / 64 * 8192 + t_row * 128 + (col_12 + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_14_bf16[4]), "r"(bvv_14_bf16[5]), "r"(bvv_14_bf16[6]), "r"(bvv_14_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv1_addr + (unsigned int)(col_12 / 64 * 8192 + t_row * 128 + col_12 % 64 * 2 ^ (col_12 / 64 * 8192 + t_row * 128 + col_12 % 64 * 2 >> 7 & 7) << 4))), "r"(r1_15_bf16[0]), "r"(r1_15_bf16[1]), "r"(r1_15_bf16[2]), "r"(r1_15_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv1_addr + (unsigned int)((col_12 + 8) / 64 * 8192 + t_row * 128 + (col_12 + 8) % 64 * 2 ^ ((col_12 + 8) / 64 * 8192 + t_row * 128 + (col_12 + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(r1_15_bf16[4]), "r"(r1_15_bf16[5]), "r"(r1_15_bf16[6]), "r"(r1_15_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv2_addr + (unsigned int)(col_12 / 64 * 8192 + t_row * 128 + col_12 % 64 * 2 ^ (col_12 / 64 * 8192 + t_row * 128 + col_12 % 64 * 2 >> 7 & 7) << 4))), "r"(r2_16_bf16[0]), "r"(r2_16_bf16[1]), "r"(r2_16_bf16[2]), "r"(r2_16_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv2_addr + (unsigned int)((col_12 + 8) / 64 * 8192 + t_row * 128 + (col_12 + 8) % 64 * 2 ^ ((col_12 + 8) / 64 * 8192 + t_row * 128 + (col_12 + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(r2_16_bf16[4]), "r"(r2_16_bf16[5]), "r"(r2_16_bf16[6]), "r"(r2_16_bf16[7]) : "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(bv_ready_addr);
                mbarrier_wait(h_ready_addr, _phase_h_ready_0);
                _phase_h_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
            }
            int slot = out_index[seg];
            if (slot >= 0) {
                long long tile = (long long)((head / out_hg * out_n + slot) * out_hg + head % out_hg) * (long long)tile_stride * 16384;
                float _tmem_load_10[16];
                tmem_ld_x16(&_tmem_load_10[0], taddr + (unsigned int)h_row_base);
                float _tmem_load_11[16];
                tmem_ld_x16(&_tmem_load_11[0], taddr + 128 + (unsigned int)h_row_base);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_14 = 0; j_14 < 16; j_14++) {
                    _tmem_load_10[j_14] = _tmem_load_10[j_14] + _tmem_load_11[j_14];
                }
                if (write_transposed == 1) {
                    long long row_at = tile + (long long)(k_row * 128 + v0);
                    {
                        float4 _v4 = make_float4(_tmem_load_10[0 + 0], _tmem_load_10[0 + 1], _tmem_load_10[0 + 2], _tmem_load_10[0 + 3]);
                        *reinterpret_cast<float4*>(out + row_at) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_10[4 + 0], _tmem_load_10[4 + 1], _tmem_load_10[4 + 2], _tmem_load_10[4 + 3]);
                        *reinterpret_cast<float4*>(out + row_at + 4) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_10[8 + 0], _tmem_load_10[8 + 1], _tmem_load_10[8 + 2], _tmem_load_10[8 + 3]);
                        *reinterpret_cast<float4*>(out + row_at + 8) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_10[12 + 0], _tmem_load_10[12 + 1], _tmem_load_10[12 + 2], _tmem_load_10[12 + 3]);
                        *reinterpret_cast<float4*>(out + row_at + 12) = _v4;
                    }
                } else {
                    #pragma unroll
                    for (int j_15 = 0; j_15 < 16; j_15++) {
                        *(reinterpret_cast<float*>(out) + (tile + (long long)((v0 + j_15) * 128 + k_row))) = _tmem_load_10[j_15];
                    }
                }
                float _tmem_load_12[16];
                tmem_ld_x16(&_tmem_load_12[0], taddr + (unsigned int)h_row_base + 16);
                float _tmem_load_13[16];
                tmem_ld_x16(&_tmem_load_13[0], taddr + 128 + (unsigned int)h_row_base + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_16 = 0; j_16 < 16; j_16++) {
                    _tmem_load_12[j_16] = _tmem_load_12[j_16] + _tmem_load_13[j_16];
                }
                if (write_transposed == 1) {
                    long long row_at_1 = tile + (long long)(k_row * 128 + v0 + 16);
                    {
                        float4 _v4 = make_float4(_tmem_load_12[0 + 0], _tmem_load_12[0 + 1], _tmem_load_12[0 + 2], _tmem_load_12[0 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_12[4 + 0], _tmem_load_12[4 + 1], _tmem_load_12[4 + 2], _tmem_load_12[4 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1 + 4) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_12[8 + 0], _tmem_load_12[8 + 1], _tmem_load_12[8 + 2], _tmem_load_12[8 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1 + 8) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_12[12 + 0], _tmem_load_12[12 + 1], _tmem_load_12[12 + 2], _tmem_load_12[12 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1 + 12) = _v4;
                    }
                } else {
                    #pragma unroll
                    for (int j_17 = 0; j_17 < 16; j_17++) {
                        *(reinterpret_cast<float*>(out) + (tile + (long long)((v0 + 16 + j_17) * 128 + k_row))) = _tmem_load_12[j_17];
                    }
                }
                float _tmem_load_14[16];
                tmem_ld_x16(&_tmem_load_14[0], taddr + (unsigned int)h_row_base + 32);
                float _tmem_load_15[16];
                tmem_ld_x16(&_tmem_load_15[0], taddr + 128 + (unsigned int)h_row_base + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_18 = 0; j_18 < 16; j_18++) {
                    _tmem_load_14[j_18] = _tmem_load_14[j_18] + _tmem_load_15[j_18];
                }
                if (write_transposed == 1) {
                    long long row_at_2 = tile + (long long)(k_row * 128 + v0 + 32);
                    {
                        float4 _v4 = make_float4(_tmem_load_14[0 + 0], _tmem_load_14[0 + 1], _tmem_load_14[0 + 2], _tmem_load_14[0 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_2) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_14[4 + 0], _tmem_load_14[4 + 1], _tmem_load_14[4 + 2], _tmem_load_14[4 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_2 + 4) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_14[8 + 0], _tmem_load_14[8 + 1], _tmem_load_14[8 + 2], _tmem_load_14[8 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_2 + 8) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_14[12 + 0], _tmem_load_14[12 + 1], _tmem_load_14[12 + 2], _tmem_load_14[12 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_2 + 12) = _v4;
                    }
                } else {
                    #pragma unroll
                    for (int j_19 = 0; j_19 < 16; j_19++) {
                        *(reinterpret_cast<float*>(out) + (tile + (long long)((v0 + 32 + j_19) * 128 + k_row))) = _tmem_load_14[j_19];
                    }
                }
                float _tmem_load_16[16];
                tmem_ld_x16(&_tmem_load_16[0], taddr + (unsigned int)h_row_base + 48);
                float _tmem_load_17[16];
                tmem_ld_x16(&_tmem_load_17[0], taddr + 128 + (unsigned int)h_row_base + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_20 = 0; j_20 < 16; j_20++) {
                    _tmem_load_16[j_20] = _tmem_load_16[j_20] + _tmem_load_17[j_20];
                }
                if (write_transposed == 1) {
                    long long row_at_3 = tile + (long long)(k_row * 128 + v0 + 48);
                    {
                        float4 _v4 = make_float4(_tmem_load_16[0 + 0], _tmem_load_16[0 + 1], _tmem_load_16[0 + 2], _tmem_load_16[0 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_3) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_16[4 + 0], _tmem_load_16[4 + 1], _tmem_load_16[4 + 2], _tmem_load_16[4 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_3 + 4) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_16[8 + 0], _tmem_load_16[8 + 1], _tmem_load_16[8 + 2], _tmem_load_16[8 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_3 + 8) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_16[12 + 0], _tmem_load_16[12 + 1], _tmem_load_16[12 + 2], _tmem_load_16[12 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_3 + 12) = _v4;
                    }
                } else {
                    #pragma unroll
                    for (int j_21 = 0; j_21 < 16; j_21++) {
                        *(reinterpret_cast<float*>(out) + (tile + (long long)((v0 + 48 + j_21) * 128 + k_row))) = _tmem_load_16[j_21];
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
            int vslice_1 = blockIdx.x % 2;
            int head_1 = blockIdx.x / 2 % num_heads;
            int seg_1 = walk_seg[blockIdx.x / (2 * num_heads)];
            int c0_1 = seq_chunk_start[seg_1];
            int nc_1 = seq_chunk_start[seg_1 + 1] - c0_1;
            int row0_1 = c0_1 * 64;
            unsigned int _phase_tiles_empty = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int c_1 = 0; c_1 < nc_1; c_1++) {
                    mbarrier_wait(tiles_empty_addr, _phase_tiles_empty);
                    mbarrier_arrive_expect_tx(tiles_full_addr, 32768);
                    int trow_1 = row0_1 + c_1 * 64;
                    #pragma unroll
                    for (int half = 0; half < 2; half++) {
                        tma_3d_gmem2smem(w_panel_addr + (0 * 2 + (unsigned int)half) * 8192, (&w_tma), half * 64, head_1, trow_1, tiles_full_addr);
                        tma_3d_gmem2smem(kg_panel_addr + (0 * 2 + (unsigned int)half) * 8192, (&kg_tma), half * 64, head_1, trow_1, tiles_full_addr);
                    }
                    _phase_tiles_empty ^= 1;
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 5) {
        { // mma_main
            int vslice_2 = blockIdx.x % 2;
            int head_2 = blockIdx.x / 2 % num_heads;
            int seg_2 = walk_seg[blockIdx.x / (2 * num_heads)];
            int c0_2 = seq_chunk_start[seg_2];
            int nc_2 = seq_chunk_start[seg_2 + 1] - c0_2;
            int row0_2 = c0_2 * 64;
            unsigned int _phase_tiles_full = 0;
            unsigned int _phase_hs_ready_0 = 0;
            unsigned int _phase_bv_ready_0 = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int c_2 = 0; c_2 < nc_2; c_2++) {
                    mbarrier_wait(tiles_full_addr, _phase_tiles_full);
                    mbarrier_wait(hs_ready_addr, _phase_hs_ready_0);
                    _phase_hs_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_0 = (((w_op_addr) >> 4) & 0x7FFF) + (0) * 1024;
                    int _smem_desc_lo_1 = (((hs_mn2_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
                    int _smem_desc_lo_0 = (((w_op_addr) >> 4) & 0x3FFF) + (0) * 1024;
                    int _smem_desc_lo_1 = (((hs_mn2_addr) >> 4) & 0x3FFF) | 0x4000000;
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
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_2 = (((w_op_addr) >> 4) & 0x7FFF) + (0) * 1024;
                    int _smem_desc_lo_3 = (((hs_mn1_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
                    int _smem_desc_lo_2 = (((w_op_addr) >> 4) & 0x3FFF) + (0) * 1024;
                    int _smem_desc_lo_3 = (((hs_mn1_addr) >> 4) & 0x3FFF) | 0x4000000;
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
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"(tmem_acc_vp), "r"(1) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_4 = (((w_op_addr) >> 4) & 0x7FFF) + (0) * 1024;
                    int _smem_desc_lo_5 = (((hs_mn0_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
                    int _smem_desc_lo_4 = (((w_op_addr) >> 4) & 0x3FFF) + (0) * 1024;
                    int _smem_desc_lo_5 = (((hs_mn0_addr) >> 4) & 0x3FFF) | 0x4000000;
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
                    :: "r"(_smem_desc_lo_4), "r"(_smem_desc_lo_5), "r"(tmem_acc_vp), "r"(1) : "memory");
                    tcgen05_commit(vp_ready_addr);
                    mbarrier_wait(bv_ready_addr, _phase_bv_ready_0);
                    _phase_bv_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_6 = ((((kg_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (0) * 1024;
                    int _smem_desc_lo_7 = (((bv_mn2_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                    int _smem_desc_lo_6 = ((((kg_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (0) * 1024;
                    int _smem_desc_lo_7 = (((bv_mn2_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    :: "r"(_smem_desc_lo_6), "r"(_smem_desc_lo_7), "r"(tmem_acc_dx), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_8 = ((((kg_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (0) * 1024;
                    int _smem_desc_lo_9 = (((bv_mn1_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                    int _smem_desc_lo_8 = ((((kg_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (0) * 1024;
                    int _smem_desc_lo_9 = (((bv_mn1_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    :: "r"(_smem_desc_lo_8), "r"(_smem_desc_lo_9), "r"(tmem_acc_dx), "r"(1) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_10 = ((((kg_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (0) * 1024;
                    int _smem_desc_lo_11 = (((bv_mn0_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                    int _smem_desc_lo_10 = ((((kg_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (0) * 1024;
                    int _smem_desc_lo_11 = (((bv_mn0_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    :: "r"(_smem_desc_lo_10), "r"(_smem_desc_lo_11), "r"(tmem_acc_dx), "r"(1) : "memory");
                    tcgen05_commit(h_ready_addr);
                    tcgen05_commit(tiles_empty_addr);
                    _phase_tiles_full ^= 1;
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
