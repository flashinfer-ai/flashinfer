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

__global__ __launch_bounds__(320, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_train_8ffd931ca0595948b608_kcp(const __grid_constant__ CUtensorMap w_tma, const __grid_constant__ CUtensorMap kg_tma, __nv_bfloat16* __restrict__ u, float* __restrict__ gk, int num_heads, int* __restrict__ seq_chunk_start, int num_seqs, int* __restrict__ walk_seg, float* __restrict__ out, int* __restrict__ out_index, int out_hg, int out_n, int tile_stride, int init_mode, int use_u, int write_transposed)
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
    #define kg_full_addr (mbar_base + 16)
    #define kg_empty_addr (mbar_base + 24)
    #define hs_ready_addr (mbar_base + 32)
    #define vp_ready_addr (mbar_base + 40)
    #define bv_ready_addr (mbar_base + 48)
    #define h_ready_addr (mbar_base + 56)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
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

    // Mbarrier init (8 barriers at smem_raw[0..64))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ring_pipe' ---
            // w_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // w_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // kg_full: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            // kg_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            // --- pipeline 'one' ---
            // hs_ready: 1 barriers, init_count=256
            mbarrier_init(smem + 32, 256);
            // vp_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            // bv_ready: 1 barriers, init_count=256
            mbarrier_init(smem + 48, 256);
            // h_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 192 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 64);
    if (warp == 0) {
        int _tmem_hold = smem + 64;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_acc_h = taddr;
    const unsigned int tmem_acc_vp = taddr + 64;
    const unsigned int tmem_acc_dx = taddr + 128;

    // ---- Role: compute ----
    if (warp <= 7) {
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
            unsigned int warp_group_idx = ((unsigned int)tid - 0) / 128;
            int cb = (int)warp_group_idx * 32;
            unsigned int mask_hi = 4294901760u;
            float init_row[16];
            #pragma unroll
            for (int j_1_0 = 0; j_1_0 < 16; j_1_0++) {
                float one_here_1_0 = 0.0f;
                if (init_mode == 0) {
                    if (v0 + cb + j_1_0 == k_row) {
                        one_here_1_0 = 1.0f;
                    }
                }
                init_row[j_1_0] = one_here_1_0;
            }
            tmem_st_x16_f32(taddr + (unsigned int)(h_row_base + cb), init_row);
            #pragma unroll
            for (int j_1_1 = 0; j_1_1 < 16; j_1_1++) {
                float one_here_1_1 = 0.0f;
                if (init_mode == 0) {
                    if (v0 + cb + 16 + j_1_1 == k_row) {
                        one_here_1_1 = 1.0f;
                    }
                }
                init_row[j_1_1] = one_here_1_1;
            }
            tmem_st_x16_f32(taddr + (unsigned int)(h_row_base + cb) + 16, init_row);
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
            tmem_st_x16_f32(taddr + 128 + (unsigned int)(h_row_base + cb), zeros);
            tmem_st_x16_f32(taddr + 128 + (unsigned int)(h_row_base + cb) + 16, zeros);
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            float h_a[16];
            float h_b[16];
            float d_a[16];
            float d_b[16];
            float vp_a[16];
            float vp_b[16];
            float gk_cur = 0.0f;
            if (nc > 0) {
                gk_cur = gk[((long long)(row0 + 64 - 1) * (long long)num_heads + (long long)head) * 128ll + (long long)k_row];
            }
            unsigned int _phase_vp_ready_0 = 0;
            unsigned int _phase_h_ready_0 = 0;
            #pragma unroll 1
            for (int c = 0; c < nc; c++) {
                float decay = approx_exp2(gk_cur);
                if (nc > c + 1) {
                    gk_cur = gk[((long long)(row0 + (c + 1) * 64 + 64 - 1) * (long long)num_heads + (long long)head) * 128ll + (long long)k_row];
                }
                tmem_ld_x16(&h_a[0], taddr + (unsigned int)(h_row_base + cb));
                tmem_ld_x16(&d_a[0], taddr + 128 + (unsigned int)(h_row_base + cb));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                tmem_ld_x16(&h_b[0], taddr + (unsigned int)(h_row_base + cb) + 16);
                tmem_ld_x16(&d_b[0], taddr + 128 + (unsigned int)(h_row_base + cb) + 16);
                #pragma unroll
                for (int j = 0; j < 16; j++) {
                    h_a[j] = h_a[j] + d_a[j];
                }
                unsigned int tmem_load_0_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_a[_lp*2 + 0], h_a[_lp*2+1 + 0]));
                    tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r1[16];
                r1[0] = h_a[0] - __uint_as_float(tmem_load_0_bf16[0] << 16);
                r1[1] = h_a[1] - __uint_as_float(tmem_load_0_bf16[0] & mask_hi);
                r1[2] = h_a[2] - __uint_as_float(tmem_load_0_bf16[1] << 16);
                r1[3] = h_a[3] - __uint_as_float(tmem_load_0_bf16[1] & mask_hi);
                r1[4] = h_a[4] - __uint_as_float(tmem_load_0_bf16[2] << 16);
                r1[5] = h_a[5] - __uint_as_float(tmem_load_0_bf16[2] & mask_hi);
                r1[6] = h_a[6] - __uint_as_float(tmem_load_0_bf16[3] << 16);
                r1[7] = h_a[7] - __uint_as_float(tmem_load_0_bf16[3] & mask_hi);
                r1[8] = h_a[8] - __uint_as_float(tmem_load_0_bf16[4] << 16);
                r1[9] = h_a[9] - __uint_as_float(tmem_load_0_bf16[4] & mask_hi);
                r1[10] = h_a[10] - __uint_as_float(tmem_load_0_bf16[5] << 16);
                r1[11] = h_a[11] - __uint_as_float(tmem_load_0_bf16[5] & mask_hi);
                r1[12] = h_a[12] - __uint_as_float(tmem_load_0_bf16[6] << 16);
                r1[13] = h_a[13] - __uint_as_float(tmem_load_0_bf16[6] & mask_hi);
                r1[14] = h_a[14] - __uint_as_float(tmem_load_0_bf16[7] << 16);
                r1[15] = h_a[15] - __uint_as_float(tmem_load_0_bf16[7] & mask_hi);
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
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(cb / 64 * 16384 + k_row * 128 + cb % 64 * 2 ^ (cb / 64 * 16384 + k_row * 128 + cb % 64 * 2 >> 7 & 7) << 4))), "r"(tmem_load_0_bf16[0]), "r"(tmem_load_0_bf16[1]), "r"(tmem_load_0_bf16[2]), "r"(tmem_load_0_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)((cb + 8) / 64 * 16384 + k_row * 128 + (cb + 8) % 64 * 2 ^ ((cb + 8) / 64 * 16384 + k_row * 128 + (cb + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(tmem_load_0_bf16[4]), "r"(tmem_load_0_bf16[5]), "r"(tmem_load_0_bf16[6]), "r"(tmem_load_0_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(cb / 64 * 16384 + k_row * 128 + cb % 64 * 2 ^ (cb / 64 * 16384 + k_row * 128 + cb % 64 * 2 >> 7 & 7) << 4))), "r"(r1_bf16[0]), "r"(r1_bf16[1]), "r"(r1_bf16[2]), "r"(r1_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)((cb + 8) / 64 * 16384 + k_row * 128 + (cb + 8) % 64 * 2 ^ ((cb + 8) / 64 * 16384 + k_row * 128 + (cb + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(r1_bf16[4]), "r"(r1_bf16[5]), "r"(r1_bf16[6]), "r"(r1_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(cb / 64 * 16384 + k_row * 128 + cb % 64 * 2 ^ (cb / 64 * 16384 + k_row * 128 + cb % 64 * 2 >> 7 & 7) << 4))), "r"(r2_bf16[0]), "r"(r2_bf16[1]), "r"(r2_bf16[2]), "r"(r2_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)((cb + 8) / 64 * 16384 + k_row * 128 + (cb + 8) % 64 * 2 ^ ((cb + 8) / 64 * 16384 + k_row * 128 + (cb + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(r2_bf16[4]), "r"(r2_bf16[5]), "r"(r2_bf16[6]), "r"(r2_bf16[7]) : "memory");
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                #pragma unroll
                for (int j_1 = 0; j_1 < 16; j_1++) {
                    h_b[j_1] = h_b[j_1] + d_b[j_1];
                }
                unsigned int tmem_load_0_bf16_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_b[_lp*2 + 0], h_b[_lp*2+1 + 0]));
                    tmem_load_0_bf16_0[_lp] = *(uint32_t*)&_bf2;
                }
                float r1_0[16];
                r1_0[0] = h_b[0] - __uint_as_float(tmem_load_0_bf16_0[0] << 16);
                r1_0[1] = h_b[1] - __uint_as_float(tmem_load_0_bf16_0[0] & mask_hi);
                r1_0[2] = h_b[2] - __uint_as_float(tmem_load_0_bf16_0[1] << 16);
                r1_0[3] = h_b[3] - __uint_as_float(tmem_load_0_bf16_0[1] & mask_hi);
                r1_0[4] = h_b[4] - __uint_as_float(tmem_load_0_bf16_0[2] << 16);
                r1_0[5] = h_b[5] - __uint_as_float(tmem_load_0_bf16_0[2] & mask_hi);
                r1_0[6] = h_b[6] - __uint_as_float(tmem_load_0_bf16_0[3] << 16);
                r1_0[7] = h_b[7] - __uint_as_float(tmem_load_0_bf16_0[3] & mask_hi);
                r1_0[8] = h_b[8] - __uint_as_float(tmem_load_0_bf16_0[4] << 16);
                r1_0[9] = h_b[9] - __uint_as_float(tmem_load_0_bf16_0[4] & mask_hi);
                r1_0[10] = h_b[10] - __uint_as_float(tmem_load_0_bf16_0[5] << 16);
                r1_0[11] = h_b[11] - __uint_as_float(tmem_load_0_bf16_0[5] & mask_hi);
                r1_0[12] = h_b[12] - __uint_as_float(tmem_load_0_bf16_0[6] << 16);
                r1_0[13] = h_b[13] - __uint_as_float(tmem_load_0_bf16_0[6] & mask_hi);
                r1_0[14] = h_b[14] - __uint_as_float(tmem_load_0_bf16_0[7] << 16);
                r1_0[15] = h_b[15] - __uint_as_float(tmem_load_0_bf16_0[7] & mask_hi);
                unsigned int r1_bf16_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r1_0[_lp*2 + 0], r1_0[_lp*2+1 + 0]));
                    r1_bf16_0[_lp] = *(uint32_t*)&_bf2;
                }
                float r2_0[16];
                r2_0[0] = r1_0[0] - __uint_as_float(r1_bf16_0[0] << 16);
                r2_0[1] = r1_0[1] - __uint_as_float(r1_bf16_0[0] & mask_hi);
                r2_0[2] = r1_0[2] - __uint_as_float(r1_bf16_0[1] << 16);
                r2_0[3] = r1_0[3] - __uint_as_float(r1_bf16_0[1] & mask_hi);
                r2_0[4] = r1_0[4] - __uint_as_float(r1_bf16_0[2] << 16);
                r2_0[5] = r1_0[5] - __uint_as_float(r1_bf16_0[2] & mask_hi);
                r2_0[6] = r1_0[6] - __uint_as_float(r1_bf16_0[3] << 16);
                r2_0[7] = r1_0[7] - __uint_as_float(r1_bf16_0[3] & mask_hi);
                r2_0[8] = r1_0[8] - __uint_as_float(r1_bf16_0[4] << 16);
                r2_0[9] = r1_0[9] - __uint_as_float(r1_bf16_0[4] & mask_hi);
                r2_0[10] = r1_0[10] - __uint_as_float(r1_bf16_0[5] << 16);
                r2_0[11] = r1_0[11] - __uint_as_float(r1_bf16_0[5] & mask_hi);
                r2_0[12] = r1_0[12] - __uint_as_float(r1_bf16_0[6] << 16);
                r2_0[13] = r1_0[13] - __uint_as_float(r1_bf16_0[6] & mask_hi);
                r2_0[14] = r1_0[14] - __uint_as_float(r1_bf16_0[7] << 16);
                r2_0[15] = r1_0[15] - __uint_as_float(r1_bf16_0[7] & mask_hi);
                unsigned int r2_bf16_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r2_0[_lp*2 + 0], r2_0[_lp*2+1 + 0]));
                    r2_bf16_0[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)((cb + 16) / 64 * 16384 + k_row * 128 + (cb + 16) % 64 * 2 ^ ((cb + 16) / 64 * 16384 + k_row * 128 + (cb + 16) % 64 * 2 >> 7 & 7) << 4))), "r"(tmem_load_0_bf16_0[0]), "r"(tmem_load_0_bf16_0[1]), "r"(tmem_load_0_bf16_0[2]), "r"(tmem_load_0_bf16_0[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)((cb + 16 + 8) / 64 * 16384 + k_row * 128 + (cb + 16 + 8) % 64 * 2 ^ ((cb + 16 + 8) / 64 * 16384 + k_row * 128 + (cb + 16 + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(tmem_load_0_bf16_0[4]), "r"(tmem_load_0_bf16_0[5]), "r"(tmem_load_0_bf16_0[6]), "r"(tmem_load_0_bf16_0[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)((cb + 16) / 64 * 16384 + k_row * 128 + (cb + 16) % 64 * 2 ^ ((cb + 16) / 64 * 16384 + k_row * 128 + (cb + 16) % 64 * 2 >> 7 & 7) << 4))), "r"(r1_bf16_0[0]), "r"(r1_bf16_0[1]), "r"(r1_bf16_0[2]), "r"(r1_bf16_0[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)((cb + 16 + 8) / 64 * 16384 + k_row * 128 + (cb + 16 + 8) % 64 * 2 ^ ((cb + 16 + 8) / 64 * 16384 + k_row * 128 + (cb + 16 + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(r1_bf16_0[4]), "r"(r1_bf16_0[5]), "r"(r1_bf16_0[6]), "r"(r1_bf16_0[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)((cb + 16) / 64 * 16384 + k_row * 128 + (cb + 16) % 64 * 2 ^ ((cb + 16) / 64 * 16384 + k_row * 128 + (cb + 16) % 64 * 2 >> 7 & 7) << 4))), "r"(r2_bf16_0[0]), "r"(r2_bf16_0[1]), "r"(r2_bf16_0[2]), "r"(r2_bf16_0[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)((cb + 16 + 8) / 64 * 16384 + k_row * 128 + (cb + 16 + 8) % 64 * 2 ^ ((cb + 16 + 8) / 64 * 16384 + k_row * 128 + (cb + 16 + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(r2_bf16_0[4]), "r"(r2_bf16_0[5]), "r"(r2_bf16_0[6]), "r"(r2_bf16_0[7]) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(hs_ready_addr);
                tmem_ld_x16(&h_a[0], taddr + (unsigned int)(h_row_base + cb));
                tmem_ld_x16(&d_a[0], taddr + 128 + (unsigned int)(h_row_base + cb));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                tmem_ld_x16(&h_b[0], taddr + (unsigned int)(h_row_base + cb) + 16);
                tmem_ld_x16(&d_b[0], taddr + 128 + (unsigned int)(h_row_base + cb) + 16);
                #pragma unroll
                for (int j_2 = 0; j_2 < 16; j_2++) {
                    h_a[j_2] = h_a[j_2] + d_a[j_2];
                }
                #pragma unroll
                for (int j_3 = 0; j_3 < 16; j_3++) {
                    h_a[j_3] = h_a[j_3] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)(h_row_base + cb), h_a);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                #pragma unroll
                for (int j_4 = 0; j_4 < 16; j_4++) {
                    h_b[j_4] = h_b[j_4] + d_b[j_4];
                }
                #pragma unroll
                for (int j_5 = 0; j_5 < 16; j_5++) {
                    h_b[j_5] = h_b[j_5] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)(h_row_base + cb) + 16, h_b);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                mbarrier_wait(vp_ready_addr, _phase_vp_ready_0);
                _phase_vp_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int trow = row0 + c * 64 + t_row;
                long long u_base = ((long long)trow * (long long)num_heads + (long long)head) * 128ll + (long long)v0;
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&vp_a[0])), "=r"(*reinterpret_cast<uint32_t*>(&vp_a[1])), "=r"(*reinterpret_cast<uint32_t*>(&vp_a[2])), "=r"(*reinterpret_cast<uint32_t*>(&vp_a[3])), "=r"(*reinterpret_cast<uint32_t*>(&vp_a[4])), "=r"(*reinterpret_cast<uint32_t*>(&vp_a[5])), "=r"(*reinterpret_cast<uint32_t*>(&vp_a[6])), "=r"(*reinterpret_cast<uint32_t*>(&vp_a[7])), "=r"(*reinterpret_cast<uint32_t*>(&vp_a[8])), "=r"(*reinterpret_cast<uint32_t*>(&vp_a[9])), "=r"(*reinterpret_cast<uint32_t*>(&vp_a[10])), "=r"(*reinterpret_cast<uint32_t*>(&vp_a[11])), "=r"(*reinterpret_cast<uint32_t*>(&vp_a[12])), "=r"(*reinterpret_cast<uint32_t*>(&vp_a[13])), "=r"(*reinterpret_cast<uint32_t*>(&vp_a[14])), "=r"(*reinterpret_cast<uint32_t*>(&vp_a[15]))
                    : "r"(taddr + 64 + (unsigned int)(vp_row_base + cb)));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col_1 = cb + lane_half * 16;
                float uv_1[16];
                uv_1[0] = 0.0f;
                uv_1[1] = 0.0f;
                uv_1[2] = 0.0f;
                uv_1[3] = 0.0f;
                uv_1[4] = 0.0f;
                uv_1[5] = 0.0f;
                uv_1[6] = 0.0f;
                uv_1[7] = 0.0f;
                uv_1[8] = 0.0f;
                uv_1[9] = 0.0f;
                uv_1[10] = 0.0f;
                uv_1[11] = 0.0f;
                uv_1[12] = 0.0f;
                uv_1[13] = 0.0f;
                uv_1[14] = 0.0f;
                uv_1[15] = 0.0f;
                if (use_u == 1) {
                    {
                        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(u + u_base + (long long)col_1);
                        uint4 _vld_0[2];
                        #pragma unroll
                        for (int _blk = 0; _blk < 2; _blk++) {
                            _vld_0[_blk] = _vptr_0[_blk];
                            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&uv_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                                (&uv_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                }
                float bvv_1[16];
                #pragma unroll
                for (int j_1_2 = 0; j_1_2 < 16; j_1_2++) {
                    bvv_1[j_1_2] = uv_1[j_1_2] - vp_a[j_1_2];
                }
                unsigned int tmem_load_0_bf16_1[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_1[_lp*2 + 0], bvv_1[_lp*2+1 + 0]));
                    tmem_load_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float r1_1[16];
                r1_1[0] = bvv_1[0] - __uint_as_float(tmem_load_0_bf16_1[0] << 16);
                r1_1[1] = bvv_1[1] - __uint_as_float(tmem_load_0_bf16_1[0] & mask_hi);
                r1_1[2] = bvv_1[2] - __uint_as_float(tmem_load_0_bf16_1[1] << 16);
                r1_1[3] = bvv_1[3] - __uint_as_float(tmem_load_0_bf16_1[1] & mask_hi);
                r1_1[4] = bvv_1[4] - __uint_as_float(tmem_load_0_bf16_1[2] << 16);
                r1_1[5] = bvv_1[5] - __uint_as_float(tmem_load_0_bf16_1[2] & mask_hi);
                r1_1[6] = bvv_1[6] - __uint_as_float(tmem_load_0_bf16_1[3] << 16);
                r1_1[7] = bvv_1[7] - __uint_as_float(tmem_load_0_bf16_1[3] & mask_hi);
                r1_1[8] = bvv_1[8] - __uint_as_float(tmem_load_0_bf16_1[4] << 16);
                r1_1[9] = bvv_1[9] - __uint_as_float(tmem_load_0_bf16_1[4] & mask_hi);
                r1_1[10] = bvv_1[10] - __uint_as_float(tmem_load_0_bf16_1[5] << 16);
                r1_1[11] = bvv_1[11] - __uint_as_float(tmem_load_0_bf16_1[5] & mask_hi);
                r1_1[12] = bvv_1[12] - __uint_as_float(tmem_load_0_bf16_1[6] << 16);
                r1_1[13] = bvv_1[13] - __uint_as_float(tmem_load_0_bf16_1[6] & mask_hi);
                r1_1[14] = bvv_1[14] - __uint_as_float(tmem_load_0_bf16_1[7] << 16);
                r1_1[15] = bvv_1[15] - __uint_as_float(tmem_load_0_bf16_1[7] & mask_hi);
                unsigned int r1_bf16_2[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r1_1[_lp*2 + 0], r1_1[_lp*2+1 + 0]));
                    r1_bf16_2[_lp] = *(uint32_t*)&_bf2;
                }
                float r2_1[16];
                r2_1[0] = r1_1[0] - __uint_as_float(r1_bf16_2[0] << 16);
                r2_1[1] = r1_1[1] - __uint_as_float(r1_bf16_2[0] & mask_hi);
                r2_1[2] = r1_1[2] - __uint_as_float(r1_bf16_2[1] << 16);
                r2_1[3] = r1_1[3] - __uint_as_float(r1_bf16_2[1] & mask_hi);
                r2_1[4] = r1_1[4] - __uint_as_float(r1_bf16_2[2] << 16);
                r2_1[5] = r1_1[5] - __uint_as_float(r1_bf16_2[2] & mask_hi);
                r2_1[6] = r1_1[6] - __uint_as_float(r1_bf16_2[3] << 16);
                r2_1[7] = r1_1[7] - __uint_as_float(r1_bf16_2[3] & mask_hi);
                r2_1[8] = r1_1[8] - __uint_as_float(r1_bf16_2[4] << 16);
                r2_1[9] = r1_1[9] - __uint_as_float(r1_bf16_2[4] & mask_hi);
                r2_1[10] = r1_1[10] - __uint_as_float(r1_bf16_2[5] << 16);
                r2_1[11] = r1_1[11] - __uint_as_float(r1_bf16_2[5] & mask_hi);
                r2_1[12] = r1_1[12] - __uint_as_float(r1_bf16_2[6] << 16);
                r2_1[13] = r1_1[13] - __uint_as_float(r1_bf16_2[6] & mask_hi);
                r2_1[14] = r1_1[14] - __uint_as_float(r1_bf16_2[7] << 16);
                r2_1[15] = r1_1[15] - __uint_as_float(r1_bf16_2[7] & mask_hi);
                unsigned int r2_bf16_2[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r2_1[_lp*2 + 0], r2_1[_lp*2+1 + 0]));
                    r2_bf16_2[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv0_addr + (unsigned int)(col_1 / 64 * 8192 + t_row * 128 + col_1 % 64 * 2 ^ (col_1 / 64 * 8192 + t_row * 128 + col_1 % 64 * 2 >> 7 & 7) << 4))), "r"(tmem_load_0_bf16_1[0]), "r"(tmem_load_0_bf16_1[1]), "r"(tmem_load_0_bf16_1[2]), "r"(tmem_load_0_bf16_1[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv0_addr + (unsigned int)((col_1 + 8) / 64 * 8192 + t_row * 128 + (col_1 + 8) % 64 * 2 ^ ((col_1 + 8) / 64 * 8192 + t_row * 128 + (col_1 + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(tmem_load_0_bf16_1[4]), "r"(tmem_load_0_bf16_1[5]), "r"(tmem_load_0_bf16_1[6]), "r"(tmem_load_0_bf16_1[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv1_addr + (unsigned int)(col_1 / 64 * 8192 + t_row * 128 + col_1 % 64 * 2 ^ (col_1 / 64 * 8192 + t_row * 128 + col_1 % 64 * 2 >> 7 & 7) << 4))), "r"(r1_bf16_2[0]), "r"(r1_bf16_2[1]), "r"(r1_bf16_2[2]), "r"(r1_bf16_2[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv1_addr + (unsigned int)((col_1 + 8) / 64 * 8192 + t_row * 128 + (col_1 + 8) % 64 * 2 ^ ((col_1 + 8) / 64 * 8192 + t_row * 128 + (col_1 + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(r1_bf16_2[4]), "r"(r1_bf16_2[5]), "r"(r1_bf16_2[6]), "r"(r1_bf16_2[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv2_addr + (unsigned int)(col_1 / 64 * 8192 + t_row * 128 + col_1 % 64 * 2 ^ (col_1 / 64 * 8192 + t_row * 128 + col_1 % 64 * 2 >> 7 & 7) << 4))), "r"(r2_bf16_2[0]), "r"(r2_bf16_2[1]), "r"(r2_bf16_2[2]), "r"(r2_bf16_2[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv2_addr + (unsigned int)((col_1 + 8) / 64 * 8192 + t_row * 128 + (col_1 + 8) % 64 * 2 ^ ((col_1 + 8) / 64 * 8192 + t_row * 128 + (col_1 + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(r2_bf16_2[4]), "r"(r2_bf16_2[5]), "r"(r2_bf16_2[6]), "r"(r2_bf16_2[7]) : "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(bv_ready_addr);
                mbarrier_wait(h_ready_addr, _phase_h_ready_0);
                _phase_h_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
            }
            int slot = out_index[seg];
            if (slot >= 0) {
                long long tile = (long long)((head / out_hg * out_n + slot) * out_hg + head % out_hg) * (long long)tile_stride * 16384ll;
                float _tmem_load_0[16];
                tmem_ld_x16(&_tmem_load_0[0], taddr + (unsigned int)(h_row_base + cb));
                float _tmem_load_36_0[16];
                tmem_ld_x16(&_tmem_load_36_0[0], taddr + 128 + (unsigned int)(h_row_base + cb));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_1_2_1 = 0; j_1_2_1 < 16; j_1_2_1++) {
                    _tmem_load_0[j_1_2_1] = _tmem_load_0[j_1_2_1] + _tmem_load_36_0[j_1_2_1];
                }
                if (write_transposed == 1) {
                    long long row_at_1_0 = tile + (long long)(k_row * 128 + v0 + cb);
                    {
                        float4 _v4 = make_float4(_tmem_load_0[0 + 0], _tmem_load_0[0 + 1], _tmem_load_0[0 + 2], _tmem_load_0[0 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_0) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_0[4 + 0], _tmem_load_0[4 + 1], _tmem_load_0[4 + 2], _tmem_load_0[4 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_0 + 4ll) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_0[8 + 0], _tmem_load_0[8 + 1], _tmem_load_0[8 + 2], _tmem_load_0[8 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_0 + 8ll) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_0[12 + 0], _tmem_load_0[12 + 1], _tmem_load_0[12 + 2], _tmem_load_0[12 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_0 + 12ll) = _v4;
                    }
                } else {
                    #pragma unroll
                    for (int j_1_3 = 0; j_1_3 < 16; j_1_3++) {
                        *(reinterpret_cast<float*>(out) + (tile + (long long)((v0 + cb + j_1_3) * 128 + k_row))) = _tmem_load_0[j_1_3];
                    }
                }
                float _tmem_load_1[16];
                tmem_ld_x16(&_tmem_load_1[0], taddr + (unsigned int)(h_row_base + cb) + 16);
                float _tmem_load_36_1[16];
                tmem_ld_x16(&_tmem_load_36_1[0], taddr + 128 + (unsigned int)(h_row_base + cb) + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_1_4 = 0; j_1_4 < 16; j_1_4++) {
                    _tmem_load_1[j_1_4] = _tmem_load_1[j_1_4] + _tmem_load_36_1[j_1_4];
                }
                if (write_transposed == 1) {
                    long long row_at_1_1 = tile + (long long)(k_row * 128 + v0 + cb + 16);
                    {
                        float4 _v4 = make_float4(_tmem_load_1[0 + 0], _tmem_load_1[0 + 1], _tmem_load_1[0 + 2], _tmem_load_1[0 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_1) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_1[4 + 0], _tmem_load_1[4 + 1], _tmem_load_1[4 + 2], _tmem_load_1[4 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_1 + 4ll) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_1[8 + 0], _tmem_load_1[8 + 1], _tmem_load_1[8 + 2], _tmem_load_1[8 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_1 + 8ll) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_1[12 + 0], _tmem_load_1[12 + 1], _tmem_load_1[12 + 2], _tmem_load_1[12 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_1 + 12ll) = _v4;
                    }
                } else {
                    #pragma unroll
                    for (int j_1_5 = 0; j_1_5 < 16; j_1_5++) {
                        *(reinterpret_cast<float*>(out) + (tile + (long long)((v0 + cb + 16 + j_1_5) * 128 + k_row))) = _tmem_load_1[j_1_5];
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
    // ---- Role: load ----
    if (warp == 8) {
        { // load_main
            int vslice_1 = blockIdx.x % 2;
            int head_1 = blockIdx.x / 2 % num_heads;
            int seg_1 = walk_seg[blockIdx.x / (2 * num_heads)];
            int c0_1 = seq_chunk_start[seg_1];
            int nc_1 = seq_chunk_start[seg_1 + 1] - c0_1;
            int row0_1 = c0_1 * 64;
            unsigned int _phase_w_empty = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int c_1 = 0; c_1 < nc_1; c_1++) {
                    int trow_1 = row0_1 + c_1 * 64;
                    mbarrier_wait(w_empty_addr, _phase_w_empty);
                    mbarrier_arrive_expect_tx(w_full_addr, 16384);
                    #pragma unroll
                    for (int half = 0; half < 2; half++) {
                        tma_3d_gmem2smem(w_panel_addr + (0 * 2 + (unsigned int)half) * 8192, (&w_tma), half * 64, head_1, trow_1, w_full_addr);
                    }
                    if (nc_1 > c_1 + 1) {
                        int trow_next = row0_1 + (c_1 + 1) * 64;
                        #pragma unroll
                        for (int half_1 = 0; half_1 < 2; half_1++) {
                            asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&w_tma))), "r"((int)(half_1 * 64)), "r"((int)(head_1)), "r"((int)(trow_next)) : "memory");
                            asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&kg_tma))), "r"((int)(half_1 * 64)), "r"((int)(head_1)), "r"((int)(trow_next)) : "memory");
                        }
                    }
                    mbarrier_wait(kg_empty_addr, _phase_w_empty);
                    mbarrier_arrive_expect_tx(kg_full_addr, 16384);
                    #pragma unroll
                    for (int half_2 = 0; half_2 < 2; half_2++) {
                        tma_3d_gmem2smem(kg_panel_addr + (0 * 2 + (unsigned int)half_2) * 8192, (&kg_tma), half_2 * 64, head_1, trow_1, kg_full_addr);
                    }
                    _phase_w_empty ^= 1;
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 9) {
        { // mma_main
            int vslice_2 = blockIdx.x % 2;
            int head_2 = blockIdx.x / 2 % num_heads;
            int seg_2 = walk_seg[blockIdx.x / (2 * num_heads)];
            int c0_2 = seq_chunk_start[seg_2];
            int nc_2 = seq_chunk_start[seg_2 + 1] - c0_2;
            int row0_2 = c0_2 * 64;
            unsigned int _phase_w_full = 0;
            unsigned int _phase_hs_ready_0 = 0;
            unsigned int _phase_bv_ready_0 = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int c_2 = 0; c_2 < nc_2; c_2++) {
                    mbarrier_wait(w_full_addr, _phase_w_full);
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
                    tcgen05_commit(w_empty_addr);
                    mbarrier_wait(kg_full_addr, _phase_w_full);
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
                    tcgen05_commit(kg_empty_addr);
                    _phase_w_full ^= 1;
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
